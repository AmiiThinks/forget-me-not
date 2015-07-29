from collections import defaultdict
from copy import deepcopy
import math

log_0_5 = math.log(0.5)

def logsumexp(*vals):
    shift = max(vals)
    return shift + math.log(sum((math.exp(v - shift) for v in vals)))

class ProbSeq:
    """Probabilistic sequence prediction

    Base class.  Derived clases need to implement update() and log_predict().
    """
    
    def __init__(self):
        self.log_prob = 0
        self.history = []
        self.update_history = True

    def use_history(self, history, update_history = False):
        self.history = history
        self.update_history = update_history
            
    def update(self, symbol, weight = 1.0):
        raise NotImplementedError('update() method must be defined for derived class, {}'.format(self.__class__.__name__))

    def log_predict(self, symbol):
        raise NotImplementedError('log_predict() method must be defined for derived class, {}'.format(self.__class__.__name__))

    def reset_history(self):
        assert self.update_history # Should not be resetting histories that are maintained outside the class
        self.history[:] = []

    def update_seq(self, seq, weight = 1.0):
        orig_log_prob = self.log_prob
        rv = 0
        for symbol in seq:
            rv += self.update(symbol, weight)
        return self.log_prob - orig_log_prob

    def log_predict_seq(self, seq):
        rv = 0
        for symbol in seq:
            rv += self.log_predict(symbol)
        return rv

    @property
    def prob(self):
        return math.exp(self.logprob)
    
    @property
    def total_loss(self):
        return self.log_prob

    def predict(self, data):
        return math.exp(self.log_predict(data))

class KT(ProbSeq):
    def __init__(self, alphabet = (0, 1), counts = None):
        super().__init__()
        if counts:
            self.counts = { a: counts[a] for a in alphabet }
        else:
            self.counts = { a: 1.0/len(alphabet) for a in alphabet }

    def update(self, symbol, weight = 1.0):
        rv = self.log_predict(symbol)
        self.counts[symbol] += weight
        self.log_prob += rv
        return rv

    def log_predict(self, symbol):
        return math.log(self.counts[symbol]) - math.log(sum(self.counts.values()))

class SAD(ProbSeq):
    """Sparse Adaptive Dirichlet Process

    From M. Hutter, "Sparse Adaptive Dirichlet-Multinomial-like Processes" in JMLR 30:1-28 (2013).

    n : size of alphabet 
    """
    
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.counts = {}
        self.t = 0

    def update(self, symbol, weight = 1.0):
        rv = self.log_predict(symbol)
        self.log_prob += rv
        self.counts.setdefault(symbol, 0)
        self.counts[symbol] += weight
        self.t += weight
        return rv

    def log_predict(self, symbol):
        m = min(len(self.counts), self.t)
        beta = m / (2 * math.log((self.t + 1) / m)) if self.t > 0 else 1

        if symbol in self.counts:
            return math.log(self.counts[symbol] / (self.t + beta))
        else:
            return math.log(beta / ((self.n - len(self.counts)) * (self.t + beta)))
        
class Averager(ProbSeq):
    def __init__(self, models):
        super().__init__()
        self.models = models
        self.log_1_over_n = math.log(1.0 / len(self.models))

        # TODO: hmmm... this is needed to make things like FMN work, but it feels wrong
        # What about the history?  The model needs to have its history reset too.
        for m in self.models:
            m.log_prob = 0

    def update(self, symbol, weight = 1.0):
        orig_log_prob = self.log_prob

        for m in self.models:
            m.update(symbol, weight)
        self.log_prob = self.log_1_over_n + logsumexp(*(m.log_prob for m in self.models))

        return self.log_prob - orig_log_prob

    def log_predict(self, symbol):
        return self.log_1_over_n + logsumexp(*(m.log_predict(symbol) + m.log_prob for m in self.models)) - self.log_prob

    def map(self):
        """
        Returns the model that is the maximum a posteriori model for the observed data
        """
        return max(self.models, key = lambda m: m.log_prob)

class CTW(ProbSeq):
    """
    Context Tree Weighting

    CTW(depth, Base)

    depth: the depth of history CTW considers conditioning on
    Base: a factory function that can be called to get an instance of a base-level sequence predictor [default: KT]
    """

    def _mkcontext(self, x):
        """
        Default context function.  
        Uses the the last depth symbols (padded with 0's) as the context.
        """
        padding = self.depth - len(x)
        return [0] * padding + x[-self.depth:]
    
    def __init__(self, depth, Base = KT, mkcontext = None):
        super().__init__()
        self.depth = depth
        self.Base = Base
        self.mkcontext = mkcontext if mkcontext else self._mkcontext 

        self.base = self.Base()
        if self.depth > 0:
            self.children = defaultdict(lambda: CTW(depth = self.depth-1, Base = self.Base))
        self.reset_history()

    def update(self, symbol, weight = 1.0):
        orig_log_prob = self.log_prob
        self._update(symbol, self.mkcontext(self.history), weight)

        # Update History
        if self.update_history: self.history.append(symbol)

        return self.log_prob - orig_log_prob
            
    def _update(self, symbol, context, weight):
        self.base.update(symbol, weight)
        if self.depth <= 0:
            self.log_prob = self.base.log_prob
        else:
            child = context[-1]
            self.children[child]._update(symbol, context[:-1], weight)
            self.log_prob = log_0_5 + logsumexp(self.base.log_prob, sum((c.log_prob for c in self.children.values())))

    def log_predict(self, symbol):
        return self._log_predict(symbol, self.mkcontext(self.history))
            
    def _log_predict(self, symbol, context):
        base_predict_logp = self.base.log_predict(symbol)

        if self.depth <= 0:        
            return base_predict_logp
        else:
            base_logp = self.base.log_prob
            children_logp = sum((c.log_prob for c in self.children.values()))

            child = context[-1]
            child_predict_logp = self.children[child]._log_predict(symbol, context[:-1])

            return log_0_5 + logsumexp(base_logp + base_predict_logp, children_logp + child_predict_logp) - self.log_prob
            
    def log_predict_seq(self, seq):
        rv = 0
        history = self.history[:]
        for symbol in seq:
            rv += self._log_predict(symbol, self.mkcontext(history))
            history.append(symbol)
        return rv
    

class PTW(ProbSeq):
    """
    Partition Tree Weighting

    PTW(height, Base)

    height: height of the PTL, can only predict over sequences of length <= 2**height
    Base: a factory function that can be called to get an instance of a base-level sequence predictor [default: KT]
    """
    class Node:
        def __init__(self, Base):
            super().__init__()
            self.Base = Base

            self.update_history = False

            self.base = Base()
            self.height = 0
            self.log_prob = 0
            self.left_log_prob = 0
            self.right_child = None
            self.t = 0

        def _at_splitpoint(self):
            """
            Checks if self.t is a power of 2
            """
            t = self.t
            return not (t & (t-1))

        def update(self, symbol, weight=1.0):
            # Update the predictor assuming no partitions
            self.base.update(symbol, weight)

            # If this is the first symbol, then there cannot be a partition here
            if not self.t: 
                self.log_prob = self.base.log_prob
            else:
                # If crossing a partition then promote this node to be one level higher
                # - its left child is its previous self, so store its log probability as the left_log_prob
                # - its right child is an emtpy base learner
                if self._at_splitpoint():
                    self.left_log_prob = self.log_prob
                    self.right_child = self.__class__(self.Base)
                    self.height += 1

                # Add the symbol to the right child, and update the node's log probability
                self.right_child.update(symbol, weight) 

                # Update log_prob, accounting for unrepresented nodes between this node and the right child
                right_log_prob = self.right_child.log_prob

                for i in range(self.height - self.right_child.height - 1):
                    right_log_prob = log_0_5 + logsumexp(self.right_child.base.log_prob, right_log_prob)

                self.log_prob = log_0_5 + logsumexp(self.base.log_prob, self.left_log_prob + right_log_prob)

            self.t += 1

        def log_predict(self, symbol):
            base_lp = self.base.log_prob + self.base.log_predict(symbol)

            if not self.t:
                return base_lp - self.log_prob
            else:
                if self._at_splitpoint():
                    right_lp = self.Base().log_predict(symbol)
                    right_base_lp = right_lp
                    left_lp = self.log_prob
                    height_diff = self.height + 1
                else:
                    right_lp = self.right_child.log_prob + self.right_child.log_predict(symbol)
                    right_base_lp = self.right_child.base.log_prob + self.right_child.base.log_predict(symbol)
                    left_lp = self.left_log_prob
                    height_diff = self.height - self.right_child.height - self.right_child._at_splitpoint()

                for i in range(height_diff - 1):
                    right_lp = log_0_5 + logsumexp(right_base_lp, right_lp)

                return log_0_5 + logsumexp(base_lp, left_lp + right_lp) - self.log_prob

    def __init__(self, height, Base = KT):
        self.height = height
        self.Base = Base

        self.log_prob = 0
        self.tree = self.Node(self.Base)

    def update(self, symbol, weight = 1.0):
        orig_log_prob = self.log_prob

        self.tree.update(symbol, weight)
        assert self.height >= self.tree.height

        self.log_prob = self.tree.log_prob
        for i in range(self.height - self.tree.height):
            self.log_prob = log_0_5 + logsumexp(self.tree.base.log_prob, self.log_prob)

        return self.log_prob - orig_log_prob

    def log_predict(self, symbol):
        tree_lp = self.tree.log_prob + self.tree.log_predict(symbol)
        tree_base_lp = self.tree.base.log_prob + self.tree.base.log_predict(symbol)
        
        for i in range(self.height - (self.tree.height + self.tree._at_splitpoint())):
            tree_lp = log_0_5 + logsumexp(tree_base_lp, tree_lp)

        return tree_lp - self.log_prob

    def map(self):
        """
        Returns a Base learner that is the maximum a posteriori predictor for the next symbol
        """
        def _nodes():
            t = self.tree
            left = 0

            while t:
                yield (t.base.log_prob + left, t.base)
                left += t.left_log_prob + log_0_5 
                t = t.right_child

        return max(_nodes(), key = lambda x: x[0])[1]

class LogStore:
    """
    Stores a "logarithmic number" of objects.  Keeps more recently added objects.

    >>> s = LogStore()
    >>> for i in range(16):
    ...     s.add(i)
    >>> list(s)
    [15, 14, 12, 8, 0]
    """
    
    def __init__(self):
        self._items = []
        self._save = []

    def __len__(self):
        return len(self._items)

    def __iter__(self):
        yield from self._items

    def __getitem__(self, i):
        return self._items[i]
        
    def add(self, m):
        if not self._items:
            self._items.append(m)
            self._save.append(True)
        else:
            i = 0
            for i in range(len(self)):
                if self._save[i]:
                    self._items[i], m = m, self._items[i]
                    self._save[i] = False
                else:
                    self._items[i] = m
                    self._save[i] = True
                    return
            self._items.append(m)
            self._save.append(True)

class FMN(PTW):
    def __init__(self, height, Base = KT, ModelStore = LogStore):
        self.models = ModelStore()
        self.models.add(Base())
        super().__init__(height, self.Base)

    def Base(self):
        return Averager([ deepcopy(m) for m in set(self.models) ])

    def update(self, symbol, weight = 1.0):
        rv = super().update(symbol, weight)
        self.models.add(deepcopy(self.map().map()))
        return rv

def Factorize(Base, nfactors):
    class Factorized(ProbSeq):
        def __init__(self):
            super().__init__()
            self.models = [ Base() for i in range(nfactors) ]
            for m in self.models:
                m.use_history(self.history)

        def use_history(self, history):
            super().use_history(history)
            for m in self.models:
                m.use_history(self.history)
                
        def update(self, symbol, weight = 1.0):
            model = self.models[len(self.history) % nfactors]
            rv = model.update(symbol, weight)
            self.log_prob += rv
            
            # Update History
            if self.update_history: self.history.append(symbol)

            return rv

        def log_predict(self, symbol):
            model = self.models[len(self.history) % nfactors]
            return model.log_predict(symbol)

        def log_predict_seq(self, seq):
            rv = 0
            rollback = 0
            for symbol in seq:
                rv += self.log_predict(symbol)
                self.history.append(symbol)
                rollback += 1
            self.history[-rollback:] = []
            return rv

    return Factorize
