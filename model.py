from collections import defaultdict
from copy import deepcopy
import math

log_0_5 = math.log(0.5)

def logsumexp(*vals):
    shift = max(vals)
    return shift + math.log(sum((math.exp(v - shift) for v in vals)))

class Model:
    """Probabilistic sequence prediction

    Base class.  Derived clases need to implement update() and log_predict().  For efficiency 
    dervied classes should also override the copy method.
    """
    
    def update(self, symbol, weight = 1.0):
        raise NotImplementedError('update() method must be defined for derived class, {}'.format(self.__class__.__name__))        

    def log_predict(self, symbol):
        raise NotImplementedError('log_predict() method must be defined for derived class, {}'.format(self.__class__.__name__))

    def update_seq(self, seq, weight = 1.0):
        """Updates the model with an entire sequence of symbols.
        """
        rv = 0
        for symbol in seq:
            rv += self.update(symbol, weight)
        return rv

    def log_predict_seq(self, seq):
        """Returns the log probability of observing an entire sequence of symbols.

        Note: This is not properly Bayesian.  It does not update the model between symbols.
        """
        rv = 0
        for symbol in seq:
            rv += self.log_predict(symbol)
        return rv

    def predict(self, symbol):
        return math.exp(self.log_predict(symbol))

    def copy(self):
        raise NotImplementedError('copy() method must be defined for derived class, {}'.format(self.__class__.__name__))

class KT(Model):
    """KT Estimator 

    AKA Beta(0.5, 0.5) prior under a binary alphabet.

    alphabet : specifies the symbols in the Dirichlet [default: (0, 1) i.e., a Beta distribution]
    counts : dictionary with initial counts [default: 1/len(alphabet) ]
    """
    def __init__(self, alphabet = (0, 1), counts = None):
        super().__init__()
        if counts:
            self.counts = { a: counts[a] for a in alphabet }
            self.sum_counts = sum(self.counts.values())
        else:
            self.counts = { a: 1.0/len(alphabet) for a in alphabet }
            self.sum_counts = 1.0

    def update(self, symbol, weight = 1.0):
        rv = self.log_predict(symbol)
        self.counts[symbol] += weight
        self.sum_counts += weight
        return rv

    def log_predict(self, symbol):
        return math.log(self.counts[symbol] / self.sum_counts)

    def copy(self):
        cls = self.__class__
        c = cls.__new__(cls)
        c.counts = dict(self.counts)
        c.sum_counts = self.sum_counts
        return c

class SAD(Model):
    """Sparse Adaptive Dirichlet Process

    From M. Hutter, "Sparse Adaptive Dirichlet-Multinomial-like Processes" in JMLR 30:1-28 (2013).

    n : size of alphabet 
    """
    
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.counts = {}
        self.sum_counts = 0

    def update(self, symbol, weight = 1.0):
        rv = self.log_predict(symbol)
        self.counts.setdefault(symbol, 0)
        self.counts[symbol] += weight
        self.sum_counts += weight
        return rv

    def log_predict(self, symbol):
        m = min(len(self.counts), self.sum_counts)
        beta = m / (2 * math.log((self.sum_counts + 1) / m)) if self.sum_counts > 0 else 1

        if symbol in self.counts:
            return math.log(self.counts[symbol] / (self.sum_counts + beta))
        else:
            return math.log(beta / ((self.n - len(self.counts)) * (self.sum_counts + beta)))
        
    def copy(self):
        cls = self.__class__
        c = cls.__new__(cls)
        c.n = self.n
        c.counts = dict(self.counts)
        c.sum_counts = self.sum_counts
        return c
    
class Averager(Model):
    """Average over a set of models

    models: a collection of Models 
    """
    def __init__(self, models):
        super().__init__()

        # Store models with their log probability (initially a uniform prior)
        log_1_over_n = math.log(1.0 / len(models))
        self.models = { m: log_1_over_n for m in models }

        self.log_prob = 0

    def update(self, symbol, weight = 1.0):
        orig_log_prob = self.log_prob

        for m in self.models:
            self.models[m] += m.update(symbol, weight)
        self.log_prob = logsumexp(*self.models.values())

        return self.log_prob - orig_log_prob

    def log_predict(self, symbol):
        return logsumexp(*(m.log_predict(symbol) + lp for m,lp in self.models.items())) - self.log_prob

    def copy(self):
        cls = self.__class__
        c = cls.__new__(cls)
        c.models = { m.copy(): lp for m,lp in self.models.items() }
        c.log_prob = self.log_prob
        return c
    
    def map(self):
        """
        Returns the model that is the maximum a posteriori model for the observed data
        """
        return max(self.models, key=lambda m: self.models[m])

class CTW(Model):
    """Context Tree Weighting

    depth: the depth of history CTW considers conditioning on
    Base: a factory function that can be called to get an instance of a base-level sequence predictor [default: KT]
    mkcontext: a function creating a context from a history [default: last depth symbols padded with 0s]
    """

    def _mkcontext(self, x):
        """Default context function.  
        Uses the the last depth symbols (padded with 0's) as the context.
        """
        padding = self.depth - len(x)
        return [0] * padding + x[-self.depth:]
    
    def __init__(self, depth, Base = KT, history = None, mkcontext = None):
        super().__init__()
        self.depth = depth
        self.Base = Base
        self.history = [] if history is None else history
        self._update_history = history is None
        self.mkcontext = mkcontext if mkcontext else self._mkcontext 

        self.base = self.Base()
        self.base_log_prob = 0
        if self.depth > 0:
            self.children = defaultdict(lambda: CTW(depth = self.depth-1, Base = self.Base, history = self.history))

        self.log_prob = 0

    def update(self, symbol, weight = 1.0):
        orig_log_prob = self.log_prob
        self._update(symbol, self.mkcontext(self.history), weight)

        if self._update_history: self.history.append(symbol)

        return self.log_prob - orig_log_prob
            
    def _update(self, symbol, context, weight):
        self.base_log_prob += self.base.update(symbol, weight)
        if self.depth <= 0:
            self.log_prob = self.base_log_prob
        else:
            child = context[-1]
            self.children[child]._update(symbol, context[:-1], weight)
            self.log_prob = log_0_5 + logsumexp(self.base_log_prob, sum((c.log_prob for c in self.children.values())))

    def log_predict(self, symbol):
        return self._log_predict(symbol, self.mkcontext(self.history))
            
    def _log_predict(self, symbol, context):
        base_predict_logp = self.base.log_predict(symbol)

        if self.depth <= 0:        
            return base_predict_logp
        else:
            children_logp = sum((c.log_prob for c in self.children.values()))

            child = context[-1]
            child_predict_logp = self.children[child]._log_predict(symbol, context[:-1])

            return log_0_5 + logsumexp(self.base_log_prob + base_predict_logp, children_logp + child_predict_logp) - self.log_prob
            
    def log_predict_seq(self, seq):
        rv = 0
        for symbol in seq:
            rv += self._log_predict(symbol, self.mkcontext(self.history))
            self.history.append(symbol)
        self.history[-len(seq):] = []
        return rv

    def copy(self):
        cls = self.__class__
        c = cls.__new__(cls)

        c.depth = self.depth
        c.Base = self.Base
        c._update_history = self._update_history
        if self._update_history:
            c.history = list(self.history)
        else:
            c.history = self.history
        c.mkcontext = self.mkcontext

        c.base = self.base.copy()
        c.base_log_prob = self.base_log_prob
        c.log_prob = self.log_prob
        if self.depth > 0:
            c.children = defaultdict(lambda: CTW(depth = c.depth - 1, Base = c.Base, history = c.history),
                                     ((symbol, child.copy()) for symbol,child in self.children.items()))
        return c

class PTW(Model):
    """Partition Tree Weighting

    height: height of the PTL, can only predict over sequences of length <= 2**height
    Base: a factory function that can be called to get an instance of a base-level sequence predictor [default: KT]
    """

    class Node:
        def __init__(self, Base):
            super().__init__()
            self.Base = Base

            self.base = Base()
            self.base_log_prob = 0
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

        def update(self, symbol, weight):
            # Update the predictor assuming no partitions
            self.base_log_prob += self.base.update(symbol, weight)

            # If this is the first symbol, then there cannot be a partition here
            if not self.t: 
                self.log_prob = self.base_log_prob
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
                    right_log_prob = log_0_5 + logsumexp(self.right_child.base_log_prob, right_log_prob)

                self.log_prob = log_0_5 + logsumexp(self.base_log_prob, self.left_log_prob + right_log_prob)

            self.t += 1

        def log_predict(self, symbol):
            base_lp = self.base_log_prob + self.base.log_predict(symbol)

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
                    right_base_lp = self.right_child.base_log_prob + self.right_child.base.log_predict(symbol)
                    left_lp = self.left_log_prob
                    height_diff = self.height - self.right_child.height - self.right_child._at_splitpoint()

                for i in range(height_diff - 1):
                    right_lp = log_0_5 + logsumexp(right_base_lp, right_lp)

                return log_0_5 + logsumexp(base_lp, left_lp + right_lp) - self.log_prob

    def __init__(self, height, Base = KT):
        self.height = height
        self.Base = Base
        self.tree = self.Node(self.Base)
        self.log_prob = 0

    def update(self, symbol, weight = 1.0):
        orig_log_prob = self.log_prob

        self.tree.update(symbol, weight)
        assert self.height >= self.tree.height

        self.log_prob = self.tree.log_prob
        for i in range(self.height - self.tree.height):
            self.log_prob = log_0_5 + logsumexp(self.tree.base_log_prob, self.log_prob)

        return self.log_prob - orig_log_prob

    def log_predict(self, symbol):
        tree_lp = self.tree.log_prob + self.tree.log_predict(symbol)
        tree_base_lp = self.tree.base_log_prob + self.tree.base.log_predict(symbol)
        
        for i in range(self.height - (self.tree.height + self.tree._at_splitpoint())):
            tree_lp = log_0_5 + logsumexp(tree_base_lp, tree_lp)

        return tree_lp - self.log_prob

    def map(self):
        """Returns a Base model that is the maximum a posteriori predictor for the next symbol
        """
        def _nodes():
            t = self.tree
            left = 0

            while t:
                yield (t.base_log_prob + left, t.base)
                left += t.left_log_prob + log_0_5 
                t = t.right_child

        return max(_nodes(), key = lambda x: x[0])[1]

def CommonHistory(Base):
    """Common History Model

    This is a helper function that allows a single common history to be shared across many models.  The idea
    is to pass it a factory function that returns the model to use.  The factory takes a list as an argument
    which can then be used by 

    Base: a factory function that takes a list and returns a Model that uses that list for its history

    # EXAMPLE: PTW model over CTW with KT estimators at the leaves.  The CTW models will have a common 
    # history so the CTW model started after 4 symbols will start predicting from that context.
    >>> model = CommmonHistory(lambda history: PTW(16, lambda: CTW(4, KT(), history = history)))
    """
    history = []
    model = Base(history)

    def update(symbol, weight = 1.0):
        rv = model.__class__.update(model, symbol, weight)
        history.append(symbol)
        return rv
    model.update = update

    return model
    
class LogStore:
    """Stores a "logarithmic number" of objects.  Keeps more recently added objects.

    The class is also indexable, with newer objects first and the oldest object last.
    
    >>> s = LogStore()
    >>> for i in range(16):
    ...     s.add(i)
    >>> list(s)
    [15, 14, 12, 8, 0]
    >>> s[-1]
    0
    >>> s[0]
    15
    """
    
    def __init__(self):
        self._items = []
        self._save = []

    def __len__(self):
        return len(self._items)

    def __iter__(self):
        for x in self._items:
            yield x

    def __getitem__(self, i):
        return self._items[i]
        
    def add(self, x):
        if not self._items:
            self._items.append(x)
            self._save.append(True)
        else:
            i = 0
            for i in range(len(self)):
                if self._save[i]:
                    self._items[i], x = x, self._items[i]
                    self._save[i] = False
                else:
                    self._items[i] = x
                    self._save[i] = True
                    return
            self._items.append(x)
            self._save.append(True)

class FMN(PTW):
    """Forget Me Not

    PTW-based model where the base model is an average over high probability models from the past.

    PTW(height, Base)

    height: height of the PTL, can only predict over sequences of length <= 2**height
    Base: a factory function that can be called to get an instance of a base-level 
        sequence predictor [default: KT]
    ModelStore: a factory function to get a set-like object for storing the models 
        (must support the 'add' method and iteration) [default: LogStore]
    """

    def __init__(self, height, Base = KT, ModelStore = LogStore):
        self.models = ModelStore()
        self.models.add(Base())
        super().__init__(height, self.Base)

    def Base(self):
        return Averager([ m.copy() for m in set(self.models) ])

    def update(self, symbol, weight = 1.0):
        rv = super().update(symbol, weight)
        self.models.add(self.map().map().copy())
        return rv
            
class Factored(Model):
    """Factored model with independent models that repeat on a fixed period.

    This is mainly for binary models over bytes where a separate model is used for
    each bit position

    # Examples
    >>> model = Factored([ KT() for i in range(8) ])
    >>> model = Factored([ CTW(16 + i) for i in range(8) ])
    """
    def __init__(self, factors):
        self.factors = factors
        self.index = -1

    def log_predict(self, symbol):
        return self.factors[self.index].log_predict(symbol)

    def update(self, symbol, weight = 1.0):
        self.index = (self.index + 1) % len(self.factors)
        return self.factors[self.index].update(symbol)

    def log_predict_seq(self, seq):
        rv = 0
        index = self.index
        for symbol in seq:
            index = (index + 1) % len(self.factors)
            rv += self.factors[index].log_predict(symbol)
        return rv

    def copy(self):
        cls = self.__class__
        c = cls.__new__(cls)
        c.factors = [ m.copy() for m in self.factors ]
        c.index = self.index
        return c

        
