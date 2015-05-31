import math

log_0_5 = math.log(0.5)


def logsumexp(*vals):
    shift = max(vals)
    return shift + math.log(sum((math.exp(v - shift) for v in vals)))


class Model:
    """Probabilistic sequence prediction

    Base class. Derived clases need to implement update(), log_predict(), and
    optionally copy() to use the Model with particular meta-models.
    """

    __slots__ = []

    def update(self, symbol, weight=1.0):
        """Updates the model with the symbol and returns the log probability
        of that symbol being next.

        weight: specifies how much weight to place on the symbol
                [default: 1]
        """
        raise NotImplementedError(
            'update() method must be defined for derived class, {}'
            .format(self.__class__.__name__))

    def log_predict(self, symbol):
        """Returns the log probability of observing the symbol next.
        """
        raise NotImplementedError(
            'log_predict() method must be defined for derived class, '
            '{}'.format(self.__class__.__name__))

    def update_seq(self, seq, weight=1.0):
        """Updates the model with an entire sequence of symbols.
        """
        rv = 0
        for symbol in seq:
            rv += self.update(symbol, weight)
        return rv

    def log_predict_seq(self, seq):
        """Returns the log probability of observing an entire sequence of
        symbols. This is not properly Bayesian! It does not update the model
        between symbols.
        """
        rv = 0
        for symbol in seq:
            rv += self.log_predict(symbol)
        return rv

    def predict(self, symbol):
        """Returns the probability of observing the symbol next.
        """
        return math.exp(self.log_predict(symbol))

    def copy(self):
        """Create a deep copy of the model.
        """
        raise NotImplementedError(
            'copy() method must be defined for derived class, {}'
            .format(self.__class__.__name__))


class KT(Model):
    """KT Estimator

    AKA Beta(0.5, 0.5) prior under a binary alphabet.

    alphabet: specifies the symbols in the Dirichlet [default: (0,1)]
    counts: dictionary with initial counts [default: 1/len(alphabet)]
    """

    __slots__ = ["counts", "sum_counts"]

    def __init__(self, alphabet=(0, 1), counts=None):
        super().__init__()
        if counts:
            self.counts = {a: counts[a] for a in alphabet}
        else:
            self.counts = {a: 1.0/len(alphabet) for a in alphabet}
        self.sum_counts = sum(self.counts.values())

    def update(self, symbol, weight=1.0):
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


class KTBinary(Model):
    """KT Estimator, specialized for binary alphabets.

    AKA Beta(0.5, 0.5) prior.

    counts: dictionary with initial counts [default: 1/2]
    """

    __slots__ = ["counts"]

    def __init__(self, counts=None):
        super().__init__()
        self.counts = [counts[0], counts[1]] if counts else [0.5, 0.5]

    def update(self, symbol, weight=1.0):
        rv = self.log_predict(symbol)
        self.counts[symbol] += weight
        return rv

    def log_predict(self, symbol):
        return math.log(self.counts[symbol] / (self.counts[0]+self.counts[1]))

    def copy(self):
        cls = self.__class__
        c = cls.__new__(cls)
        c.counts = list(self.counts)
        return c


class SAD(Model):
    """Sparse Adaptive Dirichlet Process

    From M. Hutter, "Sparse Adaptive Dirichlet-Multinomial-like Processes"
    in JMLR 30:1-28 (2013).

    n : size of alphabet
    """

    def __init__(self, n):
        super().__init__()
        self.n = n
        self.counts = {}
        self.sum_counts = 0

    def update(self, symbol, weight=1.0):
        rv = self.log_predict(symbol)
        self.counts.setdefault(symbol, 0)
        self.counts[symbol] += weight
        self.sum_counts += weight
        return rv

    def log_predict(self, symbol):
        m = min(len(self.counts), self.sum_counts)
        if self.sum_counts > 0:
            beta = m / (2 * math.log((self.sum_counts + 1) / m))
        else:
            beta = 1

        if symbol in self.counts:
            return math.log(self.counts[symbol] / (self.sum_counts + beta))
        else:
            return math.log(beta / ((self.n - len(self.counts)) *
                                    (self.sum_counts + beta)))

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
        self.models = {m: log_1_over_n for m in models}

        self.log_prob = 0

    def update(self, symbol, weight=1.0):
        orig_log_prob = self.log_prob

        for m in self.models:
            self.models[m] += m.update(symbol, weight)
        self.log_prob = logsumexp(*self.models.values())

        return self.log_prob - orig_log_prob

    def log_predict(self, symbol):
        return logsumexp(*(m.log_predict(symbol) + lp
                           for (m, lp) in self.models.items())) - self.log_prob

    def copy(self):
        cls = self.__class__
        c = cls.__new__(cls)
        c.models = {m.copy(): lp for (m, lp) in self.models.items()}
        c.log_prob = self.log_prob
        return c

    def map(self):
        """Returns the model that is the maximum a posteriori model for the
        observed data.
        """
        return max(self.models, key=lambda m: self.models[m])


class CTW(Model):
    """Context Tree Weighting

    From Willems et al., "The Context-Tree Weighting Method: Basic Properties"
    in IEEE Transactions on Information Theory 41 (1995).

    depth: the depth of history CTW considers conditioning on
    model_factory: a factory function that can be called to get an instance of
        a base-level sequence predictor [default: KT]
    mkcontext: a function creating a context from a history [default: last
        depth symbols padded with 0s]
    binary_context: specifies whether the contexts are binary [default: True]
    history: a history to use to form the context, if not provided it will
        make its own [default: None]
    """
    class Node:
        """CTW Node
        Handles arbitary contexts using a dictionary to index the children.
        """
        __slots__ = ['base', 'base_log_prob', 'children_log_prob', 'log_prob',
                     'node_factory', 'children', '_refcount']

        # Class for storing child nodes
        # - must return None if no child exists for a particular context symbol
        # - should be able to iterate over children
        # We build this on top of a dictionary
        class _children_store(dict):
            def __getitem__(self, key): return self.get(key)
            def __iter__(self): return self.values()

        def __init__(self, base, node_factory):
            self.base = base
            self.base_log_prob = 0
            self.children_log_prob = 0
            self.log_prob = 0
            self.node_factory = node_factory
            self.children = self._children_store()
            self._refcount = 1

        def update(self, symbol, weight, context):
            orig_log_prob = self.log_prob

            # Update base model
            self.base_log_prob += self.base.update(symbol, weight)

            if context:
                # Get the next symbol and associated child
                cnext = context.pop()
                child = self.children[cnext]

                # If there's no child, make one
                if not child: child = self.children[cnext] = self.node_factory()
                    
                # If the child is shared, copy it
                elif child._refcount > 1:
                    child._refcount -= 1
                    child = self.children[cnext] = child.copy()

                # Update the child node
                self.children_log_prob += child.update(symbol, weight, context)

                # Update our log probability
                self.log_prob = log_0_5 + logsumexp(self.base_log_prob, 
                                                    self.children_log_prob)
            else: 
                # For leaf nodes, the probability comes just from the base model
                self.log_prob = self.base_log_prob

            return self.log_prob - orig_log_prob
            
        def log_predict(self, symbol, context):
            base_log_prob = (self.base_log_prob + 
                             self.base.update(symbol, weight))

            if context:
                cnext = context.pop()
                child = self.children[cnext]
                if not child: child=self.children[cnext]=self.node_factory()

                children_log_prob = (self.children_log_prob + 
                                     child.log_predict(symbol, context))

                return (log_0_5 + 
                        logsumexp(base_log_prob, children_log_prob) - 
                        self.log_prob)
            else:
                return base_log_prob - self.log_prob

        def copy(self):
            cls = self.__class__
            r = cls.__new__(cls)
            
            r.base = self.base.copy()
            r.base_log_prob = self.base_log_prob
            r.children_log_prob = self.children_log_prob
            r.log_prob = self.log_prob
            r.node_factory = self.node_factory
            r.children = self._children_store(self.children)
            r._refcount = 1

            for c in r.children:
                if c: c._refcount += 1

            return r
    
    class BinaryNode(Node):
        """CTW Node for binary contexts

        Stores the children as lists instead of dictionary for speed/memory
        efficiency.
        """
        __slots__ = []
        def _children_store(self, x = None): 
            return list(x) if x else [None, None]

    def _mkcontext(self, x):
        """Default context function.  
        Uses the the last depth symbols (padded with 0's) as the context.
        """
        padding = self.depth - len(x)
        return [0] * padding + x[-self.depth:]
    
    def __init__(self, depth, model_factory=KTBinary, 
                 mkcontext=None, binary_context=True, history=None):
        super().__init__()
        self.depth = depth
        self.history = [] if history is None else history
        self._update_history = history is None
        self.model_factory = model_factory
        self.mkcontext = mkcontext if mkcontext else self._mkcontext 

        if binary_context:
            self.Node = self.__class__.BinaryNode
        else:
            self.Node = self.__class__.Node
        self.size = 0
        self.tree = self.node_factory()

    def node_factory(self):
        """Creates a new node for the CTW tree.
        """
        self.size += 1
        return self.Node(self.model_factory(), self.node_factory)

    def update(self, symbol, weight=1.0):
        context = self.mkcontext(self.history)
        if self._update_history: self.history.append(symbol)
        return self.tree.update(symbol, weight, context)

    def log_predict(self, symbol):
        return self.tree.log_predict(symbol, self.mkcontext(self.history))

    def log_predict_seq(self, seq):
        """Predict a sequence being sure to update the context between 
        predictions.
        """
        rv = 0
        for symbol in seq:
            rv += self.log_predict(symbol)
            self.history.append(symbol)
        self.history[-len(seq):] = []
        return rv
    
    def copy(self):
        cls = self.__class__
        r = cls.__new__(cls)
        r.__dict__.update(self.__dict__)
        r.tree = self.tree.copy()
        return r

class CTW_KT(Model):
    """Context Tree Weighting specialized to binary KT estimators as the
    base model.

    From Willems et al., "The Context-Tree Weighting Method: Basic Properties"
    in IEEE Transactions on Information Theory 41 (1995).

    depth: the depth of history CTW considers conditioning on
    history: a history to use to form the context, if not provided it will 
        make its own [default: None]
    """
    
    class Node:
        __slots__ = ['base_counts', 'base_log_prob', 'children_log_prob',
                     'log_prob', 'children', '_refcount']
                    
        def __init__(self):
            self.base_counts = [0.5, 0.5] 
            self.base_log_prob = 0.0
            self.children_log_prob = 0.0
            self.log_prob = 0.0
            self.children = [None, None]
            self._refcount = 1

        def update_base(self, symbol, weight):
            """Updates the base KT estimator at the node
            """
            self.base_log_prob += \
              math.log(self.base_counts[symbol] /
                       (self.base_counts[0] + self.base_counts[1]))
            self.base_counts[symbol] += weight
            
        def update(self, symbol, weight, context):
            orig_log_prob = self.log_prob

            # Update base model
            self.update_base(symbol, weight)

            if context:
                # Get the next symbol and associated child
                # Copy the child if it's shared before we update it
                cnext = context.pop()
                child = self.children[cnext]
                if not child: child = self.children[cnext] = self.__class__()
                elif child._refcount > 1: 
                    child._refcount -= 1
                    child = self.children[cnext] = child.copy()

                # Update the child node
                self.children_log_prob += child.update(symbol, weight, context)

                # Update our log probability
                self.log_prob = log_0_5 + logsumexp(self.base_log_prob, 
                                                    self.children_log_prob)
            else: 
                # For leaf nodes, the probability comes just from the base model
                self.log_prob = self.base_log_prob

            return self.log_prob - orig_log_prob
            
        def log_predict(self, symbol, context):
            base_log_prob = (self.base_log_prob + 
                             self.base.update(symbol, weight))

            if context:
                cnext = context.pop()
                child = self.children[cnext]
                if not child: child = self.children[cnext] = self.__class__()

                children_log_prob = (self.children_log_prob + 
                                     child.log_predict(symbol, context))

                return (log_0_5 + 
                        logsumexp(base_log_prob, children_log_prob) - 
                        self.log_prob)
            else:
                return base_log_prob - self.log_prob

        def copy(self):
            cls = self.__class__
            r = cls.__new__(cls)

            r.base_counts = list(self.base_counts)
            r.base_log_prob = self.base_log_prob
            r.children_log_prob = self.children_log_prob
            r.log_prob = self.log_prob
            r.children = list(self.children)
            r._refcount = 1

            for c in r.children:
                if c: c._refcount += 1

            return r
    
    def _mkcontext(self, x):
        """Default context function.  
        Uses the the last depth symbols (padded with 0's) as the context.
        """
        padding = self.depth - len(x)
        return [0] * padding + x[-self.depth:]
    
    def __init__(self, depth, history=None):
        super().__init__()
        self.depth = depth
        self.history = [] if history is None else history
        self._update_history = history is None
        self.tree = self.Node()

    def update(self, symbol, weight=1.0):
        context = self._mkcontext(self.history)
        if self._update_history: self.history.append(symbol)
        return self.tree.update(symbol, weight, context)

    def log_predict(self, symbol):
        return self.tree.log_predict(symbol, self._mkcontext(self.history))

    def log_predict_seq(self, seq):
        rv = 0
        for symbol in seq:
            rv += self.log_predict(symbol)
            self.history.append(symbol)
        self.history[-len(seq):] = []
        return rv
    
    def copy(self):
        cls = self.__class__
        r = cls.__new__(cls)
        r.__dict__.update(self.__dict__)
        r.tree = self.tree.copy()
        return r


class PTW(Model):
    """Partition Tree Weighting

    From J. Veness et al., "Partition Tree Weighting" in Data Compression 
    Conference (2013).

    model_factory: a factory function that can be called to get an instance of
        a base-level sequence predictor [default: KTBinary]
    min_partition_length: the minimum length of partitions in the tree, always
        gets rounded up to a power of 2 [default: 1]
    """
    class Node:
        __slots__ = ['base', 'height', 'node_factory', 'count', 'base_log_prob',
                     'left_log_prob', 'log_prob', 'right_child']
        
        def __init__(self, base, height, node_factory):
            self.base = base
            self.height = height
            self.node_factory = node_factory
            self.count = 0
            self.base_log_prob = 0.0
            self.left_log_prob = 0.0
            self.log_prob = 0.0
            self.right_child = None 

        def partition_is_complete(self):
            """Checks if partition is complete by checking if the count is 
            larger than 2 ** height.
            """
            return (self.count >> self.height) > 0

        def update(self, symbol, weight):
            # If partition is complete, then promote this node one level higher
            #   - new left child is the old node
            #   - new right child is a new node
            if self.partition_is_complete():
                self.left_log_prob = self.log_prob
                self.right_child = self.node_factory()
                self.height += 1

            # Update the base model
            self.base_log_prob += self.base.update(symbol, weight)

            # If this node is not a leaf:
            #   - Update the right child
            #   - Determine correct right child log probability accounting for 
            #     unrepresented nodes
            #   - Update own log probability
            if self.right_child:
                self.right_child.update(symbol, weight)

                right_log_prob = self.right_child.log_prob

                # height correction
                for i in range(self.height - self.right_child.height - 1):  
                    right_log_prob = log_0_5 + logsumexp(self.right_child.base_log_prob, right_log_prob)
                
                self.log_prob = log_0_5 + logsumexp(self.base_log_prob, 
                                                    self.left_log_prob + right_log_prob)
            # If this node is a leaf:
            #   - Log probability is just the base model's log probability
            else:
                self.log_prob = self.base_log_prob

            self.count += 1

        def log_predict(self, symbol):
            base_log_prob = self.base_log_prob + self.base.log_predict(symbol)

            if self.partition_is_complete():
                return (log_0_5 + logsumexp(base_log_prob, self.log_prob), base_log_prob, self.height + 1)

            if not self.right_child:
                return base_log_prob, base_log_prob, self.height
            
            right_log_prob, right_base_log_prob, right_height = self.right_child.log_predict(symbol)
            for i in range(self.height - right_height - 1):
                right_log_prob = log_0_5 + logsumexp(right_base_log_prob, right_log_prob)

            return log_0_5 + logsumexp(base_log_prob, self.left_log_prob + right_log_prob), base_log_prob, self.height

        def copy(self, root=False):
            cls = self.__class__
            r = cls.__new__(cls)

            r.height = self.height
            r.node_factory = self.node_factory
            r.count = self.count
            r.base_log_prob = self.base_log_prob
            r.left_log_prob = self.left_log_prob
            r.log_prob = self.log_prob

            r.base = self.base.copy()
            r.right_child = self.base.copy()

            return r
    
    def __init__(self, model_factory=KTBinary, min_partition_length=1):
        self.model_factory = model_factory
        self.min_height = int(math.log(min_partition_length, 2))

        self.log_prob = 0
        self.log_prob_adjustment = 0 # This tracks the adjustment factors for not knowing the sequence length
        self.tree = self.node_factory()

    def node_factory(self):
        return self.Node(self.model_factory(), self.min_height, self.node_factory)
        
    def update(self, symbol, weight=1.0):
        orig_log_prob = self.log_prob

        # Check if changing the height of the tree, and if so record the log probability adjustment
        if self.tree.partition_is_complete():
            self.log_prob_adjustment += self.tree.log_prob - (log_0_5 + logsumexp(self.tree.base_log_prob, self.tree.log_prob))
                
        # Update tree, adjust log probability
        self.tree.update(symbol, weight)
        self.log_prob = self.tree.log_prob + self.log_prob_adjustment

        return self.log_prob - orig_log_prob

    def log_predict(self, symbol):
        # Check if this symbol changes the height of the tree, and calculate the adjustment
        if self.tree.partition_is_complete():
            log_prob_adjustment = self.log_prob_adjustment + \
              self.tree.log_prob - (log_0_5 + logsumexp(self.tree.base_log_prob, self.tree.log_prob))
        else:
            log_prob_adjustment = self.log_prob_adjustment

        # Get the tree prediction and adjust the log probability
        log_prob, _, _ = self.tree.log_predict(symbol)

        return log_prob + log_prob_adjustment - self.log_prob

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

class PTWFixedLength(PTW):
    """Partition Tree Weighting

    From J. Veness et al., "Partition Tree Weighting" in Data Compression 
    Conference (2013).

    length: length of the sequence to predict
    model_factory: a factory function that can be called to get an instance of a 
        base-level sequence predictor [default: KTBinary]
    min_partition_length: the minimum length of partitions in the tree, always
        gets rounded up to a power of 2 [default: 1]
    """

    def __init__(self, length, **kwargs):
        super().__init__(**kwargs)
        self.height = int(math.ceil(math.log(length, 2)))

    def update(self, symbol, weight=1.0):
        orig_log_prob = self.log_prob
        
        # Update tree
        self.tree.update(symbol, weight)

        # Adjust log probability for the fixed height
        self.log_prob = self.tree.log_prob
        for i in range(self.height - self.tree.height):
            self.log_prob = log_0_5 + logsumexp(self.tree.base_log_prob, self.log_prob)
            
        return self.log_prob - orig_log_prob
    
    def log_predict(self, symbol):
        log_prob, base_log_prob, height = self.tree.log_predict(symbol)

        for i in range(self.height - height):
            log_prob = log_0_5 + logsumexp(base_log_prob, log_prob)

        return log_prob - self.log_prob

def CommonHistory(Base):
    """Common History Model

    This is a helper function that allows a single common history to be shared across many models.  The idea
    is to pass it a factory function that returns the model to use.  The factory takes a list as an argument
    which can then be used by 

    Base: a factory function that takes a list and returns a Model that uses that list for its history

    # EXAMPLE: PTW model over CTW with KT estimators over base 10 digits at the leaves.  
    # The CTW models will have a common history so the CTW model started after 4 symbols will start
    # predicting from that context.
    >>> model = CommmonHistory(lambda history: 
        PTW(16, lambda: CTW(4, lambda: KT(alphabet = range(10)), history = history)))
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

    model_factory: a factory function that can be called to get an instance of a base-level 
        sequence predictor [default: KTBinary]
    min_partition_length: the minimum length of partitions in the tree, always
        gets rounded up to a power of 2 [default: 1024]
    model_store_factory: a factory function to get a set-like object for storing the models 
        (must support the 'add' method and iteration) [default: LogStore]
    """

    def __init__(self, model_factory = KTBinary, min_partition_length = 1024, model_store_factory = LogStore):
        # Create the initial model store with just one model
        # We have to do this before initialize our super class, so that
        #   self.model_factory() will work
        self.models = model_store_factory()
        self.models.add(model_factory())

        # Rest of the initalization
        super().__init__(self.model_factory, min_partition_length = min_partition_length)
        self.model_period = (1 << self.min_height)
        self.t = 0

    def model_factory(self):
        return Averager([ m.copy() for m in self.models ])

    def update(self, symbol, weight = 1.0):
        rv = super().update(symbol, weight)

        self.t += 1
        if self.t % self.model_period == 0:
            self.models.add(self.map().map().copy())

        return rv

    
class Factored(Model):
    """Factored model with independent models that repeat on a fixed period.

    This is mainly for binary models over bytes where a separate model is used for
    each bit position.

    # Examples
    >>> model = Factored([ KT() for i in range(8) ])
    >>> model = Factored([ CTW(16 + i) for i in range(8) ])
    """
    __slots__ = ['factors', 'index']
    
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
