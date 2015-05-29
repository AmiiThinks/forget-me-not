import abc
import numpy as np
from numpy import exp, log
import functools
from pandas import Series
from collections import deque

from helpers import *
log_0_5 = log(0.5)

class Model(metaclass=abc.ABCMeta):

    @classmethod
    def from_sequence(cls, seq, **kwargs):
        me = cls(**kwargs)
        seq = helpers.listify(seq)
        for s in seq:
            me.update(s)
        return me
    
    @classmethod
    def log_predict(cls, seq, **kwargs):
        return cls.from_sequence(seq, **kwargs).log_prob

    def __init__(self):
        self.reset()

    @abc.abstractmethod
    def update(self, data):
        """
        Update according to learning rules
        """
        pass

    @abc.abstractmethod
    def log_predict(self, data):
        """
        Return the log probability of the data point under the current model
        """
        pass

    def predict(self, data):
        return exp(self.log_predict(data))

    @property
    def total_loss(self):
        """
        This is cumulative loss of the entire sequence.
        """
        return -self.log_prob

    def reset(self):
        """
        Wipe the model entirely
        """
        self.log_prob = 0.0
        self.num_steps = 0


class ModelStore(metaclass=abc.ABCMeta):
    """
    Abstract class for defining a storage mechanism
    """
    
    def __init__(self):
        self._items = []
        self.t = 0
    
    def __len__(self):
        return len(self._items)
    
    def __iter__(self):
        for x in self._items:
            yield x
            
    def __getitem__(self, i):
        return self._items[i]
    
    def add(self, x):
        """
        This is where each model store defines its particular mechanism
        """
        self.t += 1
    
class LogStore(ModelStore):
    """
    Stores a log(t) number of objects, weighted towards more recent objects
    
    #TODO: allow >1 gaps between models
    
    >>> store = LogStore()
    >>> for i in range(12):
    ...     store.add(i)
    >>> list(store)
    [11, 10, 8, 0]
    """
    def add(self, x):
        with suppress(IndexError):
            self._items.pop(mscb(self.t))
        self._items.insert(0, x)
        self.t += 1
    
class PTW(Model):
    """
    Will compute the PTW for the desired depth (and all the ones in between, because why not)
    Shares the model across all the relevant depths
    
    
    
    >>> p = PTW(KT, depth=2)
    >>> p.predict(0)
    0.5
    >>> p.predict(1)
    0.5
    
    >>> exp(p.update(0))
    0.5
    >>> exp(p.completed_log_probs)
    array([ 0.5,  1. ,  1. ])
    >>> exp(p.log_prob)
    0.5
    
    >>> approx(p.predict(0), 0.6875)
    True
    >>> approx(p.predict(1), 0.3125)
    True
    
    >>> approx(exp(p.update(0)), 0.6875)
    True
    >>> exp(p.completed_log_probs)    
    array([ 1.    ,  0.3125,  1.    ])
    
    >>> p = PTW(KT, depth=5)
    >>> p.log_predict(0) == p.update(0)
    True
    >>> print(p.get_child_string(), '  ', p.get_model_string()) # 0
    PTW^0    KT(0:0)

    >>> p.log_predict(0) == p.update(0)
    True
    >>> print(p.get_child_string(), '  ', p.get_model_string()) # 1
    PTW^1    KT(0:1)

    >>> p.log_predict(0) == p.update(0)
    True
    >>> print(p.get_child_string(), '  ', p.get_model_string()) # 2
    PTW^1 PTW^0    KT(0:2) KT(2:2)

    >>> p.log_predict(0) == p.update(0)
    True
    >>> print(p.get_child_string(), '  ', p.get_model_string()) # 3
    PTW^2    KT(0:3)

    >>> p.log_predict(0) == p.update(0)
    True
    >>> print(p.get_child_string(), '  ', p.get_model_string()) # 4
    PTW^2 PTW^0    KT(0:4) KT(4:4)

    >>> p.log_predict(0) == p.update(0)
    True
    >>> print(p.get_child_string(), '  ', p.get_model_string()) # 5
    PTW^2 PTW^1    KT(0:5) KT(4:5)

    >>> p.log_predict(0) == p.update(0)
    True
    >>> print(p.get_child_string(), '  ', p.get_model_string()) # 6
    PTW^2 PTW^1 PTW^0    KT(0:6) KT(4:6) KT(6:6)

    >>> p.log_predict(0) == p.update(0)
    True
    >>> print(p.get_child_string(), '  ', p.get_model_string()) # 7
    PTW^3    KT(0:7)

    >>> p.log_predict(0) == p.update(0)
    True
    >>> print(p.get_child_string(), '  ', p.get_model_string()) # 8
    PTW^3 PTW^0    KT(0:8) KT(8:8)
    """
    
    def __init__(self, model_factory, depth=1, deepest=0):
        """
        >>> p = PTW(KT, 4)
        >>> p.max_steps
        16
        >>> p.log_prob
        0.0
        >>> p.predict(0)
        0.5
        >>> p.predict(1)
        0.5
        """
        self.Model = model_factory
        self.depth = depth
        self.deepest = deepest
        self.max_steps = 2**self.depth

        self.log_prob = 0.0
        self.num_steps = 0
        self.models = {0: self.Model()} # models are indexed by starting timestep, 
        self.model_log_probs = {0: 0} # ditto
        
        self.completed_log_probs = [0 for _ in range(self.depth+1)]
        if self.deepest != 0:
            raise NotImplementedError("Oops, not there yet")
        
    def __repr__(self):
        # TODO: update this when we can pass histories
        return "<PTW^{}_{}(0:{})>".format(self.depth, self.Model.__name__,
                                         self.num_steps-1)
    
    def to_string(self, depth=None, t=None):
        """
        Construct a string representation of the partition log_prob at the 
        given depth and time (default to full tree and current time)
        
        >>> p = PTW(KT, depth=12)
        >>> print(p.to_string())
        PTW^12(0:-1)
        
        >>> print(p.to_string(t=10))
        PTW^12(0:10)

        >>> print(p.to_string(depth=5, t=100))
        PTW^5(96:100)
        """
        if depth is None:
            depth = self.depth
        if t is None:
            t = self.num_steps - 1
        if t < 0:
            start = 0
        else:
            start = binary_boundary(t, depth)
        return "PTW^{}({}:{})".format(depth, start, t)

    def get_child_string(self):
        """
        Return a string representation of the current completed subtrees

        >>> p = PTW(KT, depth=4)
        >>> for _ in range(4): d = p.update(0)
        >>> print(p.get_child_string())
        PTW^2
        >>> d = p.update(0); print(p.get_child_string())
        PTW^2 PTW^0
        >>> d = p.update(0); print(p.get_child_string())
        PTW^2 PTW^1
        >>> d = p.update(0); print(p.get_child_string())
        PTW^2 PTW^1 PTW^0
        >>> d = p.update(0); print(p.get_child_string())
        PTW^3
        """
        p = "PTW^{}"
        
        if self.num_steps == self.max_steps:
            return p.format(self.depth)

        return " ".join([p.format(i) for i in range(self.depth-1, -1,-1) \
                         if self.completed_log_probs[i] != 0.0])        
    
    def get_model_string(self):
        """
        Return a string representation of the currently-stored models

        >>> p = PTW(KT, depth=4)
        >>> print(p.get_model_string())
        KT(0:-1)
        >>> d = p.update(0); print(p.get_model_string())
        KT(0:0)
        >>> d = p.update(0); print(p.get_model_string())
        KT(0:1)
        >>> d = p.update(0); print(p.get_model_string())
        KT(0:2) KT(2:2)
        >>> d = p.update(0); print(p.get_model_string())
        KT(0:3)
        >>> for _ in range(3): d = p.update(0)
        >>> print(p.get_model_string())
        KT(0:6) KT(4:6) KT(6:6)
        """
        return " ".join(["{}({}:{})".format(self.Model.__name__,
                                           k, self.num_steps-1) for k in sorted(self.models)])
            
        
    @property
    def at_deepest(self):
        """
        Check if our current timestep is on the smallest splitpoint we care about
        """
        return at_partition_start(self.num_steps, self.deepest+1)
    
    @property
    def mscb(self):
        return mscb(self.num_steps)

    def update(self, sym):
        """
        >>> n = PTW(KT)
        >>> exp(n.update(0))
        0.5
        >>> exp(n.model_log_probs[0])
        0.5
        >>> exp(n.completed_log_probs)
        array([ 0.5,  1. ])
        >>> exp(n.update(0))
        0.625
        >>> exp(n.completed_log_probs)
        array([ 1.    ,  0.3125])
        >>> exp(n.update(0))
        Traceback (most recent call last):
        ...
        ValueError: Timestep 2 exceeds max for <PTW^1_KT(0:1)>

        """
        if self.num_steps >= self.max_steps:
            msg = "Timestep {} exceeds max for {!r}".format(self.num_steps, self)
            raise ValueError(msg)
        
        # figure out which depths have been completed
        cur_bound = self.mscb
        # store this so we can report the change
        cur_prob = self.log_prob
        
    
        # for now we can make this on every step even though technically we don't need it
        # every time
        self.model_log_probs[self.num_steps] = 0
        self.models[self.num_steps] = self.Model()

        # update the models
        for m in self.models:
            self.model_log_probs[m] += self.models[m].update(sym)

        part = self.model_log_probs[self.num_steps]
        complete = part
        # now walk up the partition calculation
        for i in range(1, self.depth + 1):
            d = i
            m = binary_boundary(self.num_steps, d)
            part = log_0_5 + log_sum_exp(self.model_log_probs[m], 
                                         self.completed_log_probs[i-1] + part)
            if i == cur_bound:
                complete = part
  
        # store the root value
        self.log_prob = part #self.partial_log_probs[-1]

        # this is where we might need to increase the depth
        # clear out the completed probs you're done using
        self.completed_log_probs[cur_bound] = complete
        for i in range(cur_bound):
            self.completed_log_probs[i] = 0
            m = binary_boundary(self.num_steps, i)
            if m != binary_boundary(self.num_steps, i+1):
                del self.models[m]
                del self.model_log_probs[m]
        
        self.num_steps += 1
    
        return self.log_prob - cur_prob


    def log_predict(self, sym):
        """
        Figure out the conditional probability by faking the predicted sequence and 
        return the change in log_prob

        Return the log prediction under the current model 
        
        >>> n = PTW(KT, depth=4)
        >>> n.log_predict(0) == log(0.5)
        True
        >>> n.predict(0)
        0.5
        
        >>> n2 = PTW(KT, depth=4)
        >>> exp(n2.update(0))
        0.5
        
        >>> n.log_predict(0) == n.update(0)
        True
        >>> n.log_predict(0) == n.update(0)
        True
        """
        pred = self.Model().log_predict(sym)
        for i in range(1, self.depth+1):
            m = binary_boundary(self.num_steps, i)
            model_pred = self.models.get(m, self.Model()).log_predict(sym) + \
                self.model_log_probs.get(m, 0)
            part = pred + self.completed_log_probs[i-1]
            pred = log_0_5 + log_sum_exp(model_pred, part)
        return pred - self.log_prob


class Averager(Model):
    """
    Adapted from Mike's code, pretty exactly
    
    Requires a collection of models
    """
    def __init__(self, models):
        self.models = models
        self.reset()    
    
    def reset(self):
        self.log_prob = 0
        log1n = log(1.0/len(self.models))
        self.models = {m: log1n for m in self.models}

    def update(self, sym):
        cur_prob = self.log_prob
        
        for m in self.models:
            self.models[m] += m.update(sym)
        self.log_prob = log_sum_exp(*self.models.values())
        
    def log_predict(self, sym):
        return log_sum_exp(*(m.log_predict(sym) + \
                             lp for m, lp in self.models.items()))

    def map(self):
        """
        Return the single model that is the maximum a posteriori model
        given the sequence so far
        """
        # TODO: break ties?
        return max(self.models, key=lambda m: self.models[m])


class KT(Model):
    """
    A simple Krichevsky–Trofimov estimator
    #TODO: check if it's okay to generalize this
    """

    def __init__(self, alphabet=(0, 1)):
        """
        Create, by default, a 0/1 KT estimator

        >>> kt = KT()
        >>> kt.predict(1)
        0.5
        >>> kt.predict(0)
        0.5
        """
        self.alphabet = alphabet
        self.num_symbols = len(alphabet)
        self.reset()

    def reset(self):
        super().reset()
        self.counts = Series({a: 1/self.num_symbols for a in self.alphabet})

    def update(self, data):
        """
        >>> kt = KT()
        >>> kt.counts[0]
        0.5
        >>> l = kt.update(0)
        >>> kt.counts[0]
        1.5
        """
        d = self.loss(data)
        self.log_prob += d

        self.counts[data] += 1
        self.num_steps += 1
        return d

    def log_predict(self, x):
        """
        >>> kt = KT()
        >>> kt.predict(0)
        0.5
        >>> exp(kt.log_predict(0))
        0.5
        >>> exp(kt.update(1))
        0.5
        >>> kt.predict(1) > kt.predict(0)
        True
        """
        return log(self.counts[x]) - log(self.num_steps+1)

    def loss(self, sym):
        """
        >>> kt = KT()
        >>> exp(kt.loss(1))
        0.5
        >>> kt.loss(0) == kt.loss(1)
        True
        >>> exp(kt.update(0))
        0.5
        """
        #P(m+1,n) = (n+1/2)/(m+n+1)
        #-log(P)=-log(n_c/t_c)=-(log(n_c)-log(t_c))=log(t_c)-log(n_c)
        return self.log_predict(sym)

class SAD(Model):
    """
    Sparse Adaptive Dirichlet Process

    From M. Hutter via Mike. Currently copied directly
    """
    
    def __init__(self, n):
        self.n = n
        self.counters = {}
        self.sum_counts = 0
        
    def update(self, symb):
        # TODO: check if defaultdict is faster
        lp = self.log_predict(sym)
        self.counts.setdefault(sym, 0)
        self.counts[sym] += 1 
        self.sum_counts += 1
        return lp
    
    def log_predict(self, symbol):
        # TODO: actually figure out what this does
        
        m = min(len(self.counts), self.sum_counts)
        beta = m / (2 * math.log((self.sum_counts + 1) / m)) if self.sum_counts > 0 else 1

        if symbol in self.counts:
            return math.log(self.counts[symbol] / (self.sum_counts + beta))
        else:
            return math.log(beta / ((self.n - len(self.counts)) * (self.sum_counts + beta)))
        

if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=False)
    print("Done!")
    
    p = PTW(KT, 12)
    import model
    m = model.PTW(12)
    for i in range(32):
        assert approx(p.predict(0), m.predict(0))
        #print("Prediction", p.predict(0), "|", m.predict(0))
        d = p.update(0);
        dm = m.update(0);
        #print("Update", d, exp(d), "|", dm, exp(dm))
        #print("Completed", p.completed_log_probs, exp(p.completed_log_probs))
        #print("Partial", p.partial_log_probs, exp(p.partial_log_probs))
        #print("Models", p.model_log_probs)
        
    print("DONE!")
        
