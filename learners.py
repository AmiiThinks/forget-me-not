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


class PTW(Model):
    """
    Will compute the PTW for the desired depth (and all the ones in between, because why not)
    Shares the model across all the relevant depths
    
    >>> n = PTW(KTEstimator, depth=2)
    >>> n.predict(0)
    0.5
    """
    
    def __init__(self, model_factory, depth=1, deepest=0):
        """
        Initializes a node with no children for the specified depth.
        
        >>> n = PTW(KTEstimator, 4)
        >>> print(n.child)
        None
        >>> n.max_steps
        16
        >>> n.log_prob
        0.0
        >>> n.predict(0)
        0.5
        >>> n.predict(1)
        0.5
        """
        self.Model = model_factory
        self.depth = depth
        self.deepest = deepest

        self.log_prob = 0.0
        self.num_steps = 0
        self.completed_nodes = {} # completed nodes are indexed by depth
        self.models = {} # models are indexed by starting timestep, 
        
    def __repr__(self):
        # TODO: update this when we can pass histories
        return "<PTW^{}_{}(0:{})>".format(self.depth, self.Model.__name__,
                                         self.num_steps - 1)
    
    
    def __str__(self):
        return "PTW^{}({}:{})".format(self.depth, 
                                      self.offset, self.num_steps + self.offset - 1)
   
    def log_predict(self, sym):
        """
        Return the log prediction under the current model 
        
        >>> n = PTW(KTEstimator, depth=4)
        >>> n.log_predict(0) == log(0.5)
        True
        >>> n.predict(0)
        0.5
        
        >>> n2 = PTW(KTEstimator, depth=4)
        >>> exp(n2.update(0))
        0.5
        
        >>> n2.predict(0) - n.predict(0) == n.log_prob
        True
        >>> n.update(0) == n.predict(0)
        True

        >>> n.update(0) == n.predict(0)
        True
        """
        model_preds = {k: v[model]}
        
        if self.child is None:
            partition_loss = self.Model().log_predict(sym) + self.completed_child
        else:
            partition_loss = self.child.log_predict(sym) + self.child.log_prob
            partition_loss = self.child.depth_correction(self.depth-1, 
                                                         model_prob=model_prob,
                                                         loss=partition_loss)
                                                     
        loss = log_0_5 + log_sum_exp(model_prob, partition_loss)
        
        return loss - self.log_prob
         
    def update(self, sym):
        """
        Update to account for the given symbol, spawning and retiring 
        children as necessary.
        Return the conditional probability of the update
        
        >>> n = PTW(KTEstimator)
        >>> n.max_steps
        2
        >>> exp(n.update(0))
        0.5
        >>> exp(n.update(0))
        0.625
        >>> exp(n.update(0))
        Traceback (most recent call last):
        ...
        ValueError: Timestep 2 exceeds max for <PTW^1_KTEstimator(0:1)>

        >>> n = PTW(KTEstimator, depth=4)
        >>> n.max_steps
        16
        >>> n.get_child_string()
        'PTW^4(0:-1)'
        >>> approx(exp(n.update(0)), 0.5)
        True
        >>> n.get_child_string()
        'PTW^4(0:0) PTW^1(0:0)'
        """
        if self.at_boundary:
            msg = "Timestep {} exceeds max for {!r}".format(self.num_steps, self)
            raise ValueError(msg)
        if self.depth == 0:
            print("You shouldn't be here")
            return self.Model().log_predict(sym)
        
        old_loss = self.log_prob        
        self.update_model(sym)
        
        # if we're depth 1 we don't need to spawn a child, we can do the bookkeeping here
        if self.depth==1:
            partial_child = self.Model().log_predict(sym)
        else:
            # spawn a child if you need one
            if self.child is None:
                self.child = PTW(self.Model, 1, offset=self.num_steps + self.offset)
            # now update your child
            self.child.update(sym)
            
            # if it's done but we still need it give it a promotion
            if self.child.at_boundary and self.child.depth < self.depth-1:
                self.child.promote()
                # when it is promoted it recomputes its log_prob for the new depth

            # make sure it is the correct loss for the depth you need
            partial_child = self.child.depth_correction(self.depth-1)

            # now you have P^(d-1)(a:b)
            # if you have a completed_child, it will be P^(d-1)(2^(d-1):t)
            # otherwise it will be your first child P^(d-1)(0:t)

        # compute your own partition loss
        self.log_prob = log_0_5 + log_sum_exp(self.model_prob,
                                                self.completed_child + partial_child)

        self.num_steps += 1
        # check if you need to store things and start a fresh child
        if self.at_half:
            self.completed_child = partial_child
            self.child = None
        elif self.at_boundary:
            self.child = None
            
        delta = self.log_prob - old_loss
        return delta    
    
    def depth_correction(self, depth, loss=None, model_prob=None):
        """
        Loop over the update to adjust for depth. By default use current log_prob, model_prob
        but for predicting might want to pass hypotheticals
        """
        if loss is None:
            loss = self.log_prob
        if model_prob is None:
            model_prob = self.model_prob

        if loss == model_prob:
            return loss
        else:
            for _ in range(depth-self.depth):
                loss = log_0_5 + log_sum_exp(model_prob, loss)
        return loss
            
    
    def update_model(self, sym):
        self.model_prob += self.model.update(sym)
        return self.model_prob
    
    @property
    def at_boundary(self):
        """
        Return true if you have completed your partition
        """
        return self.num_steps == self.max_steps
    
    @property
    def at_half(self):
        """
        Return true if the current timestep completes your first half
        """
        return self.num_steps == self.half_steps
    
    def promote(self):
        """
        Increase the depth you can handle: increment depth, and reset max,
        store the completed loss (your previous total), 
        keep the model, should be childless
        """
        self.depth += 1
        self.half_steps, self.max_steps = self.max_steps, self.max_steps * 2
        self.completed_child = self.log_prob
        # correct your total loss for the new depth
        self.log_prob = log_0_5 + log_sum_exp(self.model_prob, self.completed_child)
        self.child = None
 
    def get_child_string(self):
        info = str(self)        
        with suppress(AttributeError):
            info += " " + self.child.get_child_string()
        return info
    
    def get_leaf(self):
        if self.child is None:
            return self
        else:
            return self.child.get_leaf()

class PTWStack(Model):
    """
    Inspired by Mike's LogStore and PTW shenanigans
    But doesn't correct appropriately for depth

    """
    def __init__(self, model_factory, depth):
        self.factory = model_factory
        self.depth = depth
        self.num_steps = 0
        self._models = []
        self._losses = []

    def update(self, sym):
        """
        Update each of the sub-models with the new symbol and 
        calculate the resulting partition probabilities

        If imaginary, will not actually do the updates, just calculate the
        updated probability
        """
        model = self.factory()
        model.update(sym)
        new_loss = model.log_prob

        # merge the models that have reached their endpoints
        for _ in range(mscb(self.num_steps)):
            model = self._models.pop()
            model.update(sym)
            model_prob = model.log_prob

            completed_child = self._losses.pop()
            new_loss = self.calculate_partition_loss(model_prob,
                                                     completed_child, new_loss)
        # update the remaining models (if any)
        for m in self._models:
            m.update(sym)

        # append the new values
        self._losses.append(new_loss)
        self._models.append(model)
        self.num_steps += 1

    @property
    def log_prob(self):
        partial_loss = 0 # the base loss for an empty thing
        # calculate the partition loss for each depth
        for i in range(self.depth):
            partial_loss = self.calculate_partition_loss(self.get_model(i+1).log_prob,
                                                         self.get_loss(i), 
                                                         partial_loss)
        return partial_loss

    def get_model(self, depth):
        """
        Return the model that looks up to 2**depth steps back
        """
        # probably should do something here for the empty list
        try:
            return self._models[-depth-1]
        except IndexError:
            pass
        try:
            return self._models[0]
        except IndexError:
            return self.factory()
    
    def get_loss(self, depth):
        """
        Return the value for the 2**depth completed subtree, if available. Otherwise 0.
        """
        try:
            return self._losses[-depth-1]
        except IndexError:
            return 0

    def log_predict(self, sym):
        """
        Return the log probability of the given symbol under the current
        model
        """
        partial_loss = self.factory().log_predict(sym)
        # calculate the partition loss for each depth
        for i in range(self.depth):
            model = self.get_model(i+1)
            model_prob = model.log_predict(sym) + model.log_prob
            partial_loss = self.calculate_partition_loss(model_prob,
                                                         self.get_loss(i), 
                                                         partial_loss)
        return partial_loss - self.log_prob

    def calculate_partition_loss(self, model_prob, 
                                 completed_child, 
                                 partial_loss):
        """
        This is pulled out mostly for easy debugging (see test_models),
        but is reasonably handy for log_predict as well
        """
        return log(2) - log_sum_exp([-model_prob, 
                                        -completed_child - partial_loss])		


class KTEstimator(Model):
    """
    A simple Krichevsky–Trofimov estimator
    #TODO: check if it's okay to generalize this
    """

    def __init__(self, alphabet=(0, 1)):
        """
        Create, by default, a 0/1 KT estimator

        >>> kt = KTEstimator()
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
        >>> kt = KTEstimator()
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
        >>> kt = KTEstimator()
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
        >>> kt = KTEstimator()
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

if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=False)
    print("Done!")
    
    p = PTW(KTEstimator, 5)
    for i in range(32):
        p.update(0)
