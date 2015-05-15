import abc
import numpy as np
import functools
import helpers
from utilities import *
from pandas import Series
from collections import deque

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
        Return the -log probability of the data point under the current model
        """
        pass

    def log_predict_sequence(self, seq):
        """
        Under the current model predict the likelihood of the sequence
        but do not actually update
        """
        loss = 0
        for s in seq:
            loss += self.log_predict(s)
        return loss

    def predict(self, data):
        return np.exp(-self.log_predict(data))

    def reset(self):
        """
        Wipe the model entirely
        """
        self.total_loss = 0.0
        self.num_steps = 0

class PTW(Model):
    """
    Inspired by Mike's LogStore and PTW shenanigans

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
        new_loss = model.total_loss

        # merge the models that have reached their endpoints
        for _ in range(mscb(self.num_steps)):
            model = self._models.pop()
            model.update(sym)
            model_loss = model.total_loss

            completed_loss = self._losses.pop()
            new_loss = self.calculate_partition_loss(model_loss,
                                                     completed_loss, new_loss)
        # update the remaining models (if any)
        for m in self._models:
            m.update(sym)

        # append the new values
        self._losses.append(new_loss)
        self._models.append(model)
        self.num_steps += 1

    @property
    def total_loss(self):
        partial_loss = 0 # the base loss for an empty thing
        # calculate the partition loss for each depth
        for i in range(self.depth):
            partial_loss = self.calculate_partition_loss(self.get_model(i+1).total_loss,
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
            model_prob = model.log_predict(sym) + model.total_loss
            partial_loss = self.calculate_partition_loss(model_prob,
                                                         self.get_loss(i), 
                                                         partial_loss)
        return partial_loss - self.total_loss

    def calculate_partition_loss(self, model_loss, 
                                 completed_loss, 
                                 partial_loss):
        """
        This is pulled out mostly for easy debugging (see test_models),
        but is reasonably handy for log_predict as well
        """
        return np.log(2) - log_sum_exp([-model_loss, 
                                        -completed_loss - partial_loss])		


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

    @property
    def log_prob(self):
        """
        This is cumulative loss of the entire sequence.
        """
        return self.total_loss

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
        self.total_loss += d

        self.counts[data] += 1
        self.num_steps += 1
        return d

    def log_predict(self, x):
        """
        >>> kt = KTEstimator()
        >>> kt.predict(0)
        0.5
        >>> np.exp(-kt.log_predict(0))
        0.5
        >>> np.exp(-kt.update(1))
        0.5
        >>> kt.predict(1) > kt.predict(0)
        True
        """
        return np.log(self.num_steps+1)-np.log(self.counts[x])

    def loss(self, sym):
        """
        >>> kt = KTEstimator()
        >>> np.exp(-kt.loss(1))
        0.5
        >>> kt.loss(0) == kt.loss(1)
        True
        >>> kt.update(0)
        """
        #P(m+1,n) = (n+1/2)/(m+n+1)
        #-log(P)=-log(n_c/t_c)=-(log(n_c)-log(t_c))=log(t_c)-log(n_c)
        return self.log_predict(sym)

if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=False)


    kt = KTEstimator()
    pt = PTW(KTEstimator, depth=12)
    from pandas import DataFrame


    values = [0]*32 + [1]*32
    kt_data = [kt.predict(0) for i in values if kt.update(i)]
    pt_data = [pt.predict(0) for i in values if pt.update(i)]

    #data = DataFrame({'kt': kt_data, 'ptw': pt_data})

    print("Done!")
