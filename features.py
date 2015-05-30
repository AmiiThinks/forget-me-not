'''
BLINC Adaptive Prosthetics Toolkit
- Bionic Limbs for Improved Natural Control, blinclab.ca
anna.koop@gmail.com

A toolkit for running machine learning experiments on prosthetic limb data

This module handles feature processing

# TODO: time domain features
'''

from sklearn import preprocessing
import numpy as np
import ast

class Filter:
    """
    Creates a filter for converting raw inputs into meaningful signals
    """
    pass
        
        

def strs_to_floats(values):
    """
    Converts elements into an array of floats where they've been stored
    as a list or dataframe of strings '[0, 0]'

    # TODO check the safety of this
    """
    strings = '[' + ', '.join(values) + ']'
    return np.array(ast.literal_eval(strings))

def scale_matrix(values):
    scaler = preprocessing.MinMaxScaler()
    return scaler.fit_transform(values)

def get_unitizer(max_class):
    def unitizer(label):
        x = np.zeros(max_class)
        x[label] = 1
        return x
    return unitizer

def running_mean(data, n):
    """
    Take a vector a values and a integer window size
    Return the vector of values that are the mean over n steps.
    Note that (right now at least) the returned vector will be n-1 elements smaller.

    >>> running_mean([1, 2, 2, 4, 1, 1], 2)
    array([ 1.5,  2. ,  3. ,  2.5,  1. ])

    >>> running_mean([1, 1, 1, 1, 2, 2, 1, 1, 2, 2, 2, 2], 4)
    array([ 1.  ,  1.25,  1.5 ,  1.5 ,  1.5 ,  1.5 ,  1.5 ,  1.75,  2.  ])
    """
    return np.convolve(data, np.ones((n, ))/n)[(n-1):-(n-1)]


def calculate_return_series(r, gamma=1, horizon=None):
    # TODO: unhack this
    return Series(calculate_return(r, gamma=gamma, horizon=horizon))

def calculate_return(r, gamma=1, horizon=None):
    """
    Returns a sequence of len(r) using gamma to calculate
    the return value.
    Should use horizon rather than discounting, will assume r[0:horizon] is the
    relevant bit


    >>> calculate_return([1, 0, 0], .5)
    array([ 1.,  0.,  0.])
    >>> calculate_return([0, 0, 1], .5)
    array([ 0.25,  0.5 ,  1.  ])
    >>> calculate_return([0, 1, 0, 0, 0, 1],.5)
    array([ 0.53125,  1.0625 ,  0.125  ,  0.25   ,  0.5    ,  1.     ])
    >>> a = calculate_return([0, 0, 1, 1, 0, 0, -1, -1], .1)
    >>> a[:5]
    array([ 0.0109989,  0.109989 ,  1.09989  ,  0.9989   , -0.011    ])
    >>> a[5:]
    array([-0.11, -1.1 , -1.  ])

    >>> calculate_return([0, 1, 0, 1, 0, 1], horizon=3)
    array([1, 2, 1, 2, 1, 1])
    >>> calculate_return([0, 1, 0, 1, 0, 1], horizon=6)
    array([3, 3, 2, 2, 1, 1])
    >>> calculate_return([0, 1, 0, 1, 0, 1], horizon=10)
    array([3, 3, 2, 2, 1, 1])
    """
    l = len(r)
    if l == 0:
        return []
    if gamma is None or gamma >= 1 or gamma is np.nan:
        r = list(r)
        ret = [sum(r[i:min(i+horizon, l)]) for i in range(l)]
        return np.array(ret)

    ret = np.array([0.0]*l)
    ret[l-1] = r[l-1]
    for i in range(l-2, -1, -1):
        ret[i] = r[i] + gamma * ret[i+1]
    return ret


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=False)
    print("Done!")
