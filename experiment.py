'''
BLINC Adaptive Prosthetics Toolkit
- Bionic Limbs for Improved Natural Control, blinclab.ca
anna.koop@gmail.com

A toolkit for running machine learning experiments on prosthetic limb data
This script handles running/logging data from experiments, specifically for
experiments which use the emg array as input and predict one of the emg 
signals as output

'''
from logfiles import *
from datasets import *

from sensors import *
from learners import *
from environments import *
from inspect import Parameter

import os
import sys
import argparse    
import pickle
import random
import pandas as pd
import numpy as np
from itertools import chain
from pandas import DataFrame, Index, Series
from sklearn import preprocessing
from sklearn.lda import LDA
from local import base_dir, test_dir


class CoinEnv(Structure):
    _fields = [Listable('probabilities', required=True, positional=True, 
                        keyword=True),
               Listable('changepoints', required=True, positional=True,
                        keyword=True),
               ]
    _inherited = ['base_dir', 'seed']
    constructor = CoinFlip


class ProtParam(Listable):
    _expected_type = str
    def set_self(self, instance, kwargs):
        val = kwargs[self.name]
        if isinstance(val, str):
            setattr(instance, self.name, Protocol.from_string(val))
        elif isinstance(val, DataFrame):
            setattr(instance, self.name, Protocol(val))
        else:
            setattr(instance, self.name, Protocol.from_list(val))

class FileEnvironment(Structure):
    _fields = [String('platform', required=True, positional=True, keyword=False),
               ProtParam('protocol', required=True, positional=True, keyword=False),
               String('pid', required=True, positional=True, keyword=False),
               Keyed('transforms', required=True, positional=False, keyword=False),
               Integer('window', required=False, keyword=True, default=empty),
               Keyed('ranges', required=False, keyword=True, default=empty),
               ]
    _inherited = ['base_dir']
    constructor = FileEnvironment
    
    @property
    def log_file(self):
        #return self.create().file_name
        return 'raw-data.txt'
    
    def get_data(self, header=None):
        """
        Using the env_base of this instance, pull all the data (header=[]),
        the raw data for the specified header, 
        the filtered data as specified by kwargs,
        or the headers specified in the environment
        """
        kwargs = self.get_params(header=header)
        return FileEnvironment.read_log(**kwargs)

    def get_params(self, header=None):
        """
        Return a dictionary of the settings for this environment, with either
        header = [] -> all the data
        header = something -> the raw data for that header
        default -> the data specified (headers and filters) at init
        """
        kwargs = vars(self).copy()
        headers = kwargs.pop('transforms')
        if header is None:
            kwargs.update(headers)
        else:
            kwargs[header] = []
        kwargs.pop('fe', None)
        kwargs.pop('_headers', None)
        kwargs.pop('_data', None)
        return kwargs

    def create(self):
        return FileEnvironment(**self.get_params())

    def __enter__(self):
        self.fe = self.create()
        self.fe.start()

    def __call__(self):
        return self.fe.update()
    
    def __exit__(self, *args):
        self.fe.reset()
        
    def run(self):
        self._data = self.get_data()
        return self._data
        
    @property
    def data(self):
        # would look for log file here if we kept one
        if hasattr(self, '_data'):
            return self._data
        else:
            return self.run()

    def get_headers(self):
        print("Redundant")
        try:
            return self.fe.headers
        except AttributeError:
            self.fe = self.create()
            return self.fe.headers

    @property
    def headers(self):
        if not hasattr(self, '_headers'):
            try:
                self._headers = self.fe.headers
            except AttributeError:
                self.fe = self.create()
                self._headers = self.fe.headers
        return self._headers
        


    #### PARAMETER CALCS ####
    @property
    def output_dim(self):
        return len(self.headers)
    
    @property
    def num_features(self):
        return self.output_dim

    #### USEFUL FUNCTIONS ####    

    def get_return_data(self, signal, gamma):
        print("Overly specific")
        with self:
            data = self.fe.get_return_data(header=target.signal, gamma=gamma)
        return data

class DataSplitter(Structure):
    _fields = [Parsable('split', required=True),
               Integer('gap', required=False, default=150),
               Integer('nsample', required=False, default=100),
               Integer('seed', required=False)
               ]

    def __call__(self, data):
        if self.split == 'half':
            half = math.floor(len(data)/2)
            return (data[:half], data[half:])
        elif self.split == 'sample':
            if not hasattr(self, 'seed'):
                self.seed = os.urandom(10)

            tot = self.gap + self.nsample
            firsts = list(range(self.gap, len(data)-tot, tot))
            random.seed(self.seed)
            random.shuffle(firsts)
            half = math.floor(len(firsts)/2)
            tr = list(chain(*[range(i, i+self.nsample) for i in firsts[half:]]))
            tr.sort()
            te = list(chain(*[range(i, i+self.nsample) for i in firsts[:half]]))
            te.sort()
            return(data.ix[tr], data.ix[te])
        else:
            return (data, data)


class PredictionExperiment(Experiment):
    _fields = [Dir('base_dir', required=True, transient=True),
               Struct('env', constructor=CoinEnv),
               Struct('predictor', constructor=Learner),
               Integer('seed', required=False),
               Integer('length', required=True)
               ]
    
    def __enter__(self):
        self.env.set_inherited(instance=self)


def sample_run():
    pass

if __name__ == "__main__":
    if len(sys.argv) > 1:
        print("Creating...")
        setup = Experiment.from_args(sys.argv[1:])
        print("Running...")
        setup.run()
        print("Done")
    else:
        sample_run()