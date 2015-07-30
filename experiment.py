'''
BLINC Adaptive Prosthetics Toolkit
- Bionic Limbs for Improved Natural Control, blinclab.ca
anna.koop@gmail.com

A toolkit for running machine learning experiments on prosthetic limb data
This script handles running/logging data from experiments.

Usage: experiment.py [ARGS]

The type of experiment is determined from the arguments presented. To define
a new type of experiment, extend the datasets.Experiment abstract class with 
the set of Structures required.

'''
from logfiles import *
from datasets import *

from environments import *
import model, modelfast, ac

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

class DataSplitter(Structure):
    """
    For splitting streams of data into disjoint sets for test/training cycles.
    
    split defines the type of split to do
    half:     simply split the data at the midpoint
    sample:   construct sequences that are nsample steps long 
              separated by gap
    """
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

class KTModel(Structure):
    _fields = [Listable('alphabet', default=(0, 1), required=True)]
    
    def create(self):
        return model.KT(alphabet=self.alphabet)

class PTWModel(Structure):
    _fields = [Struct('factory', required=True, default=KTModel, positional=True),
               Integer('minpart', required=True, default=1, keyword=True),
               ]
    
    def create(self):
        if self.factory == CTWModel:
            return model.CommonHistory(lambda history: model.PTW(lambda: modelfast.CTW_KT(self.ctw.depth,
                                                                                          history = history)))
        else:
            return model.PTW(self.factory.create)


class FMNModel(Structure):
    _fields = [Struct('factory', required=True, default=KTModel, positional=True),
               Struct('store', required=True, default=LogStore, positional=True),
               Integer('minpart', required=True, default=1048, keyword=True),
               ]
    
    def create(self):
        if self.factory == CTWModel:
            return model.CommonHistory(lambda history: model.FMN(lambda: modelfast.CTW_KT(self.ctw.depth,                                                                                          history = history)))
        else:
            return model.FMN(self.factory.create)            

    
class CTWModel(Structure):
    _fields = [Struct('factory', required=True, default=KTModel, positional=True, keyword=False),
               Integer('depth', required=True, default=48, keyword=True, positional=False),
               Boolean('fast', required=True, default=True, keyword=True, positional=False)
               ]
    
    def create(self):
        if self.fast:
            return modelfast.CTW_KT(self.depth)
        else:
            return mode.CTW_KT(self.depth)
    """
    if args['-m'] == "CTW":
        probmodel = model.CTW(depth)
    elif args['-m'] == "FastCTW":
        probmodel = modelfast.CTW_KT(depth)
    elif args['-m'] == "PTW_FastCTW":
        probmodel = model.CommonHistory(lambda history: model.PTW(lambda: modelfast.CTW_KT(depth, history = history)))
    elif args['-m'] == "FMN_FastCTW":
        probmodel = model.CommonHistory(lambda history: model.FMN(lambda: modelfast.CTW_KT(depth, history = history)))
    elif args['-m'] == "CTW_KT":
        probmodel = model.CTW_KT(depth)
    elif args['-m'] == "KT":
        probmodel = model.KT()
    elif args['-m'] == "PTW":
        probmodel = model.PTW()
    elif args['-m'] == "FMN":
        probmodel = model.FMN()
    elif args['-m'] == "PTW_CTW":
        probmodel = model.CommonHistory(lambda history: model.PTW(lambda: model.CTW(depth, history = history)))
    elif args['-m'] == "FMN_CTW":
        probmodel = model.CommonHistory(lambda history: model.FMN(lambda: model.CTW(depth, history = history)))
    elif args['-m'] == "CTW_PTW":
        probmodel = model.CTW(depth, lambda: model.PTW())
    else:
        raise Error()

    infile = args['INFILE']
    outfile = args['OUTFILE']

    
    def create(self):
        if self.fast:
            return modelfast.CTW_KT(self.depth)
        else:
            return model.CTW(self.depth)



class CalCorpus(Structure):
    _fields = [ProtParam('protocol', required=True, positional=True, keyword=False)]
    _inherited = ['base_dir']



class ModelExperiment(Experiment):
    _fields = [Parsable('model', default=model.KT),
               ]

    def __enter__(self):
        self.env_len
        self.code_len
        

    def run(self):
        infile = self.env.log_file_path
        outfile = '/dev/null'
        
        msglen = os.path.getsize(self.env.log_file_path)
        codelen = 0
        probmodel = self.model.create()
    
        print("Compressing {} ({} bytes)\n".format(infile, msglen))
        start_time = time.time()
        with open(infile, 'rb') as infs, open(outfile, 'wb') as outfs:
            outfs.write(msglen.to_bytes(4, sys.byteorder))
            
            for b in ac.compress_bytes(probmodel, _bytes_with_progress(infs, msglen)):
                codelen += 1
                outfs.write(bytes([b]))
            elapsed_time = time.time() - start_time
    
        print("\n\nCompression statistics:")
        print("    {:15} {:7.4f}%".format("Ratio:", (codelen+4)/msglen * 100))
        print("    {:15} {:d} bytes".format("Size:", (codelen+4)))
        print("    {:15} {:7.5f}".format("Bits per Byte:", (codelen+4) * 8 / msglen))
        print("    {:15} {:7f}s".format("Time:", elapsed_time))
        print("    {:15} {:7f}".format("KB/s:", msglen / 1024 / elapsed_time))
        if hasattr(probmodel, 'size'): print("    {:15} {:d}".format("# of Nodes:", probmodel.size))
        print("    {:15} {:7.4f} MB".format("Memory Used:", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6))

"""    
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