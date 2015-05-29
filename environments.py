'''
BLINC Adaptive Prosthetics Toolkit
- Bionic Limbs for Improved Natural Control, blinclab.ca
anna.koop@gmail.com

A toolkit for running machine learning experiments on prosthetic limb data

This module file environments
# TODO: expand to handle ros environments
'''
import os
import pandas as pd
import numpy.random as random
from sklearn import preprocessing
from contextlib import contextmanager
from features import calculate_return, get_unitizer
from collections import defaultdict
from functools import partial
from datasets import *
from logfiles import *
from local import base_dir, test_dir



class EnvDir(namedtuple('EnvDir', 'dirname steps')):
    def __str__(self):
        if self.steps:
            return "{}-{}".format(self.dirname, self.steps)
        else:
            return self.dirname

class Protocol():
    """
    Really lightweight object for handling conversions from lists to base_dir
    to string
    
    >>> df = DataFrame({'steps': [100], 'dirname': ['hand-wrist']})
    >>> print(Protocol(df))
    hand-wrist-100
    >>> print(Protocol.from_string('hand-wrist-100_hand-wrist-10'))
    hand-wrist-100_hand-wrist-10
    >>> print(Protocol.from_list(['hand-wrist', 100]))
    hand-wrist-100
    """
    @staticmethod
    def get_protocol_parts(protocol):
        """
        Break down a protocol string into its component parts
    
        >>> Protocol.get_protocol_parts('fred')
        [('fred', 0)]
        >>> Protocol.get_protocol_parts('fred-')
        [('fred', 0)]
        >>> Protocol.get_protocol_parts('fred-12')
        [('fred', 12)]
        >>> Protocol.get_protocol_parts('fred-_jones-')
        [('fred', 0), ('jones', 0)]
        >>> Protocol.get_protocol_parts('fred_jones')
        [('fred', 0), ('jones', 0)]
        >>> Protocol.get_protocol_parts('fred-15_jones-12')
        [('fred', 15), ('jones', 12)]
        >>> Protocol.get_protocol_parts('fred-pos-12')
        [('fred-pos', 12)]
        """
        parts = protocol.split('_')
        ps = []
        for p in parts:
            subparts = p.split('-')
            # check if the last part is empty, as for test-
            if not subparts[-1]:
                name = '-'.join(subparts[:-1])
                num = 0
            else:
                #TODO: allow floats someday for % repeats
                # if the last bit is an integer, great
                try:
                    num = int(subparts[-1])
                    name = '-'.join(subparts[:-1])
                except ValueError:
                    name = '-'.join(subparts)
                    num = 0
            assert isinstance(name, str)
            assert isinstance(num, int)
            ps.append((name, num))
        return ps

    @classmethod
    def from_list(cls, protocols):
        return cls(DataFrame(list(zip(protocols[0::2], 
                                      protocols[1::2])), 
                             columns=['dirname', 'steps'], dtype=int))

    @classmethod
    def from_string(cls, protocol):
        """
        Parse a string representation of protocol
        
        >>> p = Protocol.from_string('hand-wrist')
        >>> p.get_dirs()
        ['hand-wrist']
        >>> p.data.dirname[0]
        'hand-wrist'

        """        
        return cls(DataFrame(Protocol.get_protocol_parts(protocol), 
                             columns=['dirname', 'steps'], dtype=int))

    def __iter__(self):
        """
        >>> p = Protocol.from_string('hand-wrist-100_hand-wrist-10')
        >>> for pr in p: print(pr, pr.steps)
        hand-wrist-100 100
        hand-wrist-10 10
        >>> pr
        EnvDir(dirname='hand-wrist', steps=10)
        """
        for v in self.data.values:
            yield EnvDir(*v)

    def __init__(self, df):
        if isinstance(df, DataFrame):
            self.data = df
        else:
            msg = "Must call custom from_ method for type {}".format(type(df))
            raise(ValueError(msg))
        
    def get_dirs(self):
        return list(self.data.dirname)

    def __str__(self):
        return '_'.join(self.data.fillna(0).apply(format_protocol, axis=1))



class Chainer():
    """
    Another lightweight object for handling repeated presentation of files.
    
    Same format as protocol, but in this case only the last element should have a number, 
    and that is the number of times to repeat the specified sequence of files
    """
    def __init__(self, *args, reps=1):
        self.reps = reps
        self.data = DataFrame(data=args, columns=['dirname'])
        self.data['steps'] = 0
        self.data



class CoinFlip():
    """
    Construct a generator/callable that returns a random binary signal
    
    For now, accepts a probability (of 1) or list of probabilities and 
    changepoints
    """
    @staticmethod
    def parse_prob(data, prob):
        """
        Calculate the 0/1 designation according to the given probability
        
        >>> CoinFlip.parse_prob(.6, .7)
        0
        >>> CoinFlip.parse_prob(.9, .7)
        1
        >>> CoinFlip.parse_prob([.2, .65, .9, .5], .5).values
        array([0, 1, 1, 1])
        """
        try:
            return 1 if data >= prob else 0
        except TypeError:
            data = Series(data)
            data[data>=prob] = 1
            data[data<prob] = 0
            
        return data.astype(int)
    
    def _iter_changes(self):
        """
        Loop through the contexts
        """
        if self.changepoints is None:
            yield (0, -1)
        else:
            yield from enumerate(self.changepoints)
            yield (len(self.changepoints), -1)
            
    def __iter__(self):
        with self:
            yield self()
                
            

    def __init__(self, prob=0.5, seed=None, changepoints=None):    
        """
        Will get seed from random if not provided, so it can be saved.
        
        If changepoints are provided, it should be a list at least as long
        as the list of probabilities. 
        
        >>> cf = CoinFlip(.5, seed=0)
        >>> (cf.timestep, cf.num_changes)
        (-1, -1)
        >>> (cf.total_changes, cf.num_contexts)
        (0, 1)
        >>> cf.seed
        0
        >>> cf.probabilities
        [0.5]
        
        >>> cf = CoinFlip()
        >>> cf.seed != 0
        True
        >>> cf.probabilities
        [0.5]
        
        >>> cf = CoinFlip([0.2, 0.8])
        >>> (cf.changepoints, cf.num_contexts)
        (None, 2)
        
        >>> cf = CoinFlip([0.2, 0.8], changepoints=[16, 32, 64])
        >>> (cf.changepoints, cf.num_contexts)
        ([16, 32, 64], 2)
        """
        self.prob = None
        if seed is None:
            # TODO: may want to generate someting
            self.seed = random.get_state()[1][0]
        else:
            self.seed = seed
        
        self.probabilities = listify(prob)
        self.num_contexts = len(self.probabilities)
        if changepoints is not None:
            self.total_changes = len(changepoints)
            self.changepoints = changepoints
        else:
            self.total_changes = 0
            self.changepoints = None
        self.reset()

    def __enter__(self):
        """
        Set up for going through the list of probabilities
        
        >>> cf = CoinFlip(0.25, seed=0)
        >>> (cf.timestep, cf.num_changes)
        (-1, -1)
        >>> s = cf.__enter__()
        >>> (cf.timestep, cf.num_changes)
        (-1, 0)
        >>> cf.prob
        0.25
        
        >>> cf = CoinFlip([0.24, .75], changepoints=[2])
        >>> (cf.prob, cf.next_change)
        (None, 0)
        >>> s = cf.__enter__()
        >>> (cf.prob, cf.next_change)
        (0.24, 2)
        """
        self.reset()
        random.seed(self.seed)
        self.update_probability()
        return self

    def reset(self):
        self.timestep = -1
        self.num_changes = -1
        self.next_change = 0
        self.context = None
        self.prob = None
    
    def __exit__(self, *args):
        self.context = None
        self.prob = None
        
    def __call__(self):
        """
        Increment the timestep, check if the probability needs to be updated,
        then return the coin flip value.
        
        This way the stored self.prob will always be the distribution that
        generated the data output
        
        
        >>> cf = CoinFlip([0.24, .75], changepoints=[2], seed=0)
        >>> cf()
        1
        >>> cf = CoinFlip(.9, seed=0)
        >>> cf()
        0
        """
        if self.timestep < 0:
            self.__enter__()
        self.timestep += 1
        self.update_probability()
        return CoinFlip.parse_prob(random.random(), self.prob)
    
    def update_probability(self):
        """
        TODO: this is totally dodgy and won't work if the changepoints are 
        out of order


        >>> cf = CoinFlip([0.24, .75], changepoints=[2])
        >>> (cf.prob, cf.num_changes)
        (None, -1)
        >>> s = cf(); s = cf();
        >>> (cf.prob, cf.num_changes)
        (0.24, 0)
        >>> s = cf()
        >>> (cf.prob, cf.num_changes)
        (0.75, 1)
        
        """
        if self.timestep < 0:
            self.context = self._iter_changes()
            self.next_change = -1
            
        if self.timestep == self.next_change:
            (self.num_changes, self.next_change) = self.context.__next__()
            ind = self.num_changes % self.num_contexts
            self.prob = self.probabilities[ind]
            
    # TODO: the probability seems backwards
    def get_data(self, terminal):
        """
        Get a bunch of data at once
        
        >>> cf = CoinFlip(.5)
        >>> data = cf.get_data(1000)
        >>> sum(data) < 525 and sum(data) > 475
        True
        
        
        >>> cf = CoinFlip([0.1, 0.9], changepoints=[1000])
        >>> data = cf.get_data(2000)
        >>> sum(data[:1000]) > sum(data[1000:])
        True
        """
        with self as flipper:
            return Series([flipper() for _ in range(terminal)])
                
            
        
    

class FileEnvironment():
    """
    Constructs a generator/callable that steps through environment file(s) and processes the features
    
    Filter parameters must be specified and given a default value of "None" so that we can make sure they are actually set.
    
    required parameters are popped off of kwargs before 
    kwargs are used to pass lists of filters for every particular file header
    """
    @classmethod
    def read_log(cls, *args, **kwargs):
        return cls(*args, **kwargs).get_all_data()
    
    def __init__(self, base_dir, platform, protocol, pid, 
                 file_name='raw-data.txt', clean_file_data=False, **kwargs):
        self.base_dir = os.path.join(base_dir, platform)
        self.suffix = os.path.join(pid, file_name)

        if clean_file_data:
            raise NotImplementedError("Don't think I have cleaning file data working")
        self.clean_file_data = clean_file_data
        if isinstance(protocol, Protocol):
            self.protocol = protocol
        elif isinstance(protocol, str):
            self.protocol = Protocol.from_string(protocol)
        else:
            self.protocol = Protocol.from_list(protocol)
        self.file_name = file_name
        self.file_filters = self.get_file_filters(**kwargs)
        self.set_headers()
        self.reset()
        
    @staticmethod
    def get_file_filters(**kwargs):
        """
        Returns a dictionary mapping feature headers to functions
        """
        # pop off particular parameters the filters need
        trace_rate = kwargs.pop('trace_rate', None)
        window = kwargs.pop('window', None)
        ranges = kwargs.pop('ranges', None)
        num_classes = kwargs.pop('num_classes', None)
        
        ff = defaultdict(list)
             
        for header, filtrs in kwargs.items():
            # for each function, append it set it as necessary
            ff[header] = []
            filtrs = listify(filtrs)
            for ftr in filtrs:
                if not ftr:
                    continue
                elif ftr == 'rollmean':
                    func = partial(pd.rolling_mean, window=window, min_periods=1)
                elif ftr == 'trace':
                    raise NotImplementedError("Don't have traced features yet")
                elif ftr == 'return':
                    func = partial(calculate_return, gamma=gamma, horizon=horizon)
                elif ftr == 'scaled':
                    func = lambda x: (x-ranges[header][0])/ranges[header][1]
                elif ftr == 'unitvect':
                    func = get_unitizer(num_classes)
                else:
                    raise ValueError("Unknown filter type {}".format(ftr))
                ff[header].append(func)
        return ff
    
    def __iter_chunks__(self):
        """
        Loop through each complete datablock
        The datablocks are determined by the protocols and may be
        cropped or looped depending on the size the protocol specifies
        """
        self.start()
        while self.dir_iter is not None:
            self.update_data_block()
            yield self.get_data_block()
        self.reset()

    def get_all_data(self):
        """
        Return a full-size dataframe for all the required data over all protocols
        """
        return pd.concat([x for x in self.__iter_chunks__()], ignore_index=True)

    def get_env_data(self, **kwargs):
        """
        For when you want the same base environment, but not necessarily the 
        block of headers you specified. Goes back to the environment file to 
        pull data
        
        kwargs holds the filter information
        {header_name: filter_types_from_left_to_right,
         parameter1: val,
         parameter2: val}
        """
        return FileEnvironment(self.base_dir, self.platform, self.protocol, self.pid,
                   **kwargs).get_all_data()

    def get_raw_data(self, header=None, **kwargs):
        """
        Get the environment data for the given header/filter info
        Return as a block
        """
        return read_log(self, self.base_dir, self.platform, self.protocol,
                        self.pid, file_name=self.file_name, **kwargs)      
        
    def get_return_data(self, header=None, gamma=None, horizon=None):
        """
        Get the full data vector for 
        Calculate the return data for these environment settings and the given parameters
        #TODO: don't know if this should be here or not.
        """
        old_headers = self.headers
        if header is not None:
            self.headers = listify(header)
        data = self.get_all_data()
        if gamma is not None or horizon is not None:
            data['Return'] = calculate_return(data.values, gamma=gamma, horizon=horizon)
        self.headers = old_headers
        return data

    def get_data_block(self):
        """
        Get all of the current data block
        Unlike the incremental version, this will duplicate rows to fill out the requested
        protocol length
        # TODO: generalize this to take protocol as a potential argument
        # TODO: or make a helper function that expands/shrinks a dataframe
        """
        if self.data is None:
            return None

        data_size = len(self.data.index)
        if self.current_max <= 0 or self.current_max == data_size:
            self.local_step = data_size
            return self.data       

        repeats = self.current_max // data_size
        remainder = self.current_max % data_size
        if repeats:
            data = pd.concat([self.data]*repeats, ignore_index=True)
        else:
            data = DataFrame()
        data = data.append(self.data[:remainder], ignore_index=True)
        self.local_step = self.current_max
        return data

    #@contextmanager
    #def __iter__(self):
        #self.start()
        #while self.dir_iter is not None:
            #self.update_data_block()
            #yield self.get_data_block()
        #self.reset()
    
    def set_headers(self, headers=None):
        """
        Look up the file headers in the first protocol path, 
        identify which ones match up with the requested headers (if none provided,
        use the keys of self.file_filters)
        
        Matching headers are saved in self.headers and the translation 
        between requested header/file_filter and actual header is stored in self.header_mapping
        """
        # set the file headers according to an actual data file
        prot = self.protocol.get_dirs()[0]
        filepath = os.path.join(self.base_dir, prot, self.suffix)
        self.file_headers = LogReader.check_headers(filepath)

        if headers is None:
            headers = list(self.file_filters.keys())

        # figure out which ones we care about
        # and which ones need custom mappings
        self.header_mapping = {}
        self.headers = []
        for h in self.file_headers:
            if h in headers:
                self.headers.append(h)
                # we won't store the mapping if we don't need to
            else:
                parts = h.split('-')
                if parts[0] in headers:
                    self.header_mapping[h] = parts[0]
                    self.headers.append(h)
            # could do other conversion checking here
    
    def reset(self):
        """
        Resets all the tracking values
        """
        self.dir_iter = None
        self.current_dir = None
        self.step = -1
        self.local_step = -1
        self.data = None

    def start(self):
        """
        Reset everything, then start up the protocol iterator
        and set the global step counter
        """
        self.reset()
        self.dir_iter = self.protocol.__iter__()
        self.step = 0

    def update(self):
        """
        Ensure that the correct data block is loaded,
        grab the current data,
        then increment the steps and return the data
        """
        self.update_data_block()
        data = self.get_data()
        self.step += 1 # global count
        self.local_step += 1 # count for the current protocol
        return data
    
    def get_data(self, index=None):
        """
        Return the data for the index passed (loops allowed, defaults to the current local time index)
        """
        if self.data is None:
            return None
        if index is None:
            index = self.local_step % len(self.data.index)
        return self.data.ix[index]

    def update_data_block(self):
        """
        Check to see if we need to load a new block of data
        And then set self.raw_data and self.data appropriately
        """
        # if we haven't started yet
        if self.dir_iter is None:
            self.start()
        # if we have some data loaded and we're below the max (or don't have a max) just keep going
        if self.data is not None and ((self.current_max == 0 and \
                                       self.local_step < len(self.data.index)) or \
                                      (self.current_max > 0 and \
                                       self.local_step < self.current_max)):
            return
        # otherwise we need to grab a datablock
        # grab the next protocol from the iterator, if possible
        try:
            prot = next(self.dir_iter)
            self.current_dir = prot.dirname
            self.current_max = prot.steps
            self.local_step = 0
        except StopIteration:
            # means we're done with our current data block and don't want another
            self.reset()
            return 
        self.raw_data = self._get_data_block()
        self.data = self._filter_data_block()    
    
    def _get_data_block(self, headers=None, directory=None, num_lines=None):
        """
        Load num_lines of raw_data from the specified directory according to the requested headers.
        Default to all headers, the current protocol directory and max
        """
        if headers is None:
            headers = self.headers
        if not directory:
            if not self.current_dir:
                self.update_data_block()
            directory = self.current_dir
        if self.current_max and self.current_max > 0:
            num_lines = self.current_max
        else:
            num_lines = None
            
        # load the data block for the current protocol and given headers
        filepath = os.path.join(str(directory), self.suffix)
        raw_data = LogReader.read_log(filepath, base_dir=self.base_dir, 
                                  headers=headers, 
                                  nrows=num_lines, 
                                  clean_file_data=self.clean_file_data)
        return raw_data
    
    def _filter_data_block(self, data=None, data_filters=None):
        """
        Fully parse the data passed (defaults to self.raw_data) according to 
        whatever filters (defaults to self.filters)
        """
        if data is None:
            #TODO: this may be inefficient
            data = self.raw_data.copy()
        if data_filters is None:
            data_filters = self.file_filters

        for h in data.columns:
            header = self.header_mapping.get(h, h)
            for func in data_filters[header]:
                data[h] = func(data[h])
        return data
  

if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=False)
    print("Done doctests")
    