'''
BLINC Adaptive Prosthetics Toolkit
- Bionic Limbs for Improved Natural Control, blinclab.ca
anna.koop@gmail.com

A toolkit for running machine learning experiments on prosthetic limb data

This module handles read and writing log files in a standard format

# TODO: robustly handle old-style files
# TODO: test
'''
import os
import ast
import itertools
import pandas as pd
from pandas import Series, DataFrame
from helpers import *
from local import base_dir, test_dir
import learners
import sensors

def de_array(values):
    """
    Take a [list of] array string(s) (usually read from a log file) and 
    return a true array

    >>> de_array("[0, 1]")
    array([[0, 1]])
    >>> de_array(["[0, 0]", "[1, 1]"])
    array([[0, 0],
           [1, 1]])
    """
    values = listify(values)
    strings = '[' + ', '.join(values) + ']'
    return np.array(ast.literal_eval(strings))

def pathify(filepath, base_dir=None):
    if base_dir is None:
        return filepath
    else:
        return os.path.join(base_dir, filepath)

def peek_char(filepointer, num_chars=1):
    fpos = filepointer.tell()
    char = filepointer.read(num_chars)
    if char:
        filepointer.seek(fpos)
    return char

def get_dirs(base_dir):
    return [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

def get_filepaths(base_dir, filename='predictions.txt', partial=False, 
                  confirm_files=True, ignore_files=None):
    """
    Return a list of all the filepaths that end in the given filename
    in the specified directory tree
    """
    ignore_files = listify(ignore_files)
    paths = []
    for (dirpath, dirnames, filenames) in os.walk(base_dir):
        if filename in filenames:
            paths.append(os.path.join(dirpath, filename))
        elif partial:
            for f in filenames:
                if filename in f:
                    paths.append(os.path.join(dirpath, f))
    for p in ignore_files:
        with suppress(ValueError):
            paths.remove(p)
    return paths

def get_dir(obj, data):
    return clean_string({k: data[k] for k in obj.req_params})

def get_params(obj, filepath):
    (base, params) = os.path.split(dirname)
    if '.txt' in params:
        if params != cls.filename:
            raise ValueError("{} is not a valid filepath for {}".format(filepath, obj.__name__))
        (base, params) = os.path.split(base)
    return split_params(params)

#class Result():
    #"""
    #Lightweight object for handling conversion from log path to parameters
    #and back
    #"""
    #req_params = ('platform', 'protocol', 'pid', 
                  #'features', 'target_index', 'gamma')

    #@classmethod
    #def get_keys(cls):
        #return set(itertools.chain(cls.req_params,
                                   #learners.TDLambda.req_params,
                                   #sensors.TileCoder.req_params))

    #@classmethod
    #def get_experiment_dir(cls, data):
        #return os.path.join(data['platform'], data['protocol'], data['pid'])

    #@classmethod
    #def split_experiment_params(cls, filepath):
        #(base, pid) = os.path.split(filepath)
        #(base, protocol) = os.path.split(base)
        #(base, platform) = os.path.split(base)
        #return (base,
                #{'pid': pid, 'platform': platform, 'protocol':protocol})

    #@classmethod
    #def get_feature_dir(cls, data):
        #return clean_string(data['features'])

    #@classmethod
    #def split_feature_params(cls, filepath):
        #(base, features) = os.path.split(filepath)
        #return (base, de_string(features))

    #@classmethod
    #def get_target_dir(cls, data):
        ## TODO: generalize this
        #return "{}-{}_gamma-{}".format(data['features'], 
                                       #data['target_index'], 
                                       #data['gamma'])

    #@classmethod
    #def split_target_params(cls, filepath):
        #(base, tmp_params) = os.path.split(filepath)
        #gamma = tmp_params.pop('gamma')
        #(features, ind) = tmp_params.pop_items()
        #return (base, 
                #{'gamma': gamma, 'features':features, 'target_index': ind})

    #@classmethod 
    #def get_sensor_dir(cls, data):
        #return sensors.TileCoder.get_dir(data)

    #@classmethod
    #def split_sensor_params(cls, filepath):
        #(base, params) = os.path.split(filepath)
        #if ('.txt' in params):
            #(base, params) = os.path.split(base)
        #return (base, split_params(params))

    #@classmethod
    #def get_agent_dir(cls, data):
        #return learners.TDLambda.from_dict(data).dirname

    #@classmethod
    #def split_agent_params(cls, filepath):
        #(base, params) = os.path.split(filepath)
        #if '.txt' in params:
            #(base, params) = os.path.split(filepath)
        #(base, agent_type) = os.path.split(base)
        #params = split_params(params)
        #params['agent_type'] = agent_type
        #return (base, params)

    #@classmethod
    #def from_filepath(cls, filepath):
        #"""
        #Return a Result object based on the settings in filepath

        #filepath must be full path, either including the .txt file
        #for the particular agent or not
        #"""
        #(base, params) = cls.split_agent_params(filepath)
        #(base, t_params) = cls.split_target_params(base)
        #params.update(t_params)
        #(base, s_params) = cls.split_sensor_params(base)
        #params.update(s_params)
        #(base, f_params) = cls.split_feature_params(base)
        #params.update(f_params)
        #(base, e_params) = cls.split_experiment_params(base)
        #params.update(e_params)
        #return cls(base, **params)

    #def __init__(self, base_dir, **params):
        #"""
        #Create an Result object from the parameters passed
        
        #>>> r = Result('tmp', protocol='hand-wrist', pid='sub-1', target_index=0, lamda=.99, alpha=.1, gamma=.9, num_tilings=8, tile_width=.5)
        #>>> r.get_name()
        #'td/alpha-0.1_gamma-0.9_lamda-0.99 emg-0_gamma-0.9 num_tilings-8_tile_width-0.5'
        #"""
        #params.setdefault('platform', 'ninapro')
        #params.setdefault('features', 'emg')
        #self.data = Series(params)
        #self.base_dir = base_dir
        #self.set_filepath()

    #def set_filepath(self):
        #self.exp_dir = self.get_experiment_dir(self.data)
        #self.exp_path = os.path.join(self.base_dir, self.exp_dir)

        #self.feature_dir = self.get_feature_dir(self.data)
        #self.feature_path = os.path.join(self.exp_path, self.feature_dir)
        #self.sensor_dir = self.get_sensor_dir(self.data)
        #self.sensor_path = os.path.join(self.feature_path, self.sensor_dir)

        #self.target_dir = self.get_target_dir(self.data)
        #self.target_path = os.path.join(self.sensor_path, self.target_dir)
        #self.agent_dir = self.get_agent_dir(self.data)
        #self.agent_path = os.path.join(self.target_path, self.agent_dir)

        #self.data['err_log'] = os.path.join(self.agent_path, 'err.txt')
        #self.data['apred_log'] = os.path.join(self.agent_path, 'pred.txt')

    #def get_name(self):
        #return ' '.join([self.agent_dir, self.target_dir, self.sensor_dir])

    #def confirm_logfile(self):
        ##TODO: add checking of validity
        #return os.path.exists(self.data['err_log'])

    #def get_args(self):
        #"""
        #Return a list of arguments, formatted for the command-line parser,
        #that will generate this kind of object
        #"""
        #params = [' '.join(format_param(name=k, value=self.data[k], code='arg')) \
                  #for k in self.get_keys()]
        #return ' '.join(params)

    #def get_params(self):
        #"""
        #Return a dictionary of the necessary and sufficient parameters that define this object
        #"""
        #return {k: self.data[k] for k in self.get_keys()}

class LogReader():
    """
    Custom file handler for reading standard log files.
    """
    @classmethod
    def check_headers(cls, filepath, subset=None):
        with open(filepath, 'r') as fp:
            line = fp.readline()
            if not line:
                return []
            while line[0] == '#':
                line = fp.readline()

        file_headers = line.split()
        
        # pass back whatever we're supposed to use for headers
        if not subset:
            return file_headers

        # make sure we have the headers we need
        headers=[]
        needed = listify(subset)
        
        for h in file_headers:
            parts = h.split('-')
            if parts[0] in needed or h in needed:
                headers.append(h)
        return headers
        
    
    @classmethod
    def read_log(cls, filepath, base_dir=None, clean_file_data=False, 
                 headers=None, **kwargs):
        """
        Load the contents of the log into a dataframe object
        """
        filepath = pathify(filepath, base_dir)
        kwargs.setdefault('sep', ' ')
        kwargs.setdefault('skiprows', 1)
        kwargs.setdefault('index_col', None)
        kwargs.setdefault('header', 0)
        kwargs.setdefault('comment', '#')
        kwargs.setdefault('nrows', None)
        if headers:
            kwargs['usecols'] = headers
        
        #if clean_file_data:
        #    converters = {h: ast.literal_eval for h in headers}
        #else:
        #    converters = None
        # sqeeze
        # converters
        data = pd.read_csv(filepath, **kwargs)
        return data
    

    @classmethod
    def from_obj(cls, exp_obj, base_dir=None, headers=None, nrows=None):
        return cls(exp_obj.logpath, base_dir=base_dir, headers=headers, 
                   nrows=nrows)

    def __init__(self, filepath, base_dir=None, headers=None, nrows=None):
        self.fullpath = pathify(filepath, base_dir)
        self.headers = headers
        self.nrows = nrows
        self.comment_rows = 0
        self.comments = []
        self.data_rows = 0
        self.fp = open(self.fullpath, 'r')

    def skip_comments(self):
        """
        Move the file pointer past the comments at the top of the file
        """
        while peek_char(self.fp, 1) == '#':
            self.comments.append(self.fp.readline()[1:])
            self.comment_rows += 1
        # put back the last character you looked at

    def get_headers(self, subset=None, complain=False):
        """
        Extract headers from current line of the file (if possible)
        Note that headers must be valid variable names, at least the first one

        If complain is true, will raise an exception if all requested headers
        are missing
        """
        self.skip_comments()
        self.file_headers = None
        if peek_char(self.fp, 1).isnumeric():
            return
        headerline = self.fp.readline()
        if headerline:
            self.file_headers = headerline.split()
        if subset:
            print("Warning! Subset thingy not implemented")
        return self.file_headers

    def __iter__(self):
        """
        Return the next data line of data as a properly labeled series object
        """
        line = self.fp.readline().split()
        yield Series(data=line, index=self.file_headers)

    def __enter__(self):
        return self.fp

    def __exit__(self, *args, **kwargs):
        self.fp.close()



def reformat_data(dirpath):
    df = LogReader.read_log(os.path.join(dirpath, 'raw-data.txt'))
    with LogWriter('emg.txt', base_dir=dirpath) as f:
        DataFrame(de_array(df.emg)).to_csv(f, sep=' ', index=False)
    with LogWriter('glove.txt', base_dir=dirpath) as f:
        DataFrame(de_array(df.glove)).to_csv(f, sep=' ', index=False)
    with LogWriter('labels.txt', base_dir=dirpath) as f:
        df[['repetition', 'rerepetition', 'restimulus', 'stimulus']].to_csv(f, sep=' ', index=False)
                

class LogWriter():
    """
    Custom file handler for dealing with log files in a standard way.

    Creates relevant directory and writes filepath to log file, 
    if one does not exist already
    """
    @classmethod
    def write_log(cls, exp_obj, data, base_dir=None, safe_mode=True):
        """
        Write the data passed to the logfile for the object passed
        """
        print("Suspicious function")
        with cls.from_obj(exp_obj, base_dir=base_dir, safe_mode=safe_mode) as fp:
            fp.set_headers(exp_obj)
            data.to_csv(fp, sep=' ', index=False)
    
    @classmethod
    def from_obj(cls, exp_obj, base_dir=None, safe_mode=True):
        self = cls(exp_obj.logpath, base_dir=base_dir, safe_mode=safe_mode)
        return self

    def __init__(self, filepath, base_dir=None, safe_mode=True, headers=None):
        if base_dir:
            self.fullpath = os.path.join(base_dir, filepath)
        else:
            self.fullpath = filepath
        self.set_headers(headers)
        self.filepath = filepath
        self.safe_mode = safe_mode

        os.makedirs(os.path.dirname(self.fullpath), exist_ok=True)

        if self.safe_mode:
            mode = 'x'
        else:
            mode = 'w'
        try:
            self.fp = open(self.fullpath, mode=mode)
        except FileExistsError:
            print("File exists, turn off safe mode to overwrite.")
            self.fp = None
        else:
            self.comment(self.filepath)
            self.write_list(self.headers)

    def write_args(self, *args):
        return self.write_list(args)

    def write_list(self, value):
        if not is_empty(value):
            self.write(' '.join(value))

    def write(self, value):
        """
        Standardize writing for all data types
        """
        if self.fp:
            self.fp.write("{}\n".format(value))
            
    def write_dataframe(self, df):
        """
        Assuming headers have already been written, store the contents of 
        the dataframe
        """
        if self.num_columns == len(df.columns):
            self.fp.write("# {}\n".format(self.filepath))
            df.to_csv(self.fp, sep=' ', index=False)
        else:
            msg = "Headers {0} in dataframe do not match {1}".format(df.columns,
                                                                     self.headers)
            raise ValueError(msg)

    def set_headers(self, value=None):
        try:
            self.headers = value.headers
        except:
            self.headers = listify(value)
        if is_empty(self.headers):
            self.num_columns = 0
        else:
            self.num_columns = len(self.headers)

    def comment(self, value):
        self.write("# {}".format(value))

    def __enter__(self):
        return self.fp

    def __exit__(self, *args):
        self.close()
        
    def close(self):
        if self.fp:
            self.fp.close()


def file_complete(filepath, req_lines=0, req_headers=None, 
                  exception=False):
    """
    Check if the file exists and is non-empty. 
    Raise exception if exception==True otherwise return boolian

    If req_lines or req_headers, will confirm that file contains at least
    that many lines and/or those headers
    """
    if not filepath:
        if exception:
            raise ValueError("Did not provide filepath {}".format(filepath))
        return False

    if not os.path.exists(filepath):
        if exception:
            raise FileNotFoundError("No such file as {}".format(filepath))
        return False

    if req_headers:
        headers = get_headers(filepath, subset=req_headers)
        if not headers:
            if exception:
                raise Warning("File did not contain required header {}".format(rh))
            return False

    if req_lines and not file_length(filepath) >= req_lines:
        if exception:
            raise Warning("Incomplete file {}".format(filepath))
        return False
    return True

def get_headers(filepath, subset=None, skip_rows=1):
    """
    Return the columns in the file. If a subset is provided will return
    the elements therein that exist in the file checking for array munging
    """
    try:
        contents = pd.read_csv(filepath, nrows=1, skiprows=skip_rows,
                               error_bad_lines=False, sep=' ')
    except IndexError:
        return []

    if not subset:
        return contents.columns
    else:
        subset = listify(subset)

        newsubset=[]
        for s in subset:
            # check alternate formatting
            if s in contents.columns:
                # note duplicates are allowed
                newsubset.append(s)
                continue
            else:
                (prefix, suffix) = de_number(s)
                if prefix in contents.columns and prefix not in newsubset:
                    newsubset.append(prefix)
                    continue

                array_string = '{}[{}]'.format(prefix, suffix)
                if array_string in contents.columns:
                    newsubset.append(array_string)
                else:
                    print("Invalid header", s)
            # TODO: exception option
            # TODO: check for string-formatted arrays?
        return newsubset

def file_length(filepath, skip_comments=True):
    """
    Return the number of data rows in the file
    """
    if not os.path.exists(filepath):
        return -1
    if skip_comments:
        skiprows  = comment_length(filepath)
    else:
        skiprows = None
    contents = pd.read_csv(filepath, skiprows=skiprows, sep=' ', usecols=[0],
                           comment='#', error_bad_lines=False)
    return len(contents)

def read_log(filepath, subset=None, req_length=0, 
             binsize=None, error_bad_file=False, 
             skip_rows=1, check_headers=True,
             **kwargs):
    """
    Load the contents of the log into a dataframe object
    """
    if not file_complete(filepath, req_lines=req_length, req_headers=subset, 
                         exception=error_bad_file):
        return None
    # now we know a properly formatted file exists
    if skip_rows is None:
        skip_rows = comment_length(filepath)

    headers = get_headers(filepath, skip_rows=skip_rows, subset=subset)
    data = pd.read_csv(filepath, sep=' ',
                       skiprows=skip_rows, 
                       error_bad_lines=error_bad_file,
                       index_col=None,
                       usecols=headers,
                       names=headers,
                       header=0,
                       comment='#',
                       **kwargs)
    return data

if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=False)
    print("Done doctests")
    #df = LogReader.read_log(base_dir+'/ninapro/hand-wrist/sub-1/raw-data.txt')
    #print(LogReader(base_dir+'/ninapro/hand-wrist-100/sub-1/emg-rollmean-scaled_ranges-emg-0-5_window-10/num_tilings-8_tile_width-0.5/emg-1/gamma-0.99/td/alpha-0.01_lamda-0.99.txt').get_headers())