import inspect
import os
import re
from inspect import Parameter, Signature
from pandas import DataFrame, Series
from collections import OrderedDict, namedtuple
from logfiles import LogReader, LogWriter
import argparse
import pickle
from datasets import *
from helpers import *

class empty:
    pass

def is_empty(val):
    return val == inspect._empty or val == Parameter.empty or val == empty

class Parsable(Parameter):
    """
    Data field storage for the logger.
    
    positional arguments occurs in the order it appeared in the _field list
    all non-positional arguments are grouped together at the end
    
    required, keyword, positional, transient flags have to be treated 
    differently than default parameter or argparse meaning because 
    of the inclusion of structs
    
    required 
        will raise a type error if they do not have a default 
        and are not provided (or if it is a struct, if its required
        parameters are not provided)
    positional 
        applies only to dirstring construction, the signature and 
         parser have no positional arguments. Positional arguments 
         get their own directory in the dirstring
    keyword
        if true, include the name in its string format (NAME-VALUE) instead of
        alone (VALUE)

    Optional extra parameters are stored in self.kwargs
    """
    _expected_type = empty

    def __init__(self, name, kind=Parameter.POSITIONAL_OR_KEYWORD,
                 required=False, keyword=True, positional=False,
                 constructor=empty, transient=False, constant=empty,
                 **kwargs):
        default = kwargs.pop('default', empty)
        self.kwargs = kwargs
        self.init_kind = kind
        self.required = required # does running the experiment require it
        self.transient = transient # is it necessary for defining the results
        self.positional = positional # is its value set by its position in dirstring
        self.keyword = keyword # is its keyword included in the dirstring
        self.constant = constant # is it a sentinel value
        
        self.choices = kwargs.get('choices', None) # is the value restricted
        self.nargs = kwargs.get('nargs', None)
        
        # nargs might have implicitly determined required
        if self.nargs:
            if self.nargs == '+':
                self.required = True
            else:
                self.required = False
        
        self.constructor = constructor
        super().__init__(name, kind=Parameter.POSITIONAL_OR_KEYWORD, 
                         default=default, annotation=self.kwargs)

    def add_parser_arg(self, parser):
        """
        Add yourself to the parser that is passed, even if not included in the dirstring
        
        Include required, default, and type arguments as applicable
        Optional kwargs let you customize/override the add_argument for special types
        """
        # TODO: should make sure only valid add_argument keywords used
        kwargs = self.kwargs.copy()
        if not is_empty(self.default):
            kwargs['default'] = self.default
        kwargs.setdefault('required', self.required)
        if not is_empty(self._expected_type):
            # TODO: should check what types translate naturally --- add warning
            # from docs
            # type= can take any callable that takes a single string argument and returns the converted value:
            kwargs['type'] = self._expected_type
        parser.add_argument("--{}".format(self.name), **kwargs)

    def stringify(self, value, code='-', remove_name=False):
        """
        Return an appropriately formatted string for your parameter settings.
        """
        if not isinstance(value, str):
            value = clean_string(value)
        if (self.keyword and not remove_name) or code=='arg':
            parts = format_param(name=self.name, value=value, code=code)
        else:
            parts = format_param(value=value, code=code)
        
        if isinstance(parts, str):
            return parts
        else:
            return format_join(parts, code=code)
  
    def de_string(self, value):
        """
        Return an appropriate literal evaluation of the passed value
        """
        return de_string(value)
  
    def get_arg_string(self, instance):
        """
        Return the command-line argument representation of the parameter
        with the value taken from the object passed
        """
        val = getattr(instance, self.name, empty)
        if is_empty(val): # or val == self.default:
            return ''
        if not isinstance(val, str):
            val = clean_string(val)
        return self.stringify(val, code='arg')

    def copy_self(self, instance, from_instance):
        """
        Copy the relevant attribute from from_instance to instance
        """
        if hasattr(from_instance, self.name):
            self.set_self(instance, {k: getattr(from_instance, self.name)})
        with suppress(TypeError, KeyError):
            self.set_self(instance, {k: from_instance[self.name]})
    
    def convert_val(self, val):
        if not is_empty(self._expected_type) and not isinstance(val, self._expected_type):
            try:
                val = self._expected_type(val) #should override this for bools/None
            except:
                raise TypeError('Could not convert {} to {}'.format(val, self._expected_type))
        elif not is_empty(self.constructor) and not isinstance(val, self.constructor):
            val = self.constructor.from_wildcard(val)
        """
        if val == 'None':
            val = None
        elif val == 'True':
            val = True
        elif val == 'False':
            val = False
        """        
        if self.choices and val not in self.choices:
            raise TypeError('{} must be one of {}'.format(self.name, self.choices))
        return val
        
    def set_self(self, instance, kwargs):
        """
        Use this to set values with explicit type checking and all
        
        Uses the kwarg dictionary to set self.name value on the given instance.
        If no valid key in dictionary, uses own default.
        
        If you don't have a default *and* no key is provided, you should complain. Also it shouldn't happen.
        
        Might want to turn this into an official generator, except for the performance overhead.
        """
        if self.name in kwargs:
            val = kwargs[self.name]
        elif not is_empty(self.constructor) and 'kwargs' in kwargs:
            try:
                val = self.constructor.from_wildcard(kwargs['kwargs'])
            except TypeError as e:
                if not self.required:
                    return
                else:
                    raise e
        elif not is_empty(self.default):
            val = self.default
        elif not self.required:
            return
        else:
            raise TypeError('Missing required keyword {}'.format(self.name))

        if not self.nargs:
            val = self.convert_val(val)
        elif self.nargs:
            val = listify(val)
            val = [self.convert_val(v) for v in val]
            
            if isinstance(self.nargs, int):
                if len(val) < self.nargs:
                    raise TypeError('{} requires {} values'.format(self.name, self.nargs))
                elif len(val) > self.nargs:
                    val = val[:self.nargs]
                    print('Dropped excess arguments from {}'.format(self.name))
                    
        setattr(instance, self.name, val)

class String(Parsable):
    _expected_type = str
    
class Integer(Parsable):
    _expected_type = int

class Float(Parsable):
    _expected_type = float


class Dir(Parsable):
    _expected_type = str
    # TODO: add set_parameter function to check directory
    # TODO: might be able to have expected_type be something file specific


class Boolean(Parsable):
    _expected_type = bool
    """
    Deal with the fact that argparse doesn't properly handle boolean strings
    In the parser set as 
        ---ARG_off
    or
        ---ARG
    """
    # TODO: add add_parser_arg and set_parameter functions

    def add_parser_arg(self, parser):
        """
        Booleans require flags in the parser args, can't be strings
        """
        bl = parser.add_mutually_exclusive_group(required=False)
        bl.add_argument('--{}_off'.format(self.name), action='store_const',
                        dest=self.name, const=False, default=self.default)
        bl.add_argument('--{}'.format(self.name), action='store_true', default=self.default)

    def get_arg_string(self, instance):
        val = getattr(instance, self.name)
        if val:
            return "--{}".format(self.name)
        else:
            return "--{}_off".format(self.name)
        
    def set_parameter(self, instance, kwargs):
        if self.name in kwargs and isinstance(kwargs[self.name], str):
            kwargs[self.name] = ast.literal_eval(kwargs[self.name])
        super().set_parameter(instance, kwargs)


class Listable(Parsable):
    """
    Listable parameters can be passed by string or as a list of separate items

    In the parser they take the form
        ---ARGs val1 val2 val3
    or
        ---ARG val1-val2-val3
    """
    def add_parser_arg(self, parser):
        # TODO: should not fail silently if parameter has annotations
        tmp = parser.add_mutually_exclusive_group(required=self.required)
        tmp.add_argument('--{}'.format(self.name), dest=self.name)
        tmp.add_argument('--{}s'.format(self.name), dest=self.name, nargs='+')

    def set_self(self, instance, kwargs):
        # TODO: straighten out Listable parameter definitions---consistency and parsability
        if self.name in kwargs:
            val = kwargs[self.name]
        elif not is_empty(self.default):
            val = self.default
        elif self.required:
            raise ValueError("Can't set {}, not available in kwargs".format(self.name))
        else:
            val = empty
        if isinstance(val, str) and self.constructor != str:
            setattr(instance, self.name, self.de_string(val))
        else:
            # might need to do some more processing here of the elements of the list
            setattr(instance, self.name, listify(val))
    
    def de_string(self, value):
        val = listify(de_string(value, suppress_eval=True))
        return [de_string(v) for v in val]


class Keyed(Parsable):
    """
    Keyed parameters are passed by dictionaries
    In the parser they take the form (like separate lists for each key with key as first element)
        --ARG key1-val1-val2 key2-val1 key3-
    """
    _expected_type = dict

    def add_parser_arg(self, parser):
        tmp = parser.add_mutually_exclusive_group(required=self.required)
        tmp.add_argument('--{}'.format(self.name), nargs='+')

    def stringify(self, value, code='-', remove_name=False):
        if isinstance(value, str):
            return super().stringify(value, code=code, remove_name=remove_name)
        else:
            return super().stringify(clean_string(value), code=code, remove_name=remove_name)

    def de_string(self, value):
        return split_params(value)

    def set_self(self, instance, kwargs):
        if self.name in kwargs:
            val = kwargs[self.name]
        elif not is_empty(self.default):
            val = self.default
        elif not self.required:
            val = empty
        else:
            raise ValueError("No {} parameter in kwargs".format(self.name))
        
        if isinstance(val, str):
            val = split_params(val)
        elif not isinstance(val, dict) and not is_empty(val):
            with suppress(TypeError):
                val = split_params('_'.join(val))
        setattr(instance, self.name, val)

kwargs = Parameter('kwargs', kind=Parameter.VAR_KEYWORD)

def has_nonpos_nonkey(params):
    possibilities = [k.name for k in params if isinstance(k, Parsable) and not k.positional and not k.keyword and not k.transient]
    if len(possibilities) == 0:
        return None
    elif len(possibilities) > 1:
        raise TypeError("Only one non-positional non-keyword parameter allowed per Structure")
    else:
        return possibilities[0]

class StructureMeta(type):
    """
    Pulls the arguments from a _fields class variable
    """
    def __new__(cls, clsname, bases, clsdict):
        param_list = [f for f in clsdict.get('_fields', [])]
        param_list.append(kwargs)
        clsdict['__signature__'] = Signature(param_list)
        return super().__new__(cls, clsname, bases, clsdict)


class Structure(metaclass=StructureMeta):
    """
    A structure groups a bunch of Parsables and allows for
    lossless conversion between instances, dictionaries, command line arguments, dirpaths, strings
    
    _fields is a list of the Parsables it holds
    _inherited is a list of variable names it copies from somewhere else 

    The dirpath structure:
       - the final directory is always the non-positional parameters (if any), alphabetically sorted
       - positional parameters each reside in their own directory in fixed order
       - only one non-keyword non-positional argument can be recovered
       - does not included any transient parameters
f
    Inherited parameters are not stored as part of the dirstring, 
      - but if they are there it shouldn't cause problems


    To get access to direct saving/loading functions, define the log file
    and either define or inherit the base_dir

    @property
    def log_file(self):
        return "data.txt"    
    """
    _fields = []
    _inherited = ['base_dir']
    constructor = type(None)
    
    log_data = False

    def __init__(self, *args, **kwargs):
        """
        Create an instance of this kind of structure according to its _fields
        Try to set inherited instance variables according to extra keywords
        """
        #print("calling init from", self, self._hello)
        bargs = self.__signature__.bind(*args, **kwargs).arguments
        for name, value in self.__signature__.parameters.items():
            #if isinstance(value, Struct):
            #    value.set_self(self, kwargs)
            if name != 'kwargs':
                value.set_self(self, bargs)
        # set inherited properties
        self.set_inherited(complain=False, **kwargs)

    def set_inherited(self, instance=None, complain=False, **kwargs):
        """
        Set inherited values according to either the instance or 
        the kwargs passed
        
        Checks kwargs first so you can use this to override instance values
        """
        for name in self._inherited:
            if kwargs and name in kwargs:
                val = kwargs[name]
            elif instance:
                val = getattr(instance, name)
            elif complain:
                raise ValueError("Need to set {} inherited field from instance or kwargs".format(name))
            else:
                continue
            setattr(self, name, val)    

    @property
    def log_dir(self):
        """
        Constructs log directory from the structure specifications
        """
        return self.get_dir_string()
    
    @property 
    def log_path(self):
        return os.path.join(self.base_dir, self.log_dir)
    
    @property
    def log_file_path(self):
        return os.path.join(self.log_path, self.log_file)
    
    @property
    def name(self):
        return self.log_dir
    
    def set_base_dir(self, base_dir):
        self.base_dir = base_dir
            
    @property
    def log_exists(self):
        """
        Check if log file exists already.
        Could do more checking of the log here or let the child classes
        overwrite to do their own checking
        """
        if self.loggable:
            return os.path.exists(self.log_file_path)
        else:
            return False
    
    @property
    def loggable(self):
        """
        Find out it this Structure can log itself
        """
        return self.log_data and hasattr(self, 'log_file')

    @property
    def data(self):
        if hasattr(self, '_data'):
            return self._data
        elif self.log_exists:
            return self.load_data()
        else:
            print("Attempting to automatically generate data")
            return self.run()

    #@profile
    def load_data(self, base_dir=None, **kwargs):
        """
        Load the data into self._data and return it
        """
        if base_dir:
            self.set_base_dir(base_dir)
        if hasattr(self, '_data'):
            return self._data

        if not self.loggable:
            raise TypeError("{} is not able to log".format(self.name))
        if not self.log_exists:
            raise TypeError("{} has not been logged".format(self.name))
        
        kwargs.setdefault('sep', ' ')
        if kwargs.get('squeeze', False):
            kwargs.setdefault('header', None)
        if not base_dir:
            base_dir = self.base_path
        self._data = LogReader.read_log(self.log_file, 
                                       base_dir=base_dir,
                                       **kwargs)
        return self._data
       
    def save_data(self, data=None):
        """
        If provided, overwrite internal data structure with passed data 
        TODO: probably should do some error checking here
        TODO: make sure models and log files are handled the same way
        Try to write _data to disk.
        """
        if not self.loggable:
            raise TypeError("{} cannot write logs.".format(self.name))
        
        if data is None:
            if not hasattr(self, '_data'):
                raise TypeError("{} has no _data to write".format(self.name))
            data = self._data
        
        if self.log_exists:
            print("Log exists, turn off safe mode to overwrite")
            return False
        
        os.makedirs(self.log_path, exist_ok=True)
        with open(self.log_file_path, 'wt') as fp:
            fp.write("# {}\n".format(self.log_file_path))
            self._data.to_csv(fp, sep=' ', index=False)
        return self.log_file_path

 
    @classmethod
    def _has_keyword(cls):
        return any([(f.keyword and not f.transient and not isinstance(f, Struct)) \
                    for f in cls._iter_params()])
    
    @classmethod
    def _has_npnk(cls):
        return has_nonpos_nonkey(cls._iter_params())

    @classmethod
    def from_wildcard(cls, wildcard):
        if isinstance(wildcard, argparse.Namespace):
            return cls.from_namespace(wildcard)
        elif isinstance(wildcard, list):
            return cls.from_args(wildcard)
        elif isinstance(wildcard, str):
            if '--' in wildcard:
                return cls.from_string(wildcard)
            else:
                return cls.from_dir_string(wildcard)
        elif isinstance(wildcard, dict):
            return cls(**wildcard)
        else:
            raise TypeError("Do not know how to construct {} from {}".format(cls, type(
                                                                                      wildcard)))

    @classmethod
    def from_args(cls, args):
        # TODO: the problem here is get_parser includes all args
        parser = cls.get_parser()
        # TODO: going to have to do this in a method call when args need 
        # to be determined at runtime
        # we only care about known args
        ns = parser.parse_known_args(args)[0]
        return cls.from_namespace(ns)
        
    @classmethod
    def from_string(cls, string):
        return cls.from_args(string.split(' '))

    @classmethod
    def from_namespace(cls, ns):
        return cls(**vars(ns))

    @classmethod
    def from_dir_string(cls, dstring):
        return cls(**cls.params_from_dir_string(dstring, deep_eval=True))

    @classmethod
    def params_from_dir_string(cls, dstring, deep_eval=False):
        """
        Extract the parameters (in command-line passable form) 
        from the directory string
        """
        parts = os.path.splitext(dstring)
        istring = dstring
        #TODO: figure out how to normalize a dirstring
        if 'txt' in parts[1]:
            dstring = os.path.dirname(dstring)
        keyworded = []
        npnk = None
        dstring = os.path.normpath(dstring)
        
        # get keyworded params from the end if necessary
        if cls._has_keyword():
            (dstring, kstring) = os.path.split(dstring)
            if cls._has_npnk():
                # this is a bit of a hack right now
                kstring = '-'.join([cls._has_npnk(), kstring])
            params = split_params(kstring, deep_eval=False)
            keyworded = [k for k in params]
        elif cls._has_npnk(): #no keyword, but a non-positional
            (dstring, pstring) = os.path.split(dstring)
            params = {cls._has_npnk(): pstring}
        else:
            params = {}
        
        # find the positional parameters
        for v in reversed(list(cls._iter_params())):
            if v.transient:
                continue
            elif isinstance(v, Struct):
                try:
                    sparams = v.constructor.params_from_dir_string(dstring, 
                                                                   deep_eval=deep_eval)
                except ValueError as e:
                    if not v.required:
                        continue
                    else:
                        raise e
                dstring = sparams.pop('base_dir')
                params.update(sparams)
            elif v.positional:
                (dstring, name) = os.path.split(dstring)
                if v.keyword:
                    # have to take off the keyword argument
                    parts = de_string(name)
                    if parts[0] != v.name:
                        if v.required:
                            raise ValueError("Dirstring {} has no value for {}".format(istring, v.name))
                        else:
                            # it was optional and needs to be parsed for the correct parsable
                            dstring = os.path.join(dstring, name)
                            continue
                    if len(parts) > 2:
                        params[v.name] = v.stringify(parts[1:])
                    else:
                        params[v.name] = parts[1]
                else:
                    params[v.name] = name
            elif v.keyword: # keyworded and not positional means it should have been in the keyword splitting
                try:
                    keyworded.remove(v.name)
                except ValueError as e:
                    # we don't need to complain unless it's a necessary value
                    # but somebody else might have added it too
                    if v.required and v.name not in params:
                        raise e
                    else:
                        continue
            elif not v.transient: # this is our npnk case
                npnk = v
            if deep_eval and not v.transient and not isinstance(v, Struct) and not v._expected_type == str:
                params[v.name] = v.de_string(params[v.name])
                
        # check if it is a sentinel value
        if not is_empty(v.constant) and params[v.name] != v.constant:
            raise ValueError("Invalid value ({}) for {}".format(params[v.name], v.name))
        # set the non-positional non-keyword parameter if necessary
        if npnk and npnk.name not in params:
            if len(keyworded) < 1:
                raise ValueError("Dirstring {} has no value for {}".format(istring, npnk.name))
            elif len(keyworded) > 1:
                raise ValueError("Dirstring {} has too many values for {}".format(istring, npnk.name))
            else:
                val = params.pop(keyworded[0])
                if deep_eval:
                    val = npnk.de_string(val)
                params[npnk.name] = val
        
        params['base_dir'] = dstring
        return params

    def pretty_print(self, ignored_keys=None, code='-'):
        """
        Construct a nicely formatted descriptor of this Structure object
        """
        ignored_keys = listify(ignored_keys)
        parts = []
        kwargs = {}
        for v in self._iter_params():
            if v.name in ignored_keys or v.transient:
                continue
            elif isinstance(v, Struct):
                parts.append(getattr(self, v.name).pretty_print(ignored_keys=ignored_keys, code=code))
            elif v.keyword:
                kwargs[v.name] = getattr(self, v.name)
            else:
                parts.append(format_param(v.name, value=getattr(self, v.name), code=code))
            parts.append(clean_string(kwargs))
        return format_join(parts, code=code)

    def get_nonpos_string(self):
        args = []
        npnk = None
        for v in self._iter_params():
            if v.transient or v.positional or isinstance(v, Struct):
                continue 
            val = getattr(self, v.name, empty)
            if is_empty(val):
                if v.required:
                    raise TypeError("How did this get constructed without a required parameter?")
                continue 
            # convert to string
            if not v.keyword: #then we have the single allowed one
                if npnk is not None:
                    raise TypeError("Structures cannot have multiple non-keyword params in dirstring")
                npnk = v.stringify(val, code='-')
            else:
                args.append(v.stringify(val, code='-'))
        if args:
            args = sorted(args)
        if npnk is not None:
            args.insert(0, npnk)

        if args:
            return format_join(args, '_')
        else:
            return ''
        
    def get_dir_string(self):
        args = []
        for v in self._iter_params():
            if v.transient: # we don't care
                continue
            if not isinstance(v, Struct) and not v.positional: # grabbed by the other function
                continue
            
            val = getattr(self, v.name, empty)
            if is_empty(val):
                if v.required:
                    raise TypeError("How did this get constructed without a required parameter?")
                continue 
            
            #if isinstance(v, Struct):
            #    s = val.get_dir_string()
            #    if v.keyword:
            #        args.append(v.name)
            #    args.append(s)                   
            #else:
                #try:
                #    val = val.get_dir_string()
                #except:
                #    val = v.stringify(val, code='-')
            val = v.stringify(val, code='-')
            assert isinstance(val, str)
                
            if v.positional or isinstance(v, Struct):
                args.append(val)
            else:
                if npnk:
                    raise TypeError("Structures cannot have multiple non-keyword params in dirstring")
                npnk = val

        key_string = self.get_nonpos_string()
        if key_string:
            args.append(key_string)
        if args:
            return os.path.join(*args)
        else:
            return ''
    

    def get_dir_string_old(self):
        args = []
        nonpos = []
        npnk = None
        for v in self._iter_params():
            if v.transient:
                continue
            elif not hasattr(self, v.name) and not v.required:
                continue
            elif isinstance(v, Struct):
                s = getattr(self, v.name).get_dir_string()
                if v.keyword:
                    args.append(v.name)
                args.append(s)                   
            elif is_empty(getattr(self, v.name)):
                continue
            else:
                val = getattr(self, v.name)
                try:
                    val = val.get_dir_string()
                except:
                    val = v.stringify(val, code='-')
                assert isinstance(val, str)
                
                if v.positional:
                    args.append(val)
                elif not v.keyword:
                    npnk = val
                else:
                    nonpos.append(val)
        nonpos = sorted(nonpos)
        if npnk:
            nonpos.insert(0, npnk)
        if nonpos:
            nonpos_dir = format_join(nonpos, '_')
            args.append(nonpos_dir)
        if args:
            return os.path.join(*args)
        else:
            return ''

    def get_record(self):
        """
        Return a Series object containing all the fields
        of this object
        """
        print("Deprecated")
        return Series({v.name: getattr(self, v.name) \
                       for v in self._iter_params() \
                       if not v.transient})

    def create(self):
        """
        Return an instance of the thing you are supposed to make
        """
        return self.constructor(**vars(self))

    @classmethod
    def get_parser(cls):
        parser = argparse.ArgumentParser()
        cls.add_parser_args(parser)
        return parser

    @classmethod
    def add_parser_args(cls, parser):
        for v in cls._iter_params():
            v.add_parser_arg(parser)

    @classmethod
    def _iter_keys(cls):
        for v in cls._iter_params():
            yield v.name

    @classmethod
    def _iter_params(cls):
        for v in cls._fields:
            yield v

    @classmethod
    def _iter_base_params(cls):
        for v in cls._fields:
            if isinstance(v, Struct):
                yield from v.constructor._iter_base_params()
            elif not v.transient:
                yield v

    def matches_param(self, param, drop_vals=None, assert_vals=None):
        key = param.name
        val = param.stringify(getattr(self, key), remove_name=True)
        if not drop_vals:
            drop_vals = {}
        if not assert_vals:
            assert_vals = {}
            
        # if this key is in the drop list and its value is one of the options, not a match
        if key in drop_vals and val in listify(drop_vals[key]):
            return False
        # if this key is in the must-have list and its value is not given, not a match
        if key in assert_vals and val not in listify(assert_vals[key]):
            return False
        
        if isinstance(param, Struct):
            return getattr(self, key).matches(drop_vals=drop_vals, assert_vals=assert_vals)
        
        return True        

    def get_parameter(self, key):
        """
        Return the named parameter for this object (doing deep search)
        If no such parameter is found, return a generic one.
        """
        for k in self._iter_base_params():
            if k.name == key:
                return k
        else:
            return Parsable(key)
    
    def matches(self, drop_vals=None, assert_vals=None):
        if assert_vals is None:
            assert_vals = {}
        if drop_vals is None:
            drop_vals = {}
        for k in assert_vals:
            if not hasattr(self, k):
                return False
            val = getattr(self, k)
            strval = self.get_parameter(k).stringify(val)
            if val not in listify(assert_vals[k]) and strval not in listify(assert_vals[k]):
                return False
        for k in drop_vals:
            if not hasattr(self, k):
                continue
            val = self.get_parameter(k).stringify(getattr(self, k))
            if val in listify(drop_vals[k]):
                return False
        #for v in self._iter_params():
        #    if not self.matches_param(v, drop_vals=drop_vals, assert_vals=assert_vals):
        #        return False
        # make sure we have every asserted key
        return True
    
    def get_flat_record(self, coll_strings=False):
        d = {}
        for v in self._iter_base_params():
            if coll_strings and (isinstance(v, Keyed) or isinstance(v, Listable)):
                val = v.stringify(getattr(self, v.name), remove_name=True)
            else:
                val = getattr(self, v.name)
            d[v.name] = val
        return Series(d)

    def items(self):
        for v in self._iter_keys():
            yield (v, getattr(self, v))

    def get_arg_string(self):
        return " ".join([p.get_arg_string(self) \
                         for p in self._fields])

#TODO: make model saver and log saver mixins

class Experiment(Structure):
    """
    TODO: make this an abstract class so that call must be defined
    TODO: move the data logging into here as well, maybe?

    Consolidates the features that are useful in experiment structures
    Allows for variable numbers of Structs
    
    The __call__ function of each paticular experiment defines what happens
    """
    log_data = True
    log_model = False
    
    def __enter__(self):
        base_path = os.path.normpath(os.path.join(self.base_dir, 
                                                  self.get_nonpos_string()))
        
        self.input_dim = None
        for s in self._iter_params():
            if isinstance(s, Struct) and hasattr(self, s.name):
                child = getattr(self, s.name)
                if s.keyword:
                    base_path = os.path.normpath(os.path.join(base_path, s.name))
                child.set_inherited(base_dir=base_path, instance=self)
                with suppress(AttributeError):
                    child.__enter__()
                base_path = child.log_path
                with suppress(AttributeError):
                    self.input_dim = child.output_dim
                self.get_inherited(s, complain=True)
                    
                # see if it needs to share anything
        # this will not be true if it has its own positional/nonpos arguments
        #assert base_path == self.log_path
        return self
    
    def __exit__(self, *args):
        for s in self._iter_keys():
            with suppress(AttributeError):
                getattr(self, s).__exit__()
        if self.log_model:
            self.save_model()
            
    def get_inherited(self, child, complain=False):
        """
        Given a Structure object, pull in whatever inherited features
        """
        if child.broadcast and hasattr(self, child.name):
            content = getattr(self, child.name)
            for att in child.broadcast:
                #TODO: could use a dictionary here if we want to translate things
                try:
                    setattr(self, att, getattr(content, att))
                except AttributeError as e:
                    if complain:
                        raise e
            
    # TODO: this is the batch approach, should probably have both incremental
    # and batch standardized somehow
    def run(self):
        with self as setup:
            return setup()
        
    def __call__(self):
        raise NotImplementedError("Must define each experiment's call method")
        
    def load_model(self):
        """
        Load the model if possible. If not, run the experiment to generate it
        """
        if hasattr(self, '_model'):
            return self._model
        elif self.log_model:
            if not self.model_exists:
                print("Generating model...")
                self.run()
                return self._model
            else:
                return pickle.load(open(self.model_file_path, 'rb'))
        else:
            raise NotImplementedError("{} does not have a model".format(self.name))
              
    def save_model(self, model=None):
        """
        Save the passed object as a pickled file
        By default saves the _model value
        """
        if model is None:
            if not hasattr(self, '_model'):
                raise TypeError("{} has no model to save.".format(self.name))
            model = self._model
        
        if self.model_exists:
            print("Model previously saved, turn off safe mode to overwrite")
            return False

        os.makedirs(self.log_path, exist_ok=True)
        pickle.dump(model, open(self.model_file_path, 'wb+'))
        return self.model_file_path
    
    @property
    def model_file_path(self):
        return os.path.join(self.log_path, self.model_file)
 
    @property
    def model_exists(self):
        try:
            return os.path.exists(self.model_file_path)
        except AttributeError:
            return False
     
    
class Struct(Parsable):
    """
    A Parsable that holds a set of Parsables
    
    factory
        if provided, will be treated as the constructor for a class that takes
        the same parameters as this struct (can be overridden)
    broadcast
        a list of internal variables that can be shared
    
    """
    positional = True
    keyword = False
    broadcast = None
    constructor = None   
    
    def add_parser_arg(self, parser):
        # TODO: this is where to add the _string option (mutually exclusive)
        if self.constructor:
            self.constructor.add_parser_args(parser)
        else:
            raise TypeError(self.name+" is unknown type, cannot add parser args")
        
    def stringify(self, value, code='-', remove_name=False):
        assert code=='-'
        # Structs handle their keywords differently than regular parsables
        if isinstance(value, Structure):
            value = value.get_dir_string()
        elif not isinstance(value, str):
            raise ValueError("{} is not a stringable type".format(type(value)))
        
        if self.keyword and not remove_name:
            return os.path.join(self.name, value)
        else:
            return value
        #TODO: double-check that this follows the same logic as the super class
        #if not isinstance(value, str):
        #    value = value.get_dir_string()
        #return super().stringify(value, code=code, remove_name=remove_name)

    def get_arg_string(self, instance):
        structure = getattr(instance, self.name)
        return " ".join([p.get_arg_string(structure) for p in structure._fields])


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=False)
    print("Done doctests")