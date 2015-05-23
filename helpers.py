'''
helper functions not split into their own modules yet
'''
import os
import re
import ast
import itertools
import pandas as pd
import numpy as np
from contextlib import suppress
from collections import defaultdict


#############################################
## Math utilities
#############################################

def binary_boundary(t, d):
    """
    Return the last <t integer that is a multiple of 2^d
    
    >>> binary_boundary(11, 4)
    0
    >>> binary_boundary(11, 3)
    8
    >>> binary_boundary(11, 2)
    8
    >>> binary_boundary(11, 1)
    10
    
    >>> binary_boundary(15, 4)
    0
    >>> binary_boundary(16, 4)
    16
    >>> binary_boundary(15, 3)
    8
    >>> binary_boundary(15, 2)
    12
    >>> binary_boundary(15, 1)
    14
    """
    return t - (t & ((1<<d) - 1))

def at_partition_start(t, d):
    """
    Return True if t represents the beginning of a binary partition of depth d
    
    >>> at_partition_start(0, 0)
    True
    >>> at_partition_start(0, 1)
    True
    >>> at_partition_start(0, 34)
    True
    

    >>> at_partition_start(0, 1)
    True
    >>> at_partition_start(1, 1)
    False
    >>> at_partition_start(2, 1)
    True
    
    >>> at_partition_start(8, 2)
    True
    >>> at_partition_start(8, 3)
    True
    >>> at_partition_start(8, 4)
    False
    """
    return not(t & ((1<<d) - 1))

def mscb(t):
    """
    Find the index of the most significant change bit,
    the bit that will change when t is incremented
    aka the power of 2 boundary we are at

    >>> mscb(0)
    0
    >>> mscb(1)
    1
    >>> mscb(7)
    3
    >>> mscb(8)
    0
    """
    return (t^(t+1)).bit_length()-1

def log_sum_exp(*args):
    """
    Calculate the log of the sum of exponentials of the vector elements.

    Use the shifting trick to minimize numerical precision errors.

    similar to np.logaddexp(x1, x2) but takes a arbitrary number of elements.

    >>> log_sum_exp(np.log(0.5), np.log(0.5))
    0.0
    >>> approx(np.exp(log_sum_exp(np.log(0.001), np.log(0.009), np.log(0.3))), .31)
    True
    """
    m = max(args)
    vals = np.array(args) - m
    return m + np.log(sum(np.exp(vals)))



############################################
## Set operatiosn
##
## Useful additions to handling combinations of data
##
############################################
def unique_dict_sets(params):
    """
    Return a list of dictionaries with every unique combination of 
    parameters given.
    Expects a dictionary where the values are lists of parameter values
    """
    return (dict(zip(params.keys(), x)) for x in itertools.product(*params.values()))

def confirm_membership(required_set, proposed):
    """
    Return True if all elements in required set are present in proposed
    
    >>> confirm_membership({'a', 'b'}, {'c': 0, 'a': 2, 'b': 0})
    True
    >>> confirm_membership({'a':3, 'b':4}, {'c': 0, 'a': 2, 'b': 0})
    True
    """
    return set(required_set).issubset(proposed)

############################################
## Parameter utilities
##
## Simplify handling arguments in function calls
##
############################################
def listify(val, copy=False):
    if val is None:
        return []
    elif isinstance(val, str):
        return [val]
    elif isinstance(val, list) or \
         isinstance(val, pd.Series) or \
         isinstance(val, pd.DataFrame):
        return val

    try:
        t = [v for v in val]
        if copy:
            return t
        else:
            return val
    except TypeError:
        return [val]


#############################################
## Format utilities
##
## converts variables into strings that are valid for file names
## uses _ to separate variables and - for equality
## can handle _ in variable name, I think
## varName-value
## listName-item1-item2-item3
## var1-value_var2-value_list-item1-item2
##
##
#############################################
def clean_string(value):
    """
    Remove non-standard characters from the string.

    >>> clean_string('hello!')
    'hello'
    >>> clean_string('joint[5]')
    'joint5'
    >>> clean_string('hello_world')
    'hello_world'
    >>> clean_string(55)
    '55'
    >>> clean_string(['a', 'b', 'c'])
    'a-b-c'
    >>> clean_string({'a':1, 'b': [2, 3]})
    'a-1_b-2-3'
    """
    if isinstance(value, list):
        return '-'.join(clean_string(i) for i in value)
    if isinstance(value, dict):
        return '_'.join([format_param(k, value[k], '-', force_both=True) for k in sorted(value.keys())])
    return re.sub(r'[^a-zA-Z0-9_.\-]', '', str(value)) 

def de_string(value, suppress_eval=False):
    """
    Try to turn this into the most common value
    Will first try to use literal_eval to parse it unless suppress_eval is given

    >>> de_string('1')
    1
    >>> de_string('-1')
    -1
    >>> de_string('-1.0')
    -1.0
    >>> de_string('0-1-2')
    -3
    >>> de_string('.9')
    0.9
    >>> de_string('0.9')
    0.9
    >>> de_string('True')
    True
    >>> de_string('a-b-c-')
    ['a', 'b', 'c', '']
    >>> de_string('emg-0-4')
    ['emg', 0, 4]
    >>> de_string(None)
    >>> de_string('None')
    >>> de_string('')
    ''
    >>> de_string('False')
    False
    >>> de_string('a1')
    'a1'
    >>> de_string('-')
    '-'
    >>> de_string('a1-')
    ['a1', '']
    >>> de_string('-.9')
    -0.9
    >>> de_string(1)
    1
    >>> d = de_string('alpha-.9_alpha_decay--1')
    >>> d['alpha']
    0.9
    >>> d['alpha_decay']
    -1
    >>> d = de_string('pos-24_neg-16_neither-')
    >>> d['pos']
    24
    >>> d['neg']
    16
    >>> d['neither']
    ''
    >>> de_string('emg-rollingmean-trace-scaled')
    ['emg', 'rollingmean', 'trace', 'scaled']
    >>> '-'.join(de_string('emg-rollingmean-trace-scaled')[1:])
    'rollingmean-trace-scaled'
    >>> '-'.join(de_string('emg')[1:]) #PATHOLOGICAL CASE
    'm-g'
    >>> '-'.join(de_string('emg-rollingmean')[1:])
    'rollingmean'
    """
    if not suppress_eval:
        with suppress(ValueError, SyntaxError):
            return ast.literal_eval(value)
    #check for dictionary or list strings
    if not isinstance(value, str):
        with suppress(TypeError):
            return [de_string(v) for v in value]
        return value
    elif value.isnumeric():
        return int(value)
    elif '_' in value:
        return split_params(value)
    elif '-' in value and len(value) > 1:
        return de_string(value.split('-'))
    #if not value:
    #    return None
    #elif not isinstance(value, str):
    #    return value
    #elif value.isnumeric():
    #    return(int(value))
    #elif value == '-':
    #    return value
    #elif not value[0].isalpha() and \
    #     (value[1].isnumeric() or value[2].isnumeric()):
    #    return float(value) if '.' in value else int(value)
    #elif value == 'True':
    #    return True
    #elif value == 'False':
    #    return False
    #elif value == 'None':
    #    return None
    #elif '-' in value:
    #    if '_' in value:
    #        return split_params(value)
    #    else:
    #        return value.split('-')
    else:
        return value

dummy = object()

def de_number(value):
    """
    Take a string with a number appended and return the prefix-number pair

    >>> de_number('a1')
    ('a', 1)
    >>> de_number('a01')
    ('a', 1)
    >>> de_number('a-1')
    ('a', 1)
    >>> de_number('a-010')
    ('a', 10)
    >>> de_number('a0.1')
    ('a', 0.1)
    >>> de_number('a[1]')
    ('a[1]', None)
    >>> de_number('a-')
    ('a-', None)
    """
    if not value[-1].isnumeric():
        return((value, None))

    ind=-1
    while value[ind].isnumeric() or value[ind] == '.':
        ind-=1
    if value[ind] == '-':
        prefix = value[:ind]
    else:
        prefix = value[:ind+1]
    suffix = de_string(value[ind+1:])
    return (prefix, suffix)


def format_param(name=None, value=dummy, code='', force_both=False):
    """
    Standardize codes for display key, value pairs
    If force_both will display both name and value regardless of code

    >>> format_param('a', 1, code='-')
    'a-1'
    >>> format_param('a', 'b', code='=')
    "a='b'"
    >>> format_param('a', 1, code='v')
    '1'
    >>> format_param('a', 1, code='n')
    'a'
    >>> format_param('a', 1, code='n-', force_both=True)
    'a-1'
    >>> format_param('a', 1, code='v=', force_both=True)
    'a=1'
    >>> format_param('a', 1, code='=')
    'a=1'
    >>> format_param('a', 1, code='v')
    '1'
    >>> format_param(value=[1, 2, 3], code='v-')
    '1-2-3'
    >>> format_param(value=[1, 2, 3], code='v')
    '[1, 2, 3]'
    >>> format_param(value=[1, 2, '3'], code='v')
    "[1, 2, '3']"
    >>> format_param(value=[1, 2, 3], code='=')
    '[1, 2, 3]'
    >>> format_param(value=[1, 2, 3], code='=', force_both=True)
    Traceback (most recent call last):
    ...
    TypeError: Must provide value name

    >>> print(format_param('flist', [1, 2, 3], code='v'))
    [1, 2, 3]
    >>> print(format_param('flist', [1, 2, 3], code='='))
    flist=[1, 2, 3]

    >>> print(format_param('mybool', True, code='='))
    mybool=True
    >>> print(format_param('mybool', False, code='='))
    mybool=False

    >>> print(format_param('flist', [1, 2, 3], code='arg'))
    ['--flist', '1', '2', '3']
    >>> print(format_param('flag', True, code='arg'))
    ['--flag', 'True']
    >>> print(format_param('flag_on', code='arg'))
    --flag_on
    """
    #TODO: move to datasets file
    if not force_both and ('n' in code or value == dummy):
        if code == 'arg':
            return '--{}'.format(name)
        else:
            return name
    elif not force_both and ('v' in code or not name):
        if '-' in code:
            return clean_string(value)
        else:
            return "{!r}".format(value)

    if not name:
        raise TypeError("Must provide value name")

    if code == 'arg':
        args = ['--{}'.format(name)]
        if isinstance(value, list):
            args.extend(map(str, value))
        else:
            args.append(str(value))
        return args
    elif '-' in code:
        return "{}-{}".format(name, clean_string(value))
    elif '=' in code:
        return "{}={!r}".format(name, value)
    elif ':' in code:
        return "{}:{!r}".format(name, value)
    elif code == ' ':
        return "{} {}".format(name, value)
    else:
        return "{} {!r}".format(name, value)
    
def format_join(parts, code='_'):
    if '_' in code:
        return '_'.join(parts)
    elif '=' in code or ':' in code:
        return ', '.join(parts)
    elif code == '':
        return ''.join(parts)
    elif '/' in code or os.path.sep in code:
        return os.path.join(parts)
    else:
        return ' '.join(parts)
    

def format_protocol(value):
    if not value[1]:
        x = dummy
    else:
        x = int(value[1])
    return format_param(value[0], x, '-')

def split_params(s, deep_eval=True):
    """
    Use the standardized string format to split up a dictionary set
    
    >>> split_params('a-1_b-2')['a']
    1
    >>> split_params('a-1_b-2', deep_eval=False)['a']
    '1'
    >>> pd = split_params('safe_mode-False_test-1-2-3')
    >>> pd['safe_mode']
    False
    >>> pd['test']
    [1, 2, 3]
    >>> pd = split_params('safe_mode-False_test-1-2-3', deep_eval=False)
    >>> pd['safe_mode']
    'False'
    >>> pd['test']
    '1-2-3'
    
    >>> split_params('td')
    {'td': None}
    >>> split_params('got-it')
    {'got': 'it'}
    
    """
    parts = s.split('_')
    params = {}      
    if len(parts) < 2 and '-' not in parts[0]:
        return {parts[0]: None}
    for i, p in enumerate(parts):
        p2 = p.split('-')
        if len(p2) < 2:
            parts[i+1] = '_'.join([p, parts[i+1]])
        elif len(p2) == 2:
            if deep_eval:
                val = de_string(p2[1])
            else:
                val = p2[1]
            params[p2[0]] = val
        else:
            if deep_eval:
                # just make sure it wasn't a negative sign
                if p2[1] == '':
                    val = de_string('-'.join(p2[1:]))
                else:
                    val = [de_string(v) for v in p2[1:]]
            else:
                val = '-'.join(p2[1:])
            params[p2[0]] = val
    return params

############################################
## Data utilities
##
## Little functions that make rearranging data handy
##
############################################

def get_regions(labels):
    """
    Given a list-like input, return a dictionary of all the distinct
    values in it with the start and end points of each region.
    
    In the pathological case, will list every point as a "region"
    
    >>> get_regions([0, 0])
    {0: [(0, 2)]}
    >>> d = get_regions([0, 1, 1, 1, 0])
    >>> d[0]
    [(0, 1), (4, 5)]
    >>> d[1]
    [(1, 4)]
    >>> d = get_regions([0, 0, 0, 1, 1, 0, 0, 0])
    >>> d[0]
    [(0, 3), (5, 8)]
    >>> d[1]
    [(3, 5)]
    """
    labels = pd.Series(labels)
    first = labels.index[0]
    last = labels.index[-1]

    changes = list(labels.index[labels.reindex() != labels.shift().reindex()])
    if changes[0] == first:
        changes = changes[1:]
    
    regions = list(zip([first] + changes, changes+[last+1]))
    label_regions = defaultdict(list)
    #label_regions[cls].append((0, None))
    for v in regions:
        cls = labels[v[0]]
        label_regions[cls].append(v)
    #label_regions[cls+1].append((None, len(labels)))
    return dict(label_regions)

def get_outer_boundaries(regions, cls, rest_cls):
    """
    Assuming rest periods alternate with other classes, find the 
    outer boundaries of the rest regions enclosing all regions for cls
    
    >>> l = [0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 2, 0, 3, 3, 0, 4]
    >>> d = get_regions(l)
    >>> get_outer_boundaries(d, 1, 0)
    (0, 12)
    >>> get_outer_boundaries(d, 2, 0)
    (10, 14)
    >>> get_outer_boundaries(d, 4, 0)
    (16, 18)
    """
    my_start = regions[cls][0][0]
    my_end = regions[cls][-1][-1]
    
    first = my_start # the rest region preceding will end at this point
    last = my_end # the following rest region will start at this point
    for r in regions[rest_cls]:
        if r[1] == my_start:
            first = r[0]
        if r[0] == my_end:
            last = r[1]
            break
    return (first, last)

def get_counts(results, true_labels):
    """
    Display the accuracy counts grouped by class label
    v = pd.DataFrame(results)
    v['truth'] = true_labels
    print("Out of", len(v), "times when the true label was", k)
    for i in :
        correct = v[i]==v.truth
        num_cor = correct.sum()
        acc = int(100*num_cor/len(v.truth))
        print(" {} got {} correct or {}% accuracy".format(i, num_cor, acc))
    """
    raise NotImplementedError()



############################################
## Testing utilities
##
## Little functions that make running tests easier
##
############################################
def approx(num1, num2, m=4, precision=None):
    """
    Returns true if num1 and num2 are within a sliver of a floating point
    m gives the multplier on the floating point precision

    >>> np.exp(np.log(.1)) == .1
    False
    >>> approx(np.exp(np.log(.1)), .1)
    True
    >>> approx(0.02, 0.01, precision=2)
    False
    >>> approx(0.02, 0.01, precision=1)
    True
    >>> approx(-0.02, -0.02, precision=1)
    True
    """
    if precision:
        return abs(num1-num2) < 10**-precision
    return abs(num1-num2) <= m * np.spacing(1)

def all_approx(l1, l2, m=4):
    """
    Returns true if every element of l1 and l2 are approximately equivalent

    >>> l1 = [.1, .2, .3]
    >>> l2 = [np.exp(np.log(l)) for l in l1]
    >>> all_approx(l1, l2)
    True
    >>> l1[0]==l2[0]
    False
    >>> l2 = l2+[.4]
    >>> all_approx(l1, l2)
    False
    >>> all_approx(l2, l1)
    False
    >>> l1 = l1+[.4]
    >>> all_approx(l1, l2)
    True

    """
    # TODO: this can be more pythonic
    same = (len(l1) == len(l2))
    i = 0
    while same and i < len(l1):
        same = approx(l1[i], l2[i], m)
        i += 1
    return same

def dicts_exact_match(d1, d2):
    """
    Return true if all the items in each dict are identical
    """
    #TODO: handle list case
    return len(set(d1.items()) - set(d2.items())) == 0

def is_empty(value):
    """
    For when you don't know if it's a numpy array or a list but you only
    care that it has content or not
    
    >>> all([is_empty([]), is_empty(''), is_empty(None), is_empty(np.array([]))])
    True
    >>> is_empty([False])
    False
    >>> is_empty(np.array([False]))
    False
    >>> is_empty([False])
    False
    >>> is_empty(np.array([1, 2, 3]))
    False
    >>> is_empty(np.array([[]]))
    True
    """
    if value is None:
        return True
    with suppress(AttributeError):
        return False if value.size else True
    with suppress(ValueError):
        return False if value else True

"""
Maybe useful stuff from the notebook

def get_changepoints(label_data, rest=None):
    num_classes = label_data.max()[0]
    classes = ['class{}'.format(c) for c in label_data.unique()]
    if rest:
        classes.insert('rest', classes.index('class{}'.formst(rest)))
    data = cl
    groups = ground_truth.groupby('label')['timestep'].apply(np.array)
    rest = groups[0].values
    changes = [i for i, v in enumerate(rest) if rest[i-1] + 1 != v]
    first_class = {}
    for i, v in enumerate(groups):#ground_truth.groupby('label')['timestep']:
        first_class[i] = v.min()
        
def get_colours(num_colours, cmap='hsv', endpoints=(0.0, 1.0)):
    cmp = pylab.get_cmap(cmap)
    points = np.linspace(endpoints[0], endpoints[1], 
                         num=num_colours, endpoint=True)
    return [cmp(p) for p in points]

def get_regions(label_data):
    # For finding all the contiguous non-0, 0-bounded labels in a single timeseries
    llabels = label_data.T.values[0]
    changes = [i for (i, l) in enumerate(llabels) if l != llabels[i-1]] # get changepoints
    regions = list(zip(changes[::2], changes[1::2]))
    label_regions = defaultdict(list)
    cls = 0
    label_regions[cls].append((0, None))
    for v in regions:
        cls = llabels[v[0]]
        label_regions[cls].append(v)
    label_regions[cls+1].append((None, len(llabels)))
    return dict(label_regions)


"""



if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=False)
    print("Done doctests")