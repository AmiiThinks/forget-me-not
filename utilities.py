import numpy as np
from numpy import log, exp
from collections import namedtuple
import os
import shutil
import re
import gzip
import argparse
import itertools

lrg_fig = (12, 8)


#############################################
## Error utilities
#############################################
class UtilitiesException(Exception):
	pass

class ParameterException(Exception):
	pass

class UserException(Exception):
	pass

def approx(num1, num2, m=4, precision=None):
	"""
	Returns true if num1 and num2 are within a sliver of a floating point
	m gives the multplier on the floating point precision

	>>> np.exp(np.log(.1)) == .1
	False
	>>> approx(np.exp(np.log(.1)), .1)
	True
	"""
	if precision:
		return abs(num1-num2) <= 10**precision
	else:
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
	same = (len(l1) == len(l2))
	i = 0
	while same and i < len(l1):
		same = approx(l1[i], l2[i], m)
		i += 1
	return same
#############################################
## Vector utilities
#############################################

def scale_features(f, mn=None, mx=None, cap_values=1, verbose=False):
	if mn is None:
		mn = min(f)
	if mx is None:
		mx = max(f)
	if mx - mn == 0:
		if verbose:
			print("Feature is a constant", mx)
		return f
	f = (f - mn) / (mx - mn)
	if cap_values:
		f = np.clip(f, 0, cap_values)
	return f

def scale_val(val, rng, cap_values=None):
	val = (val - rng[0]) / (rng[1] - rng[0])
	if cap_values:
		val = np.clip(val, 0, cap_values)
	return val
	

def decay_trace(f, trace=0, replacing_traces=True):
	num_steps = len(f)
	max_value = max(f)
	decayed = np.array([0]*num_steps, dtype=float)
	decayed[0] = f[0]
	for t in range(1, num_steps):
		decayed[t] = f[t] + trace * decayed[t-1]
		if replacing_traces:
			decayed[t] = min(decayed[t], max_value)
	return decayed

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
	if gamma is None or gamma >= 1:
		r = list(r)
		ret = [sum(r[i:min(i+horizon, l)]) for i in range(l)]
		return np.array(ret)

	ret = np.array([0.0]*l)
	ret[l-1] = r[l-1]
	for i in range(l-2, -1, -1):
		ret[i] = r[i] + gamma * ret[i+1]
	return ret

def get_bin_index(signal, bins):
	"""
	Returns the index of the bin that feature belongs in.
	bin_bounds contains the upper bound on each bin's value, sorted.
	"""
	# could do this binary search-like at least
	index = 0
	for v in bin_bounds:
		if signal > v:
			return index
		index += 1
	return index - 1


def log_sum_exp(v):
	"""
	Calculate the log of the sum of exponentials of the vector elements.

	Use the shifting trick to minimize numerical precision errors.
	Argument is a list-like thing

	similar to np.logaddexp(x1, x2) but takes a 1d list.

	>>> log_sum_exp([log(0.5), log(0.5)])
	0.0
	>>> approx(exp(log_sum_exp([log(0.001), log(0.009), log(0.3)])), .31)
	True
	"""
	m = max(v)
	x = m * np.ones(np.size(v))
	return m + np.log(sum(np.exp(v - x)))



#############################################
## Math utilities
#############################################

def mscb(t):
	"""
	Find the index of the most significant change bit,
	the bit that will change when t is incremented

	>>> mscb(0)
	0
	>>> mscb(1)
	1
	>>> mscb(7)
	3
	>>> mscb(8)
	0
	"""
	return int(np.log2(t ^ (t + 1)))


def argmax(values):
	"""
	Return the index of the largest value, randomly breaking ties.

	>>> argmax([0, 1])
	1
	>>> argmax([1, 0])
	0

	>>> a = argmax([0, 1, 1])
	>>> a in (1, 2)
	True

	>>> counter = {0: 0, 1: 0}
	>>> for _ in range(20): counter[argmax([1, 1])] += 1
	>>> counter[0] > 0
	True
	>>> counter[1] > 0
	True
	"""
	values = np.array(values)
	mx = np.max(values)
	val = np.where(values==mx)[0]
	return np.random.choice(val)

def calc_alpha_init(alpha, decay):
	"""
	Calculate the numerator such that at t=0, a/(decay+t)=alpha
	"""
	if not decay or decay <= 0:
		return alpha
	else:
		return float(alpha * decay)


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
	"""
	return re.sub(r'[^a-zA-Z0-9_.]', '', str(value))

def join_items(values, sort=False):
	"""
	Converts a list of items into a string with special characters removed
	Underscores are left

	>>> join_items(['a','b','c'])
	'a-b-c'
	>>> join_items(['a[0]','a[1]'])
	'a0-a1'
	>>> join_items(['true_value',8])
	'true_value-8'
	>>> join_items(['true_values',.1,.2])
	'true_values-0.1-0.2'
	>>> join_items(['elbow_joint','wrist_joint','shoulder_joint'], True)
	'elbow_joint-shoulder_joint-wrist_joint'

	>>> join_items('fred')
	'fred'
	>>> join_items(.9)
	'0.9'
	>>> join_items(None)
	'None'
	>>> join_items('')
	''
	>>> join_items(['a_return', 'a', 'b', 'a_trace'], sort=True)
	'a-a_return-a_trace-b'
	"""
	if isinstance(values, str):
		return clean_string(values)

	try:
		val = []
		for v in values:
			val.append(clean_string(v))
		if sort:
			val.sort()
		return "-".join(val)
	except TypeError:
		return str(values)

def split_items(item_string):
	"""
	Splits a string of - separated items into its component parts.

	>>> split_items('true_values-0.1-0.2')
	['true_values', 0.1, 0.2]
	>>> split_items('a-b-c')
	['a', 'b', 'c']
	>>> split_items('true_value-8')
	['true_value', 8]
	>>> split_items('elbow_joint-shoulder_joint-wrist_joint')
	['elbow_joint', 'shoulder_joint', 'wrist_joint']
	>>> split_items('fred')
	['fred']
	>>> split_items('None')
	[None]
	>>> split_items('alpha-0.1_gamma-0.9')
	['alpha', '0.1_gamma', 0.9]
	"""
	parts = item_string.split('-')
	items = []
	# now clean up the types
	for v in parts:
		if v.isnumeric():
			items.append(int(v))
		elif v == 'None':
			items.append(None)
		else:
			try:
				items.append(float(v))
			except:
				items.append(v)
	return items


def get_best_file_match(required_features, base_dir):
	"""
	Looks for files in the given directory that have filenames that
	match required_features (using feature1_feature2.txt naming convention)
	
	If no files exist, return the standard filename
	Otherwise return the file that includes all the required features with the
	fewest extra features
	"""
	best_match = join_items(remove_modifiers(required_features), sort=True)
	required = set(split_items(best_match))
	best_match = best_match+".txt"
	fe = file_exists(os.path.join(base_dir, best_match), check_zip=True)
	if fe:
		return os.path.split(fe)[1]
	
	options = get_all_text_files(base_dir, keep_extension=True)
	if not options:
		return best_match
	extra_features = len(required_features) * 10

	for o in options:
		opt = split_ext(o)[0]
		match = match_items(required, opt)
		if match > 0 and match < extra_features:
			extra_features = match
			best_match = o
	return best_match


def split_ext(filepath):
	"""
	Take off extension, and check for another if extension is gz
	
	>>> split_ext('tmp.txt')
	('tmp', '.txt')
	>>> split_ext('tmp.txt.gz')
	('tmp', '.txt.gz')
	"""
	(fn, ext) = os.path.splitext(filepath)
	if ext=='.gz':
		(fn, ext) = os.path.splitext(fn)
		ext += '.gz'
	return (fn, ext)

def match_items(required, potential):
	if isinstance(required, str):
		required = set(split_items(required))
	if isinstance(potential, str):
		potential = set(split_items(potential))
	return match_sets(required, potential)

def match_sets(required, potential):
	"""
	Compare the features in required to potential,
	return the number of extra features in potential
	-1 means missing features
	"""
	if required.issubset(potential):
		return len(potential) - len(required)
	else:
		return -1

def shorten_keys(params):
	"""
	Creates a shorter version of the keys in params
	"""
	param_names = {}
	for n in params:
		parts = n.split('_')
		firsts = [p[0] for p in parts]
		param_names[n] = ''.join(firsts)
	return param_names
	

def join_params(**params):
	"""
	Creates a string from the key-value pairs with _ separating them,
	sorted by key

	>>> join_params(alpha=.5, gamma=.9)
	'alpha-0.5_gamma-0.9'
	>>> join_params(features=['a','b','c'],depth=15)
	'depth-15_features-a-b-c'

	>>> join_params(alpha=.1, trace_rate=None, l=['a','b'])
	'alpha-0.1_l-a-b_trace_rate-None'
	"""
	param_list = get_sorted_keys(params)
	values = []
	for k in param_list:
		values.append(k+'-'+join_items(params[k]))
	return "_".join(values)

def params_to_args(**params):
	"""
	Turns a dictionary of parameters into a command-line list of arguments
	
	>>> params = {'horizon':2, 'trace_rate': .9, 'input_features':['a', 'b'], 'manager': 'env'}
	>>> params_to_args(**params)
	['--horizon', '2', '--input_features', 'a', 'b', '--manager', 'env', '--trace_rate', '0.9']
	>>> params_to_args(alpha=.1, **params)
	['--alpha', '0.1', '--horizon', '2', '--input_features', 'a', 'b', '--manager', 'env', '--trace_rate', '0.9']
	>>> params_to_args(alpha=.1, weight_reset=True)
	['--alpha', '0.1', '--weight_reset']
	>>> params_to_args(alpha=.1, weight_reset=False)
	['--alpha', '0.1']
	"""
	args = []
	keys = get_sorted_keys(params)
	for k in keys:
		if params[k] == False:
			continue
		args.append('--'+k)
		if params[k] == True:
			continue
		
		if isinstance(params[k], str):
			args.append(params[k])
			continue
		try:
			args.extend([str(v) for v in params[k]])
		except:
			args.append(str(params[k]))
	return args

def params_to_arg_string(**params):
	"""
	Turns a dictionary of parameters into a command-line argument
	
	>>> params = {'horizon':2, 'trace_rate': .9, 'input_features':['a', 'b'], 'manager':'env'}
	>>> params_to_arg_string(**params)
	'--horizon 2 --input_features a b --manager env --trace_rate 0.9'
	"""
	args = params_to_args(**params)
	return ' '.join(args)

def get_all_dirs(dirpath, base_dir=None):
	"""
	Return an list of the directories in dirpath, starting with the directory in base_dir
	If base_dir is provided but dirpath does not contain it, return None
	
	>>> get_all_dirs('/tmp/asdf/fred')
	['tmp', 'asdf', 'fred']
	>>> get_all_dirs('/tmp/asdf/fred/') #doesn't care about final slash
	['tmp', 'asdf', 'fred']
	>>> get_all_dirs('tmp/asdf/fred') # doesn't care about starting slash
	['tmp', 'asdf', 'fred']
	>>> get_all_dirs('/tmp/asdf/fred/raw.txt') #does include file
	['tmp', 'asdf', 'fred', 'raw.txt']

	>>> get_all_dirs('./tmp/asdf//fred') # ignores extraneous details
	['tmp', 'asdf', 'fred']
	
	>>> get_all_dirs('/tmp/asdf/fred', base_dir='tmp')
	['asdf', 'fred']
	>>> get_all_dirs('/tmp/asdf/fred', base_dir='/tmp/')
	['asdf', 'fred']
	>>> get_all_dirs('tmp/asdf/fred', base_dir='/tmp/') # base_dir must match
	>>> get_all_dirs('tmp/asdf/fred/', base_dir='../tmp')
	>>> get_all_dirs('../tmp/asdf/fred/', base_dir='../tmp')
	['asdf', 'fred']
	"""
	if not base_dir:
		post = os.path.normpath(dirpath)
	elif base_dir in dirpath:
		(pre, post) = dirpath.split(os.path.normpath(base_dir))
		post = os.path.normpath(post)
	else:
		return
	dirs = []
	(head, tail) = os.path.split(post)
	while tail:
		dirs.append(tail)
		(head, tail) = os.path.split(head)
	dirs.reverse()
	return dirs

def get_dir_params(dirpath, params):
	"""
	Split off the tail directory and add the parameters in that string to param
	dictionary passed. Return the head directory
	
	>>> params = {}
	>>> get_dir_params('/tmp/alpha-1.0_alpha_decay--1', params)
	'/tmp'
	>>> len(params)
	2
	>>> params['alpha_decay']
	-1
	>>> params['alpha']
	1.0
	"""
	(head, tail) = os.path.split(dirpath)
	params.update(split_params(tail))
	return head
	
def join_number(string, num, width=None):
	"""
	Join a number to the end of a string in the standard way
	If width is provided will backfill
	
	>>> join_number('fred', 10)
	'fred-10'
	>>> join_number('fred', 10, 3)
	'fred-010'
	"""
	num = str(num)
	if width:
		num = num.rjust(width, '0')

	return string + '-' + str(num)

def split_number(string):
	"""
	Splits off a number from the end of the string and returns the tuple

	>>> split_number('raw-data.txt-500')
	('raw-data.txt', 500)
	>>> split_number('square-box-square-2.5')
	('square-box-square', 2.5)
	>>> split_number('fred')
	('fred', None)
	>>> split_number('fred-jones')
	('fred-jones', None)
	>>> split_number(0)
	('', 0)
	>>> split_number('0')
	('', 0)
	>>> print(split_number([0]))
	None
	"""
	try:
		parts = string.split('-')
	except AttributeError:
		try:
			string * string
			return ('', string)
		except TypeError:
			return None
	
		
	end = parts[-1]
	if '.' in end:
		try:
			num = float(end)
		except:
			num = None
	else:
		try:
			num = int(end)
		except:
			num = None
	if num is not None:
		parts.pop(-1)
	return ('-'.join(parts), num)

def split_params(param_string):
	"""
	Splits a parameter string into its key-value pairs


	>>> d = split_params('alpha-0.5_gamma-0.9')
	>>> d['alpha']
	0.5
	>>> d['gamma']
	0.9

	>>> d = split_params('depth-15_features-a-b-c')
	>>> d['depth']
	15
	>>> d['features']
	['a', 'b', 'c']

	>>> d = split_params('alpha-0.1_l-a-b_trace_rate-None')
	>>> d['alpha']
	0.1
	>>> d['l']
	['a', 'b']
	>>> d['trace_rate']
	>>> print(d['trace_rate'])
	None
	
	>>> split_params('a-b-c')
	{'a': ['b', 'c']}
	>>> split_params('a_b_c')
	{}
	"""
	#TODO: check for negatives i.e. alpha--1
	parts = param_string.split('_')
	params = {}

	for i in range(len(parts)):
		param = split_items(parts[i])
		if len(param) < 2:
			try:
				parts[i+1] = parts[i] + "_" + parts[i+1]
			except:
				pass
			continue
		elif len(param) == 2:
			params[param[0]] = param[1]
		elif len(param) == 3 and len(param[1]) == 0:
			params[param[0]] = -param[2]
		else:
			params[param[0]] = param[1:]
	return params

def add_modifiers(headers, modifiers, keep_unmodified=True, sort=False):
	if isinstance(headers, str):
		headers = [headers]
	if isinstance(modifiers, str):
		modifiers = [modifiers]
		
	mod = []
	for h in headers:
		if keep_unmodified:
			mod.append(h)
		for m in modifiers:
			if m not in h:
				mod.append(h+"_"+m)
	return mod

def split_modifiers(mod_string, mod_set=None):
	"""
	Removes modifiers from the given string and returns the 
	original name plus a list of the modifiers present (checked against mod_set if provided)
	
	>>> split_modifiers('joint_active_scaled_return', ['trace', 'scaled', 'return'])
	('joint_active', ['scaled', 'return'])
	>>> split_modifiers('joint_active_scaled')
	('joint', ['active', 'scaled'])
	
	>>> split_modifiers('joint_active_scaled', {'scaled':'fred','trace':'active'})
	('joint_active', ['scaled'])
	
	>>> split_modifiers('joint_scaled_trace')
	('joint', ['scaled', 'trace'])
	
	>>> split_modifiers('joint_trace_scaled')
	('joint', ['trace', 'scaled'])
	
	>>> split_modifiers('joint_scaled_active_trace', ['trace', 'scaled'])
	('joint_active', ['scaled', 'trace'])
	
	>>> split_modifiers('trace_rate_scaled', ['trace', 'scaled'])
	('trace_rate', ['scaled'])
	"""
	parts = mod_string.split('_')
	if mod_set is None:
		return (parts[0], parts[1:])
	name = [parts[0]]
	mods = []
	
	for p in parts[1:]:
		if p in mod_set:
			mods.append(p)
		else:
			name.append(p)
	
	return ('_'.join(name), mods)

def remove_modifiers(*values, sort=False, mod_set=None):
	"""
	Removes _scaled, etc, from the feature list to create a unique set
	of the features as in the environment directory


	>>> features = ['obs2_scaled_decayed','obs1_scaled','obs2_scaled','obs1_return'] 
	>>> remove_modifiers(*features, sort=False)
	['obs2', 'obs1']
	>>> remove_modifiers(*features, sort=True)
	['obs1', 'obs2']

	>>> remove_modifiers('trace_rate','trace_key_scaled', 'trace_rate_scaled')
	['trace']

	>>> remove_modifiers('trace_rate','trace_key_scaled', 'trace_rate_trace', mod_set=['scaled', 'trace'])
	['trace_rate', 'trace_key']

	"""
	features = []
	for f in values:
		(name, mods) = split_modifiers(f, mod_set=mod_set)
		if name not in features:
			features.append(name)
	if sort:
		features.sort()
	return features	

def get_indices(data_list):
	return {d: i for i, d in enumerate(data_list)}

def get_sorted_keys(data):
	keys = list(data.keys())
	keys.sort()
	return keys


#############################################
## File utilities
##
## File format - values are space-separated
## Lines at the beginning starting with '#' are ignored
## The first non-comment line is assumed to be the column headers
## The data is converted to floats and stored in a header-indexed dictionary
## of numpy arrays

def is_zip(filepath):
	"""
	Return true if filepath ends in 'gz' extension
	"""
	return os.path.splitext(filepath)[1] == '.gz'

def make_dirs(dirpath, debug=False):
	"""
	Recursively creates every directory in dirpath if it does not exists.
	Returns True/False on success/failure


	>>> newdir = '/tmp/asdf/fdsa/fred'
	>>> os.path.exists(newdir)
	False
	>>> os.path.exists('/tmp/asdf/fdsa')
	False
	>>> make_dirs(newdir)
	'/tmp/asdf/fdsa/fred'
	>>> os.path.exists(newdir)
	True
	>>> os.path.exists('/tmp/asdf/fdsa')
	True
	>>> shutil.rmtree('/tmp/asdf')
	
	>>> make_dirs('/Users/fred', debug=False)
	False
	"""
	if not os.path.exists(dirpath):
		try:
			os.mkdir(dirpath)
		except OSError as e:
			if debug:
				print(e)
			(head, tail) = os.path.split(dirpath)
			if '/' not in head or os.path.exists(head):
				return False
			else:
				if(make_dirs(head)):
					return make_dirs(dirpath)
	return dirpath

def get_array_headers(array_name, length):
	"""
	Standardize array naming!
	
	>>> get_array_headers('tile_index', 3)
	['tile_index-0', 'tile_index-1', 'tile_index-2']
	>>> get_array_headers('a', 1)
	['a-0']
	>>> get_array_headers('a', 10)[0]
	'a-00'
	>>> get_array_headers('a', 1000)[1]
	'a-0001'
	"""
	width = len(str(length))
	return [join_items([array_name, str(i).zfill(width)]) for i in range(length)]

def set_attributes(obj, include_all=True, validate_params=False, valid_params=None, **params):
	"""
	Set attributes of the obj according to arguments in params
	
	include_all will add all the arguments in params to the object
	if not will only add those that are in valid_params
	
	if validate_params, will check that the params in valid_params are not None
	
	
	# TODO: add validate_params (which will only set it if the object
	has that attribute) and valid_params (which will add/change only those
	attributes in the list
	
	>>> required_params = ['a','b','c']
	>>> ignore_list = ['c']
	>>> ns = argparse.Namespace()
	
	>>> set_attributes(ns, a=1, b=None)
	>>> ns.a
	1
	>>> print(ns.b)
	None
	
	>>> set_attributes(ns, include_all=False, a=2, b=2)
	>>> ns.a
	1
	
	>>> set_attributes(ns, include_all=True, a=3, b=3, c=3, valid_params=['a','b'])
	>>> ns.a
	3
	>>> ns.c
	3
	
	>>> set_attributes(ns, include_all=False, a=4, b=4, c=4, valid_params=['a','b'])
	>>> ns.a
	4
	>>> ns.c
	3
	
	>>> set_attributes(ns, validate_params=True, a=5, b=5, c='foo')
	>>> ns.a
	5
	>>> ns.c
	'foo'
	
	>>> set_attributes(ns, validate_params=True, a=6, b=None)
	Traceback (most recent call last):
	...
	ParameterException: Required parameter b set to None
	
	>>> set_attributes(ns, validate_params=True, a=7, b=0, c=None, valid_params=['a','b'])
	>>> ns.b
	0
	
	>>> set_attributes(ns, validate_params=True, a=8, b=8, c=None, valid_params=['a','b'])
	>>> ns.b
	8
	>>> print(ns.c)
	None

	>>> set_attributes(ns, a=9, b=9, valid_params=['a','b','c'])
	>>> ns.a
	9
	>>> print(ns.c)
	None
	
	>>> set_attributes(ns, a=10, b=10, valid_params=['a','b','c', 'd'])
	Traceback (most recent call last):
	...
	ParameterException: Required parameter d missing


	>>> set_attributes(ns, validate_params=True, valid_params=['a','b','c'])
	Traceback (most recent call last):
	...
	ParameterException: Required parameter c set to None
	
	>>> set_attributes(ns, validate_params=True, c=11, valid_params=['a','b','c'])
	>>> ns.a
	9
	>>> ns.c
	11
	"""
	# make sure all required values are here
	if valid_params:
		for k in valid_params:
			if k not in params:
				if not hasattr(obj, k):
					raise ParameterException("Required parameter {0} missing".format(k))
				else:
					params[k] = getattr(obj, k)
			
			
	for k, v in params.items():
		check_value = False
		# see if we're supposed to add this parameter 
		if not include_all:
			if not valid_params or (valid_params and k not in valid_params):
				continue
		
		# see if we are supposed to validate the value, and if it's excluded
		if validate_params and (not valid_params or k in valid_params):
			check_value = True
			
		if check_value and v is None:
			raise ParameterException("Required parameter {0} set to None".format(k))
		else:
			setattr(obj, k, v)
	return

def validate_params(params, required_params, validate_values=False):
	"""
	Make sure the iterable params contains all elements of required_params
	If validate_values is True, make sure params[k] are set.
	If required_params is a dictionary, make sure params[k] are set to the values given
	
	>>> validate_params(['a','b','c'], ['a','b'])
	True
	>>> validate_params(['a','b','c'], ['a','b','d'])
	False
	>>> validate_params({'a':0,'b':1,'c':2}, ['a','b'])
	True
	>>> validate_params({'a':0,'b':1,'c':2}, ['a','b','d'])
	False
	>>> validate_params({'a':0,'b':1,'c':2}, ['a','b'], validate_values=True)
	True
	>>> validate_params({'a':0,'b':1,'c':2}, ['a','b','d'], validate_values=True)
	False
	>>> validate_params({'a':None,'b':1,'c':2}, ['a','b','d'], validate_values=True)
	False
	
	>>> validate_params({'a':0,'b':1,'c':2}, {'a':0,'b':2}, validate_values=False)
	True
	>>> validate_params({'a':0,'b':1,'c':2}, {'a':0,'b':2}, validate_values=True)
	False
	>>> validate_params({'a':0,'b':1,'c':2}, {'a':0,'b':1}, validate_values=True)
	True
	>>> validate_params({'a':None,'b':1,'c':2}, {'a':0,'b':1}, validate_values=True)
	False
	>>> validate_params({'a':None,'b':1,'c':2}, {'a':None,'b':1}, validate_values=True)
	True
	>>> validate_params({'a':0,'b':1,'c':2}, {'a':0,'b':1, 'd':2}, validate_values=True)
	False
	>>> validate_params({'a':None,'b':1,'c':2}, {'a':[0, None],'b':1, 'c':2}, validate_values=True)
	True

	"""
	# every key (or element) in required_params must be present in the given params
	for k in required_params:
		if k not in params: 
			return False
		elif validate_values:
			try:
				# see if we got a dictionary of parameters
				p_val = params.get(k)
			except AttributeError:
				# if it's not a dictionary, it doesn't have values, obviously
				return False
			# now we need to check if the given parameter value is valid
			try:
				req_vals = required_params.get(k)

				# check if there's a list of requirements
				try:
					if p_val not in req_vals:
						return False
				except TypeError:
					# check if it matches the required value
					if p_val != req_vals:
						return False
			except AttributeError:
				# if the requirements are not specified, just make sure it's set to something
				if p_val is None:
					return False
	# and if we pass all the checks for all the required_params, it's valid
	return True
def read_strings(filepointer):
	"""
	Get the next non-commented line of strings from a file, 
	separated by whitespace
	
	>>> f = open('results/testing/pos/Large/raw-data.txt', 'r')
	>>> read_strings(f)
	['a', 'b', 'c', 'd', 'e', 'f', 'step']
	>>> for _ in range(100): t = read_strings(f)
	>>> print(t)
	None
	>>> f.closed
	False
	>>> f.close()
	>>> read_strings(f)
	"""
	line = '#'
	try:
		while line and line[0]=='#':
			line = filepointer.readline()
	except (IOError, ValueError):
		return None
	if line:
		return line.split()
	else:
		return None
	
def num_data_lines(filepath):
	"""
	Return the number of non-string lines in the file
	
	>>> num_data_lines('results/testing/pos/Large/raw-data.txt')
	16
	>>> num_data_lines('asdfasdfasdf')
	-1
	"""
	if not file_exists(filepath):
		return -1
	count = 0
	with open(filepath, 'r') as f:
		while read_floats(f):
			count += 1
	f.close()
	return count

def read_floats(filepointer):
	"""
	Get the next line of floats from a file, separated by whitespace

	>>> f = open('results/testing/pos/Large/raw-data.txt', 'r')
	>>> read_floats(f)
	[0.0, 0.2, 500.0, 0.0, 0.001, 0.0, 1.0]
	>>> for _ in range(100): line = read_floats(f)
	>>> print(line)
	None
	"""
	data = read_strings(filepointer)
	if not data:
		return None
	try:
		data = [float(x) for x in data]
		return data
	except:
		# try the next line
		return read_floats(filepointer)
	
def get_header_list(filename, sort=True):
	"""
	Returns a (default sorted) list of the keys in the specified log file,
	taken from the first non-commented line
	"""
	headers = []
	try:
		if os.path.splitext(filename)[1]=='.gz':
			f = gzip.open(filename, 'rb')
			zipped = True
		else:
			f = open(filename, 'r')
			zipped = False
	except:
		return None
	
	line = '#'
	while line and line[0] == '#':
		if zipped:
			line = f.readline().decode()
		else:
			line = f.readline()
		headers = line.split()
		if line and not headers:
			line = '#'
	if sort:
		headers.sort()
	return headers

def bin_data(data, num_bins):
	"""
	Take a numpy array and return a smaller one with the
	data binned 
	Takend from http://stackoverflow.com/questions/6163334/binning-data-in-python-with-scipy-numpy
	"""
	slices = np.linspace(0, 100, num_bins+1, True).astype(np.int)
	counts = np.diff(slices)

	mean = np.add.reduceat(data, slices[:-1]) / counts
	return mean

def get_header_indices(filepath):
	"""
	Returns a dictionary of the column indices keyed by the header in the file
	"""
	headers = get_header_list(filepath, sort=False)
	return {h: i for i, h in enumerate(headers)}

def skip_comments(filepointer):
	"""
	Moves file pointer to the next non-comment line and returns
	the comments as a list of strings.
	"""
	comments = []
	data = '#'
	try:
		pos = filepointer.tell()
	except:
		print("Could not read file.")
		return None	
	
	while data[0] == '#':
		data = filepointer.readline()
		if not data:
			raise Exception("Unexpected end of file while reading comments.")

		if data[0] == '#':
			comments.append(data)
			pos = filepointer.tell()
		else:
			filepointer.seek(pos)
	return comments

def get_formatted_data(line, indices=None):
	"""
	Indices, if provided, gives the column indices for the desired data. If
	not provided, will return all the data.
	"""
	file_data = str.strip(line).split(' ')
	if indices is None:
		data = list(range(len(file_data)))
	else:
		data = list(indices)
		
	for i, file_column in enumerate(data):
		if file_column is not None:
			datum = file_data[file_column]
		else:
			datum = ' '
		if '.' in datum:
			try:
				datum = float(datum)
			except:
				pass
		else:
			try:
				datum = int(datum)
			except:
				pass
		data[i] = datum
	return data
	

def get_log_data(filename, subset=None, keep_comments=False, dtype=float, binsize=0):
	"""
	Returns the data from the file. If more than one header is provided, returns it
	as a dictionary, keyed by the header. If only one header is provided, returns an nparray
	of the specified dtype
	"""
	if binsize != 0:
		print("not implemented yet")
	if subset is None:
		subset = get_header_list(filename)
	data = {'comments': []}
	if not file_exists(filename):
		return {}
	
	# TODO: join these together again, for simplicity to handle zip files
	# reads
	lines = []
	if 'gz' in filename:
		with gzip.open(filename, 'rb') as f:
			for l in f:
				lines.append(l.decode())
	else:
		with open(filename, 'r') as f:
			lines = f.readlines()
	
	# store the comments separately		
	lcount = 0		
	for notes in lines:
		if notes == '':
			return False
		elif notes == '\n':  #something bad happened
			print("Extraneous newline in file", filename)
			continue
		elif notes[0] == '#':
			data['comments'].append(str.strip(notes[2:]))
		headers = notes.split()
		lcount += 1
		if notes[0] != '#':
			break
		
	# in case the headers were passed with invalid characters 
	indices = get_indices(headers)
	for h in headers:
		indices[clean_string(h)]=indices[h]

	# now find out what headers of the subset we have
	valid_headers = []
	for k in subset:
		if k in indices:
			data[k] = []
			valid_headers.append(k)
		else: 
			print("No such column as", k, "in file")

	# and now store the data
	for i in range(lcount, len(lines)):
		content = lines[i].split()
		for k in valid_headers:
			data[k].append(content[indices[k]])
			
	# if we're only looking up one column, return it
	if len(valid_headers) == 1:
		return np.array(data[valid_headers[0]], dtype=dtype)
	
	if not keep_comments:
		data.pop('comments')
		
	# try converting the data
	for k in data:
		if k in valid_headers:
			try:
				data[k] = np.array(data[k], dtype=dtype)
			except:
				pass
	return data

def get_snippets(infile, outfile, start, length, headers=None):
	data = get_log_data(infile, subset=headers)
	comments = data.pop('comments')
	new_data = {k: data[k][start:start + length] for k in data}
	save_log_data(outfile, new_data, comments)
	
	
		
	


def simplify_logs(dir_path, headers):
	filename = os.path.join(dir_path, 'raw-data.txt')

	print("Looking for data in", dir_path)
	if file_exists(filename):
		if len(headers) == len(get_header_list(filename, sort=False)):
			print("Already shortened")
			return
		
		data = get_log_data(filename, headers)
		print("Loaded data")
		notes = data.pop('comments')
		save_log_data(filename, data, notes)
		print("Saving with reduced headers")
	else:
		sub_dirs = get_all_subdirs(dir_path)
		for s in sub_dirs:
			simplify_logs(os.path.join(dir_path, s), headers)


def save_log_data(filepath, data, notes=None):
	keys = list(data.keys())
	keys.sort()
	num_points = len(data[keys[0]])
	if notes is None:
		notes = []

	num_lines = 0

	with open(filepath, 'w') as f:
		# notes
		f.write('# '+filepath+'\n')
		for n in notes:
			if n[0] != '#':
				f.write('# ')
			f.write(str.strip(n)+'\n')

		# headers
		f.write(' '.join(keys)+'\n')

		# data
		for i in range(num_points):
			d = [str(data[k][i]) for k in keys]
			f.write(' '.join(d)+'\n')
	return filepath

def unzip(filepath, cleanup=False):
	"""
	Takes an existing gzipped file and unzips it in place, returning the path
	"""
	(uzfile, ext) = os.path.splitext(filepath)
	if ext != '.gz':
		return filepath
	if os.path.exists(uzfile):
		return uzfile
	
	with gzip.open(filepath, 'rb') as f_in:
		with open(uzfile, 'w') as f_out:
			for line in f_in:
				f_out.write(line.decode())
	
	if cleanup and file_exists(uzfile):
		os.remove(filepath)
	return uzfile

def zipfile(filepath, cleanup=False):
	"""
	Takes an existing text file and zips it in place, returning the new path
	"""
	zfile = filepath+".gz"
	with open(filepath, 'rb') as f_in:
		with gzip.open(zfile, 'wb') as f_out:
			f_out.writelines(f_in)	
	
	if cleanup and file_exists(zfile):
		os.remove(filepath)
	return zfile


def file_exists(filepath, make_dir=False, check_zip=False, unzip_file=True):
	if not filepath:
		return False
	if os.path.exists(filepath):
		return filepath
	elif is_zip(filepath) and os.path.exists(os.path.splitext(filepath)[0]):
		return os.path.splitext(filepath)[0]
	elif check_zip:
		zipfile = filepath+'.gz'
		if os.path.exists(zipfile):
			if unzip_file:
				return unzip(zipfile)
			else:
				return zipfile
	else:
		if make_dir:
			print("Should do this separately")
			(dirpath, file) = os.path.split(filepath)
			if '.txt' in file:
				return make_dirs(dirpath)
			else:
				return make_dirs(filepath)
		return False

def get_all_subdirs(dir_path):
	"""
	Get a list of all the subdirectories of dir_path, ignoring files
	Returns the relative path
	"""
	ls = os.listdir(dir_path)
	dirs = []
	for f in ls:
		if os.path.isdir(os.path.join(dir_path, f)):
			dirs.append(f)
	return dirs

def get_all_text_files(log_path, keep_extension=True):
	ls = os.listdir(log_path)
	files = []
	for f in ls:
		(fn, ext) = split_ext(f)
		if 'txt' not in ext:
			continue
		elif keep_extension:
			files.append(f)
		else:
			files.append(fn)
	return files

def check_param_matches(candidate, settings=None, required=None, restricted=None):
	"""
	Return true if all of the keys specified in required are present and match
	or do not match settings if specified.
	"""
	print("Deprecated?")
	if not settings:
		settings = {}
	if not required:
		required = []
	if not restricted:
		restricted = []
	
	# required keys must be present in candidate and match the value in settings,
	# if provided
	for p in required:
		if p not in candidate:
			return False
		elif p in settings and settings[p] != candidate[p]:
			return False
		#TODO: generalize this to allow lists of required/restricted
	
	# restricted keys must not match the value in settings
	for p in restricted:
		if settings[p] == candidate[p]:
			return False
	
	return True

def get_all_combinations(param_opt):
	"""
	For a dictionary of potential parameter values, return an iterator 
	of dictionaries of all possible combinations
	
	Snatched from here: http://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists
	
	>>> d = get_all_combinations({'alpha': [.1, .25], 'alpha_decay': [-1, 500, 100]})
	>>> len(list(d))
	6
	>>> d = get_all_combinations({'alpha': [.1], 'alpha_decay': [-1, 500, 100]})
	>>> len(list(d))
	3
	
	>>> get_all_combinations({})
	{}
	>>> get_all_combinations(None)
	{}
	"""
	if not param_opt:
		return {}
	return (dict(zip(param_opt.keys(), x)) for x in itertools.product(*param_opt.values()))

def get_shared_keys(param_list):
	"""
	For the given list of parameter dictionaries, return a list of the dictionary
	keys that appear in every parameter dictionary
	
	>>> get_shared_keys([{'a':0, 'b':1, 'c':2, 'd':3}, {'a':0, 'b':1, 'c':3}, {'a':0, 'b':'beta'}])
	['a', 'b']
	>>> get_shared_keys([{'a':0, 'd':3}, {'a':0, 'b':1, 'c':2, 'd':3}, {'a':0, 'b':1, 'c':2}])
	['a']
	"""
	if not param_list:
		return
	keys = set(param_list[0].keys())
	for i in range(1, len(param_list)):
		keys = keys.intersection(param_list[i].keys())
	keys = list(keys)
	keys.sort()
	return keys

def get_unique_keys(param_list):
	"""
	For the given iterable of parameter dictionaries, return a list of the 
	keys that do not appear in every parameter dictionary
	
	>>> get_unique_keys([{'a':0, 'b':3}, {'a':0, 'c': 0}, {'a':1, 'b':2}])
	['b', 'c']
	>>> get_unique_keys(({'a':0, 'b':3}, {'a':0, 'b': 0, 'c': 0}, {'a':1, 'b':2}))
	['c']
	"""
	if not param_list:
		return
	counts = {}
	max_count = len(param_list)
	for p in param_list:
		for k in p:
			counts[k] = 1 + counts.get(k, 0)
	unique = []
	# now find out which keys are not shared
	for k in counts:
		if counts[k] < max_count:
			unique.append(k)
	unique.sort()
	return unique
	
def get_shared_values(param_list):
	"""
	For the given list of parameter dictionaries, make a dictionary of the 
	keys and values that are the same for every set in the list
	
	>>> get_shared_values([{'a':0, 'b':1, 'c':2, 'd':3}, {'a':0, 'b':1, 'c':3}, {'a':0, 'b':'beta'}])
	{'a': 0}
	>>> get_shared_values([{'a':0, 'd':3}, {'a':0, 'b':1, 'c':2, 'd':3}, {'a':0, 'b':1, 'c':2}])
	{'a': 0}
	>>> get_shared_values([{'a':0, 'b':1, 'c':2, 'd':3}, {'a':0, 'b':1, 'c':3}, {'a':0, 'b':'1'}])
	{'a': 0}
	>>> s = get_shared_values([{'a':0, 'b':1, 'c':2, 'd':3}, {'a':0, 'b':1, 'c':3}, {'c': 3, 'a':0, 'b':1}])
	>>> len(s)
	2
	"""
	if not param_list:
		return
	keys = [p.keys() for p in param_list]
	shared_keys = set(keys[0]).intersection(*keys)
	shared = {k: param_list[0][k] for k in shared_keys}
	for p in param_list:
		for k in p:
			if k in shared and shared[k] != p[k]:
				shared.pop(k)
	return shared
			

def group_by_keys(param_list, keys):
	"""
	Return a dictionary of the unique sets of param values for the given keys,
	indexed by a name made up of those values
	Return a dictionary of the form {key_name: {params}} where key_name
	is constructed from the unique values of each element of keys in param_list
	
	>>> param_list = [{'a':0, 'b':1, 'c':2, 'd':3, 'name':'counter'}, {'a':0, 'b':1, 'c':3, 'name':'jumper'}, {'a':0, 'b':'beta', 'name':'greek'}]
	>>> g = group_by_keys(param_list, 'a')
	>>> for k in get_sorted_keys(g): print(k, len(g[k]))
	a-0 3
	
	>>> g = group_by_keys(param_list, ['a'])
	>>> for k in get_sorted_keys(g): print(k, len(g[k]))
	a-0 3

	>>> g = group_by_keys(param_list, ['a', 'b'])
	>>> for k in get_sorted_keys(g): print(k, len(g[k]))
	a-0_b-1 2
	a-0_b-beta 1
	
	>>> g = group_by_keys(param_list, ['c', 'b'])
	>>> for k in get_sorted_keys(g): print(k, len(g[k]))
	b-1_c-2 1
	b-1_c-3 1
	b-beta_c-None 1
	
	>>> g = group_by_keys(param_list, ['name'])
	>>> for k in get_sorted_keys(g): print(k, len(g[k]))
	name-counter 1
	name-greek 1
	name-jumper 1

	>>> g = group_by_keys(param_list, [])
	>>> for k in get_sorted_keys(g): print(k, len(g[k]))
	 3
	"""
	keys = list(keys)
	names = {}
	for p in param_list:
		
		if len(keys) > 0:
			key = join_params(**{k: p.get(k, None) for k in keys})
			#vals = {k: p.get(k, None) for k in keys}
			#name = join_params(**vals)
			#names[name]=vals
		else:
			key = ''
		if key in names:
			names[key].append(p)
		else:
			names[key]=[p]
	return names
		
def group_by_key(param_list, key):
	"""
	Return a dictionary of the form {key_value: [param_list]}
	where the list of param sets is broken up according to common key values
	
	>>> param_list = [{'a':0, 'b':1, 'c':2, 'd':3, 'name':'counter'}, {'a':0, 'b':1, 'c':3, 'name':'jumper'}, {'a':0, 'b':'beta', 'name':'greek'}]
	>>> g = group_by_key(param_list, 'a')
	>>> for k in get_sorted_keys(g): print(k, len(g[k]))
	0 3
	None 0
	>>> g = group_by_key(param_list, 'b')
	>>> for k in get_sorted_keys(g): print(k, len(g[k])) 
	1 2
	None 0
	beta 1
	>>> g = group_by_key(param_list, 'd')
	>>> for k in get_sorted_keys(g): print(k, len(g[k]))
	3 1
	None 2
	>>> g = group_by_key(param_list, 'name')
	>>> for k in get_sorted_keys(g): print(k, len(g[k]))
	None 0
	counter 1
	greek 1
	jumper 1
	"""
	group = {'None':[]}
	for p in param_list:
		if key in p:
			val = str(p[key])
			if val in group:
				group[val].append(p)
			else:
				group[val]=[p]
		else:
			group['None'].append(p)
	return group
		

def add_arguments(arg_dict, parser, namespace=None):
	"""
	For each entry in arg_dict add the argument to the parser if it is
	not already in the namespace provided.
	
	If sources is a dictionary of strings, will use the strings as the
	help message for the key
	If source is a dictionary of dictionaries, will pass the dictionary
	elements as parameters to add_argument
	"""
	for k in arg_dict:
		if namespace and hasattr(namespace, k):
			continue
		try:
			h = arg_dict[k]
			if isinstance(h, dict):
				parser.add_argument('--'+k, **h)
			else:
				parser.add_argument('--'+k, help=h)
		except:
			parser.add_argument('--'+k, help='manager parameter')


def prompt_arguments(args, namespace=None):
	"""
	Print each value in args
	If namespace is given, print the value of that variable in namespace
	Otherwise print the value in the args dictionary (args can be a list or dict)
	"""
	for k in args:
		if namespace and hasattr(namespace, k):
			val = getattr(namespace, k)
		else:
			try:
				val = args[k]
			except:
				val = ''
		print("{0:15} :  {1}".format(k, val))

	choice = input("\nEnter an argument string to change these values:\n\n")
	return choice

#############################################
## Signal-passing utilities
#############################################
PairTuple = namedtuple('PairTuple', 's y ystar')
RLTuple = namedtuple('RLTuple', 's r true_return')

def Pair(x, y, ystar=None):
	return PairTuple(s=x, y=y, ystar=ystar)
def LabelPair(x, y):
	return PairTuple(s=x, y=y, ystar=y)

def RLPair(x, r, truth=None):
	return RLTuple(s=x, r=r, true_return=truth)

#############################################
## Graph utilities
#############################################

if __name__ == "__main__":
	headers =  ['adcObs[0]',
 'adcObs[1]',
 'adcObs[2]',
 'adcObs[3]',
 'controlSignal1',
 'controlSignal2',
 'decayJointActivity[0]',
 'decayJointActivity[1]',
 'emgAvg[0]',
 'emgAvg[1]',
 'emgAvg[2]',
 'emgAvg[3]',
 'jointActiveSpeed[0]',
 'jointActiveSpeed[1]',
 'jointActiveVelocity[0]',
 'jointActiveVelocity[1]',
 'loggerSteps',
 'newpos',
 'robot/Goal0',
 'robot/Goal1',
 'robot/Goal2',
 'robot/Goal3',
 'robot/Load0',
 'robot/Load1',
 'robot/Load2',
 'robot/Load3',
 'robot/Speed0',
 'robot/Speed1',
 'robot/Speed2',
 'robot/Speed3',
 'robot/Temperature0',
 'robot/Temperature1',
 'robot/Temperature2',
 'robot/Temperature3',
 'robot/Voltage0',
 'robot/Voltage1',
 'robot/Voltage2',
 'robot/Voltage3']
	#dir_path = '/Users/anna/code/modeling/results/exarm-dc/square-box-square/'
	#simplify_logs(dir_path, headers)
	import doctest
	doctest.testmod(verbose=False)
	print("Done!")
