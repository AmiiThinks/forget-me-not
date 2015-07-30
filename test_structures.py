import unittest
from datasets import *


class TestSimplest(Structure):
    _fields = [Parsable('base', required=True, positional=True, keyword=False),
               Parsable('myParam', required=True, positional=False, keyword=True)]

def test_simplest():
    t = TestSimplest(base='a', myParam='b')
    assert t.get_dir_string() == 'a/myParam-b'
    
def test_simpler_strings():
    t = TestSimplest(base='baseInst', myParam='paramInst')
    assert t.get_arg_string() == '--base baseInst --myParam paramInst'
    assert t.get_dir_string() == 'baseInst/myParam-paramInst'

def test_simple_assert_matches():
    t = TestSimplest(base='baseInst', myParam='paramInst')
    assert t.matches(assert_vals={'base': ['baseInst', 'other']}) == True
    assert t.matches(assert_vals={'myParam': ['baseInst', 'other']}) == False

def test_simple_negative_matches():
    t = TestSimplest(base='baseInst', myParam='paramInst')
    assert t.matches(drop_vals={'base': ['baseInst', 'other']}) == False
    assert t.matches(drop_vals={'myParam': ['baseInst', 'other']}) == True

def test_extra_matches():
    t = TestSimplest(base='baseInst', myParam='paramInst')
    assert t.matches(assert_vals={'base2': ['baseInst', 'other']}) == False
    t = TestSimplest(base='baseInst', myParam='paramInst')
    assert t.matches(assert_vals={'base': []}) == False
    assert t.matches(drop_vals={'base2': ['baseInst', 'other']}) == True
    
def test_simple_arg_parse():
    t = TestSimplest(base='a', myParam='b')
    assert t.get_arg_string() == '--base a --myParam b'
    t2 = TestSimplest.from_args(['--base', 'a', '--myParam', 'b'])
    assert t.get_dir_string() == t2.get_dir_string()
    
def test_simple_none():
    t = TestSimplest(base='a', myParam=None)
    assert t.get_dir_string() == 'a/myParam-None'
    assert t.myParam is None
    

class TestNonList(Structure):
    _fields = [Parsable('mine', keyword=False, positional=True, default='sub-1'),
               Listable('mylist', keyword=False, positional=True, default='0-1-2'),
               Keyed('mykey', keyword=False, positional=True, default='a-0_b-1')]
    
def test_nonlist():
    t = TestNonList()
    assert t.mine == 'sub-1'
    assert t.mylist == [0, 1, 2]
    assert t.mykey['a'] == 0
    assert t.mykey['b'] == 1

def test_nonlist_string():
    t = TestNonList()
    ds = t.get_dir_string()
    print(ds)
    assert ds == 'sub-1/0-1-2/a-0_b-1'
    
def test_funky_matches():
    t = TestNonList()
    assert t.matches(assert_vals={'mine': ['sub-1', 'sub-2'], 
                                  'mylist':['0-1-2', '1-2-3'],
                                  'mykey': ['a-0_b-0', 'a-0_b-1']}) == True
    assert t.matches(assert_vals={'mine': ['sub-1', 'sub-2'], 
                                  'mylist':['0-1-2', '1-2-3'],
                                  'mykey': ['a-0_b-0', 'a-1_b-1']}) == False
    assert t.matches(assert_vals={'mine': ['sub-1', 'sub-2'], 
                                  'mylist':['0-1-2', '1-2-3'],
                                  'mykey': ['a-0_b-0', 'a-0_b-1'],
                                  'other': ['a', 'b']}) == False

class TestTypes(Structure):
    _fields = [Parsable('RPK', required=True, positional=True, keyword=True),
               Parsable('RPnoK', required=True, positional=True, keyword=False),
               Parsable('RnoPK', required=True, positional=False, keyword=True),
               Parsable('RnoPnoK', required=True, positional=False, keyword=False),
               Parsable('noRPK', required=False, positional=True, keyword=True),
               Parsable('noRnoPK', required=False, positional=False, keyword=True),
               #Parsable('noRnoPnoK', required=False, positional=False, keyword=False),
               #Parsable('noRPnoK', required=False, positional=True, keyword=False),
               # can't have optional without a keyword, too hard to parse
               ]  
    
def test_all_type_config():
    t = TestTypes(RPK="overkill", 
                  RPnoK="arrogant", 
                  RnoPK="simple",
                  RnoPnoK="pushy",
                  noRPK="verbose",
                  noRnoPK="simpleopt"
                  )
    print(t.get_dir_string())
    assert t.get_dir_string() == "RPK-overkill/arrogant/noRPK-verbose/pushy_RnoPK-simple_noRnoPK-simpleopt"
    assert hasattr(t, 'noRPK')

def test_nonreq_type():
    t = TestTypes(RPK="overkill", 
                  RPnoK="arrogant", 
                  RnoPK="simple",
                  RnoPnoK="pushy"
                  )
    assert t.get_dir_string() == "RPK-overkill/arrogant/pushy_RnoPK-simple"
    assert not hasattr(t, 'noRPK')
    
    
def test_reverse_types():
    d = "RPK-overkill/arrogant/noRPK-verbose/pushy_RnoPK-simple_noRnoPK-simpleopt"
    dp = TestTypes.params_from_dir_string(d)
    print(dp)
    assert dp['RPK'] == 'overkill'
    assert dp['RPnoK'] == "arrogant"
    assert dp["RnoPK"] == "simple"
    assert dp["noRPK"] == "verbose"
    assert dp["noRnoPK"] == "simpleopt"
                  
def test_missing_opt():
    d = "RPK-overkill/arrogant/noRPK-verbose/pushy_RnoPK-simple"
    dp = TestTypes.params_from_dir_string(d)
    print(dp)
    assert dp['RPK'] == 'overkill'
    assert dp['RPnoK'] == "arrogant"
    assert dp["RnoPK"] == "simple"
    assert dp["noRPK"] == "verbose"
    assert 'noRnoPK' not in dp
    
def test_missing_pos():
    d = "RPK-overkill/arrogant/pushy_RnoPK-simple_noRnoPK-simpleopt"
    dp = TestTypes.params_from_dir_string(d)
    print(dp)
    assert dp['RPK'] == 'overkill'
    assert dp['RPnoK'] == "arrogant"
    assert dp["RnoPK"] == "simple"
    assert dp["noRnoPK"] == "simpleopt"
    assert 'noRPK' not in dp


class TestBools(Structure):
    _fields = [Boolean('safety', default=True),
               Boolean('verbose', default=False),
               Parsable('hello', default='world')]

def test_falses():
    t = TestBools.from_args(["--safety_off", "--verbose_off"])
    assert t.safety == False
    assert t.verbose == False
    print(t.get_arg_string())
    assert t.get_arg_string() == "--safety_off --verbose_off --hello world"
    
def test_trues():
    t = TestBools.from_args(["--safety", "--verbose", "--hello", "universe"])
    assert t.safety
    assert t.verbose 
    assert t.get_arg_string() == "--safety --verbose --hello universe"

class TestImpossibleParsable(Structure):
    _fields = [Parsable('opt', required=True, positional=False),
               Parsable('helpful', required=True, positional=False, default='nada')]

def test_impossible_string():    
    t = TestImpossibleParsable(opt='hello')    
    print(t.get_dir_string())
    assert t.get_dir_string() == 'helpful-nada_opt-hello'

def test_gottaGiveSomething():
    t = TestImpossibleParsable(opt='hello')
    try:
        t = TestImpossibleParsable()
    except TypeError:
        return True
    return False

   
class TestKeywordlessKeyed(Structure):
    _fields = [Keyed('myReq', required=True, keyword=False, positional=True),
               Parsable('hello', required=True, default='world')]

def test_keywordless():
    t = TestKeywordlessKeyed(myReq='got-it')
    print(t.get_dir_string())
    assert t.get_dir_string() == 'got-it/hello-world'


class TestKeywordedKeyed(Structure):
    _fields = [Keyed('myKeyReq', required=True, keyword=True),
               Parsable('hello', required=True, default='world', positional=True)]

def test_keyworded():
    t = TestKeywordedKeyed(myKeyReq='got-it')
    print(t.get_dir_string())
    assert t.get_dir_string() == 'hello-world/myKeyReq-got-it'


class TestDefaults(Structure):
    """
    If empty is used as the default, then it's only an attribute if it's been set.
    If None is used, it's assumed to be a valid value
    """
    _fields = [Parsable('must', required=True),
               Parsable('default', required=False, default=None),
               Parsable('conditional', required=False, default=empty)]

def test_defaults_set():
    t = TestDefaults(must='hello', default='world', conditional='hi')
    assert t.get_dir_string() == 'conditional-hi_default-world_must-hello'
    tas = t.get_arg_string() 
    assert tas == '--must hello --default world --conditional hi'
    t2 = TestDefaults.from_args(tas.split())
    assert t2.get_dir_string() == t2.get_dir_string()
    
def test_defaults_not_set():
    t = TestDefaults(must='hello')
    assert t.default == None
    assert t.get_dir_string() == 'default-None_must-hello'
    tas = t.get_arg_string()
    assert tas == '--must hello --default None '
    t2 = TestDefaults.from_args(tas.split())
    assert t2.get_dir_string() == t2.get_dir_string()


class TestListables(Structure):
    _fields = [Listable('myPrimeList', required=True, keyword=False),
               Listable('myOtherList', required=True, keyword=True),
               ]

def test_list_params():
    t = TestListables(myPrimeList='a-b-c', myOtherList='0-1-2')
    ds = t.get_dir_string()
    print(ds)
    print("npnk key: ", t._has_npnk)
    assert ds == 'a-b-c_myOtherList-0-1-2'
    dp = t.params_from_dir_string(ds)
    print(dp)
    assert dp['base_dir'] == ''
    assert dp['myPrimeList'] == 'a-b-c'
    assert dp['myOtherList'] == '0-1-2'
    
def test_number_list():
    t = TestListables(myPrimeList='0-1-2', myOtherList='0.99')
    assert t.myPrimeList == [0, 1, 2]
    assert t.myOtherList == [0.99]

    
class TestStructs(Structure):
    _fields = [Parsable('nom', default='hello', required=True),
               Struct('child', dtype=TestSimplest),
               Struct('problem_child', dtype=TestDefaults),
               ]
    
def test_simple_inherit():
    ts = TestStructs(nom='hi', base='a', myParam='b', must='hello')
    assert ts.child.base == 'a'
    assert ts.nom == 'hi'
    

class TestChoices(Structure):
    _fields = [Parsable('pick1', choices=['a', 'b', 'c'])]

def test_choices():
    tc = TestChoices(pick1='a')
    assert tc.pick1 == 'a'

def test_bad_choices():
    try:
        tc = TestChoices(pick1='d')
        assert False
    except TypeError:
        assert True

class TestNargs(Structure):
    _fields = [Parsable('must', nargs=1),
               Parsable('may', nargs='+'),
               Parsable('might', nargs='*')]
    
def test_simple_nargs():
    tn = TestNargs.from_args('--must hello --may be --might somewhere'.split())
    assert tn.must == ['hello']
    assert tn.may == ['be']
    assert tn.might == ['somewhere']
    
    tn = TestNargs.from_args('--must be there --may be here --might be somewhere'.split())
    assert tn.must == ['be']
    assert tn.may == ['be', 'here']
    assert tn.might == ['be', 'somewhere']

def test_nargs_direct():
    tn = TestNargs(must='hello', may='be', might='somewhere')
    assert tn.must == ['hello']
    assert tn.may == ['be']
    assert tn.might == ['somewhere']
    
    tn = TestNargs(must=['hello'], may=['be'], might=['somewhere'])
    assert tn.must == ['hello']
    assert tn.may == ['be']
    assert tn.might == ['somewhere']

    tn = TestNargs(must=['be', 'there'], may=['be', 'here'], 
                   might=['be', 'somewhere'])
    assert tn.must == ['be']
    assert tn.may == ['be', 'here']
    assert tn.might == ['be', 'somewhere']

    
def test_missing_narg():
    try:
        tn = TestNargs.from_args('--must --may --might'.split())
        assert False
    except SystemExit:
        assert True
    tn = TestNargs.from_args('--must too many --may one --might two'.split())
    assert tn.must == ['too']
    tn = TestNargs.from_args('--must just --may one --might'.split())
    assert tn.must == ['just']
    assert tn.may == ['one']
    assert tn.might == []

def test_missing_narg_keyword():
    try:
        tn = TestNargs()
        assert False
    except TypeError:
        assert True
    tn = TestNargs(must=['too', 'many'], may='one', might='two')
    assert tn.must == ['too']
    assert tn.may == ['one']
    assert tn.might == ['two']

    tn = TestNargs(must=['just'], may='one')
    assert tn.must == ['just']
    assert tn.may == ['one']
    assert not hasattr(tn, 'might') #TODO: is this difference going to be problematic?


class TestAllNargs(Structure):
    _fields = [Parsable('strict', nargs=2),
               Integer('binary', nargs='?'),
               Parsable('flexible', nargs='*'),
               Parsable('something', nargs='+')]
    
def test_all_nargs_given():
    tn = TestAllNargs(strict=['a', 'b'], binary='0', flexible=['1', '2', '3'], something=['d', 'e'])
    assert tn.strict == ['a', 'b']
    assert tn.binary == [0]
    assert tn.flexible == ['1', '2', '3']
    assert tn.something == ['d', 'e']

def test_mandatory_nargs_given():
    tn = TestAllNargs(strict=['a', 'b'], something='a')
    assert tn.strict == ['a', 'b']
    assert tn.something == ['a']
    
def test_all_nargs_args():
    tn = TestAllNargs.from_args('--strict a b --binary 0 --flexible 1 2 3 --something d e'.split())
    assert tn.strict == ['a', 'b']
    assert tn.binary == [0]
    assert tn.flexible == ['1', '2', '3']
    assert tn.something == ['d', 'e']


class TestChoiceNargs(Structure):
    _fields = [Parsable('select', choices=['a', 'b','c']),
               Parsable('letters', nargs='+'),
               Integer('numbers', choices=[0, 1, 2], nargs='+')]
    
def test_chosen_nargs():
    test = TestChoiceNargs(select='a', letters=['d', 'e', 'f'], numbers=[0, 1])
    assert test.select == 'a'
    assert test.letters == ['d', 'e', 'f']
    assert test.numbers == [0, 1]
    
def test_invalid_narg_choices():
    test = TestChoiceNargs(select='a', letters='a', numbers=0)
    assert test.numbers == [0]
    try:
        test = TestChoiceNargs(select='a', letters='a', numbers=99)
        assert False
    except TypeError:
        assert True
        

if __name__ == '__main__':
    unittest.main()