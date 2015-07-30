import unittest
from datasets import *


def test_simple_parsable():
    t = Parsable('mine', keyword=True)
    assert t.stringify('hello') == 'mine-hello'
    assert Parsable('mine', keyword=False).stringify('hello') == 'hello'

def test_list_parsable():
    # probably should not automatically convert lists
    t = Parsable('mylist', keyword=True)
    assert t.stringify(['a', 'b']) == 'mylist-a-b'
    assert t.stringify('a-b') == 'mylist-a-b'
    assert Parsable('mylist', keyword=False).stringify(['a', 'b']) == 'a-b'

def test_keyed_parsable():
    # probably should not automatically convert keys either
    t = Parsable('mykeyed', keyword=True).stringify({'a': 0, 'b': 1})
    assert t == 'mykeyed-a-0_b-1'
    t = Parsable('mykeyed', keyword=False).stringify({'a': 0, 'b': 1})
    assert t == 'a-0_b-1'

def test_positional():
    pass

def test_parsable_types():
    pass

def test_parsable_setting():
    pass

def test_type_failure():
    pass

def test_booleans():
    pass

def test_defaults():
    pass

def test_nargs():
    pass

def test_constant():
    pass

def test_constructor():
    pass

def test_parser():
    # with all kinds of arguments and defaults, determine what is
    # returns from the parser
    pass

def test_struct():
    pass

if __name__ == '__main__':
    unittest.main()