import unittest
from environments import *


def test_seed_changing():
	e1 = CoinFlip(0.2)
	e2 = CoinFlip(0.2)
	assert e1.seed != e2.seed