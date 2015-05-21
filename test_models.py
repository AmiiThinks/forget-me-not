import unittest
from learners import *
import model

def test_partition_bounds():
    p = PTW(KTEstimator, depth=4)
    assert p.get_child_string() == "PTW^4(0:-1)"
    p.update(0) 
    assert p.get_child_string() == "PTW^4(0:0) PTW^1(0:0)"
    assert p.log_prob == log(0.5)
    assert p.child.log_prob == log(0.5)
    assert p.child.completed_child == log(0.5)
    p.update(0)
    assert p.get_child_string() == "PTW^4(0:1) PTW^2(0:1)"
    assert p.completed_child == 0.0
    assert p.child.completed_child != 0.0
    p.update(0)
    assert p.get_child_string() == "PTW^4(0:2) PTW^2(0:2) PTW^1(2:2)"
    assert p.child.child.completed_child == log(0.5)
    p.update(0)
    assert p.get_child_string() == "PTW^4(0:3) PTW^3(0:3)"
    assert p.completed_child == 0.0
    assert p.child.completed_child != 0.0
    assert p.child.child == None
    p.update(0)
    assert p.get_child_string() == "PTW^4(0:4) PTW^3(0:4) PTW^1(4:4)"
    p.update(0)
    assert p.get_child_string() == "PTW^4(0:5) PTW^3(0:5) PTW^2(4:5)"
    p.update(0)
    assert p.get_child_string() == "PTW^4(0:6) PTW^3(0:6) PTW^2(4:6) PTW^1(6:6)"
    p.update(0)
    print(p.get_child_string())
    assert p.get_child_string() == "PTW^4(0:7)"
    p.update(0)
    print(p.get_child_string())
    assert p.get_child_string() == "PTW^4(0:8) PTW^1(8:8)"
    p.update(0)
    print(p.get_child_string())
    assert p.get_child_string() == "PTW^4(0:9) PTW^2(8:9)"
    p.update(0)
    print(p.get_child_string())
    assert p.get_child_string() == "PTW^4(0:10) PTW^2(8:10) PTW^1(10:10)"
    p.update(0)
    print(p.get_child_string())
    assert p.get_child_string() == "PTW^4(0:11) PTW^3(8:11)"
    p.update(0)
    print(p.get_child_string())
    assert p.get_child_string() == "PTW^4(0:12) PTW^3(8:12) PTW^1(12:12)"
    p.update(0)
    print(p.get_child_string())
    assert p.get_child_string() == "PTW^4(0:13) PTW^3(8:13) PTW^2(12:13)"
    p.update(0)
    print(p.get_child_string())
    assert p.get_child_string() == "PTW^4(0:14) PTW^3(8:14) PTW^2(12:14) PTW^1(14:14)"
    p.update(0)
    print(p.get_child_string())
    assert p.get_child_string() == "PTW^4(0:15)"
    try:
        p.update(0)
        assert False
    except ValueError:
        assert True

def test_deep_partitions():
    p = PTW(KTEstimator, depth=8)
    assert p.get_child_string() == "PTW^8(0:-1)"
    p.update(0) 
    assert p.get_child_string() == "PTW^8(0:0) PTW^1(0:0)"
    p.update(0)
    assert p.get_child_string() == "PTW^8(0:1) PTW^2(0:1)"
    p.update(0)
    assert p.get_child_string() == "PTW^8(0:2) PTW^2(0:2) PTW^1(2:2)"
    p.update(0)
    assert p.get_child_string() == "PTW^8(0:3) PTW^3(0:3)"
    assert p.predict(0) > p.child.predict(0)
    p.update(0)
    assert p.get_child_string() == "PTW^8(0:4) PTW^3(0:4) PTW^1(4:4)"
    p.update(0)
    assert p.get_child_string() == "PTW^8(0:5) PTW^3(0:5) PTW^2(4:5)"
    p.update(0)
    assert p.get_child_string() == "PTW^8(0:6) PTW^3(0:6) PTW^2(4:6) PTW^1(6:6)"
    p.update(0)
    assert p.get_child_string() == "PTW^8(0:7) PTW^4(0:7)"

def test_depth_differences():
    p2 = PTW(KTEstimator, depth=2)
    p4 = PTW(KTEstimator, depth=4)
    p8 = PTW(KTEstimator, depth=8)

    # greater depth means greater resistance to splitting
    for _ in range(2):
        p2.update(0); p4.update(0); p8.update(0)
    assert p2.total_loss > p4.total_loss
    assert p4.total_loss > p8.total_loss
    assert p8.total_loss > -p8.model_prob
    for _ in range(2):
        p2.update(0); p4.update(0); p8.update(0)
    assert p2.total_loss > p4.total_loss
    assert p4.total_loss > p8.total_loss
    assert p8.total_loss > -p8.model_prob

    # but our internal nodes are similar
    assert p4.log_prob == p2.depth_correction(4)
    assert p4.child.log_prob == p2.depth_correction(p4.child.depth)
    assert p4.child.total_loss == p8.child.total_loss

    for _ in range(4):
        p4.update(0); p8.update(0)
    assert p4.total_loss > p8.total_loss
    assert p8.total_loss > -p8.model_prob
    assert p8.log_prob == p4.depth_correction(8)
    
    

def test_setup():
    pt = PTW(KTEstimator, depth=2)
    assert pt.predict(0) == 0.5
    assert pt.predict(1) == 0.5

def test_first_steps():
    p = PTW(KTEstimator, depth=1)
    assert p.predict(0) == 0.5
    assert np.exp(p.update(0)) == 0.5
    assert p.num_steps == 1
    assert np.exp(p.log_prob) == 0.5
    assert np.exp(p.completed_child) == 0.5
    assert np.exp(p.model_prob) == 0.5
    assert p.child is None

    first_loss = p.log_prob
    pred = p.update(0)
    second_loss = p.log_prob
    assert second_loss - first_loss == pred

def test_interesting_points():
    p = PTW(KTEstimator, depth=12)
    assert approx(p.predict(0) + p.predict(1), 1)
    for _ in range(32):
        p.update(0)
    assert approx(p.predict(0) + p.predict(1), 1)
    assert p.predict(0) > p.predict(1)
    p.update(1)
    assert p.num_steps == 33
    assert p.child is not None
    for _ in range(30):
        p.update(1)
    assert p.predict(1) > p.model.predict(1)


def test_first_steps_depth():
    p = PTW(KTEstimator, depth=5)
    p.update(0)
    assert p.log_prob == log(0.5)
    assert p.model.predict(0) == 0.75
    assert p.predict(0) < 0.75

def test_prob_sum():
    # probabilities of a discrete alphabet should sum to one
    p = PTW(KTEstimator, depth=12)
    assert approx(p.predict(0) + p.predict(1), 1)
    p.update(0)
    assert approx(p.predict(0) + p.predict(1), 1)
    p.update(0)
    print(p.predict(0), p.predict(1), p.predict(0) + p.predict(1))
    assert approx(p.predict(0) + p.predict(1), 1, precision=4)
    

def test_model_update():
    # the total loss having seen a symbol should equal the loss for predicting
    # the signal
    pt = PTW(KTEstimator, depth=12)
    mpt = model.PTW(12, model.KT)
    for i in range(16):
        print(i)
        assert mpt.log_predict(0) == pt.log_predict(0)
        assert mpt.log_predict(0) == mpt.update(0)
        assert pt.log_predict(0) == pt.update(0)


def test_improved_model():
    # the probability of seeing a symbol should be greater once we've seen 
    # a symbol
    pt = PTW(KTEstimator, depth=12)
    for _ in range(10):
        p0 = pt.predict(0)
        pt.update(0)
        assert pt.predict(0) > p0
    p1 = pt.predict(1),
    assert approx(p1 + pt.predict(0), 1, precision=4)
    pt.update(1)
    assert approx(pt.predict(1) + pt.predict(0), 1, precision=8)
    assert pt.predict(1) > p1

def test_ptw_cost():
    # there should be a small penalty to PTW if no switches have occurred
    pt = PTW(KTEstimator, depth=5)
    mpt = model.PTW(5, Base=model.KT)
    kt = KTEstimator()
    pt.update(0); kt.update(0); mpt.update(0)
    for i in range(11):
        assert approx(pt.update(0), mpt.update(0))
        kt.update(0)
        print(i+1, pt.predict(0), kt.predict(0), mpt.predict(0))
        assert pt.log_prob == mpt.log_prob
        assert pt.model_prob == kt.log_prob
        assert mpt.predict(0) < kt.predict(0)
        assert mpt.predict(0) == pt.predict(0)
        assert pt.predict(0) < kt.predict(0)
        assert pt.predict(0) > pt.predict(1)
    assert approx(pt.predict(0) + pt.predict(1), 1)
    assert approx(kt.predict(0) + kt.predict(1), 1)


def test_all_sequences_sum():
    """
    Generate all possible k-length binary sequences. Calculate the log prob of them all.
    Make sure they sum to 1
    """

def test_compare_kt():
    mKT = model.KT()
    m_loss = 0
    aKT = KTEstimator()
    a_loss = 0
    for _ in range(16):
        m_loss += mKT.update(0)
        a_loss += aKT.update(0)
    assert approx(a_loss, m_loss, precision=15)
    assert approx(mKT.predict(0), aKT.predict(0))
    assert approx(mKT.predict(1), aKT.predict(1))
    assert approx(aKT.predict(0) + aKT.predict(1), 1)

def test_compare_ptw():
    mPTW = model.PTW(1, Base=model.KT)
    aPTW = PTW(KTEstimator, depth=1)
    for _ in range(2):
        assert mPTW.predict(0) == aPTW.predict(0)
        assert mPTW.update(0) == aPTW.update(0)
    try:
        aPTW.update(0)
        assert False
    except ValueError:
        assert True
    
    mPTW = model.PTW(2, Base=model.KT)
    aPTW = PTW(KTEstimator, depth=2)
    for _ in range(4):
        assert mPTW.predict(0) == aPTW.predict(0)
        assert mPTW.update(0) == aPTW.update(0)
    try:
        aPTW.update(0)
        assert False
    except ValueError:
        assert True

    mPTW = model.PTW(12, Base=model.KT)
    aPTW = PTW(KTEstimator, depth=12)
    for _ in range(4):
        assert approx(mPTW.predict(0), aPTW.predict(0))
        assert approx(mPTW.update(0), aPTW.update(0))
    assert approx(mPTW.log_prob, aPTW.log_prob)
    
    for _ in range(2*4):
        assert approx(mPTW.predict(0), aPTW.predict(0))
        assert approx(mPTW.update(0), aPTW.update(0))
    assert approx(mPTW.log_prob, aPTW.log_prob)

    try:
        aPTW.update(0)
        assert False
    except ValueError:
        assert True




class DebugModel():
    def __init__(self, t=None, tp1=None, left=None, right=None):
        if tp1 is None:
            tp1 = t
        self.bounds = (t, tp1)
        self.loss_bound = t
        self.left = left
        self.right = right
        try:
            self.num_steps = tp1-t+1
        except:
            self.num_steps = 0

    def update(self, data):
        if self.bounds[0] is None:
            self.bounds = (data, data)
        else:
            self.bounds = (self.bounds[0], data)
        self.num_steps += 1

    @property
    def log_prob(self):
        if self.num_steps == 0:
            return DebugModel()
        else:
            return DebugModel(*self.bounds)

    def __repr__(self):
        if self.left is None:
            return "{}:{}".format(*self.bounds)
        else:
            return "{2}:{0}_{1}:{3}".format(self.left,
                                            self.right,
                                            *self.bounds)
    def __len__(self):
        return self.num_steps

'''
defunct at the moment
class DebugPTL(PTW):
	def calculate_partition_loss(self, new_model, left_loss, new_loss):
		if new_loss:
			return DebugModel(left_loss.bounds[0], new_loss.bounds[1],
			                  left=left_loss.bounds[1], right=new_loss.bounds[0])	
		else:
			return DebugModel(*left_loss.bounds)


def test_debug_model():
	t = DebugModel()
	assert str(t) == "None:None"
	assert len(t) == 0
	t.update(0)
	assert str(t) == "0:0"
	assert len(t) == 1
	t.update(1)
	assert str(t) == "0:1"
	assert len(t) == 2

def test_partition_list():
	p = DebugPTL(DebugModel, depth=5)
	p.update(0)
	assert str(p._models) == "[0:0]"
	assert str(p._losses) == "[0:0]"
	p.update(1)
	assert str(p._models) == "[0:1]"
	assert str(p._losses) == "[0:0_1:1]"
	p.update(2)
	assert str(p._models) == "[0:2, 2:2]"
	assert str(p._losses) == "[0:0_1:1, 2:2]"
	p.update(3)
	assert str(p._models) == "[0:3]"
	assert str(p._losses) == "[0:1_2:3]"
	p.update(4)
	assert str(p._models) == "[0:4, 4:4]"
	assert str(p._losses) == "[0:1_2:3, 4:4]"
	p.update(5)
	assert str(p._models) == "[0:5, 4:5]"
	assert str(p._losses) == "[0:1_2:3, 4:4_5:5]"
	p.update(6)
	assert str(p._models) == "[0:6, 4:6, 6:6]"
	assert str(p._losses) == "[0:1_2:3, 4:4_5:5, 6:6]"
	for i in range(7, 15):
		p.update(i)
	assert str(p._models) == "[0:14, 8:14, 12:14, 14:14]"
	assert str(p._losses) == "[0:3_4:7, 8:9_10:11, 12:12_13:13, 14:14]"
	p.update(15)
	assert str(p._models) == "[0:15]"
	assert str(p._losses) == "[0:7_8:15]"

'''
'''
# very old tests

class PTWdValues(unittest.TestCase):
	global sample_seq, ktp, pr
	sample_seq = {'empty': (),
		          'single': (1,),
		          'single0': (0,),
		          'flipped': (1, 0),
		          'repeated': (1, 1),
		          'alternating': (1, 0, 1, 0),
		          'three': (1, 1, 1),
		          'four': (1, 1, 1, 1),
		          'five': (1, 1, 1, 1, 1),
		          'six': (1, 1, 1, 1, 1, 1),
		          'eight': (1, 1, 1, 1, 1, 1, 1, 1)}
	ktp = {k: kt.KTModel(v).get_prob()
		   for k, v in sample_seq.items()}
	pr = {'empty': [1.0 for _ in range(5)],
		  'single': [ktp['single'] for _ in range(5)],
		  'single0': [ktp['single'] for _ in range(5)],
		  'flipped': [PTWd.quick_calc(i,
		                              ktp['flipped'],
		                              ktp['single'],
		                              ktp['single'])
		              for i in range(5)],
		  'repeated': [PTWd.quick_calc(i,
		                               ktp['repeated'],
		                               ktp['single'],
		                               ktp['single'])
		               for i in range(5)]}
	pr['alternating'] = [ktp['alternating'],
		                 .5 * ktp['alternating'] + .5 * ktp['flipped'] ** 2,
		                 .5 * ktp['alternating'] + .5 * pr['flipped'][1] ** 2
		                 ]
	pr['four'] = [ktp['four'],
		          .5 * ktp['four'] + .5 * ktp['repeated'] ** 2,
		          .5 * ktp['four'] + .5 * pr['repeated'][1] ** 2]
	pr['three'] = [ktp['three'],
		           .5 * ktp['three'] + .5 * ktp['repeated'] * ktp['single'],
		           .5 * ktp['three'] + .5 * pr['repeated'][1] * pr['single'][1]]
	pr['five'] = [ktp['five'],
		          .5 * ktp['five'] + .5 * pr['repeated'][0] * pr['three'][0],
		          .5 * ktp['five'] + .5 * pr['four'][1] * pr['single'][1]]
	pr['six'] = [ktp['six'],
		         .5 * ktp['six'] + .5 * pr['repeated'][0] * pr['four'][0],
		         .5 * ktp['six'] + .5 * pr['four'][1] * pr['repeated'][0]]
	pr['eight'] = [ptw.ptw_recursive(i, kt.KTModel,
		                             sample_seq['eight'],
		                             (1, 0), False) for i in range(4)]


	def test_constructor(self):
		"""Constructor can take a sequence argument"""
		for desc, probs in pr.items():
			seq = sample_seq[desc]
			for depth, prob in enumerate(probs):
				lprob = log2(prob)
				s = "of {0} at depth {1} should be {2}".format(desc,
								                               depth,
								                               prob)
				if depth is not 0 and len(seq) > exp2(depth):
					with self.assertRaises(ptw.SequenceLengthError) as cm:
						m = PTWd(depth, kt.KTModel, symbols=(1, 0), sequence=seq)
					the_exception = cm.exception
					self.assertIsInstance(the_exception,
										  ptw.SequenceLengthError,
										  "Depth {0} and seq {1}".format(depth,
										                                 seq))
				else:
					m = PTWd(depth, kt.KTModel, symbols=(1, 0), sequence=seq)
					self.assertEqual(list(m.sequence), list(seq),
									 "Should create "  + desc + " sequence")
					self.assertAlmostEqual(m.get_prob(),
										   prob,
										   msg = "Probability " + s,
										   places = PRECISION)
					self.assertAlmostEqual(m.prob,
										   lprob,
										   msg = "Log probability of " + s,
										   places = PRECISION)

	def test_extend_sequence(self):
		"""Extending the general model should work as expected"""
		for desc, probs in pr.items():
			seq = sample_seq[desc]

			for depth, prob in enumerate(probs):
				if depth is not 0 and len(seq) > exp2(depth):
					with self.assertRaises(ptw.SequenceLengthError) as cm:
						m = PTWd(depth, kt.KTModel, symbols=(1, 0), sequence=seq)
					the_exception = cm.exception
					self.assertIsInstance(the_exception,
										  ptw.SequenceLengthError,
										  "Depth {0} and seq {1}".format(depth,
										                                 seq))
				elif depth is not 0:
					m = PTWd(depth, kt.KTModel)
					m.extend_sequence(seq)
					self.assertEqual(m.sequence, list(seq),
									 "Empty model should allow extension")

	def test_conditional_prob_sum(self):
		"""The conditional probability of all symbols should sum to one"""
		for desc, probs in pr.items():
			seq = sample_seq[desc]

			for depth, prob in enumerate(probs):
				if len(seq) + 1 < exp2(depth):
					m = PTWd(depth, kt.KTModel, symbols=(1, 0), sequence=list(seq))
					p1 = m.conditional_prob(1)
					p2 = m.conditional_prob(0)
					self.assertAlmostEqual(p1+p2, 1.0,
										   msg = "{0}: {1}".format(desc, seq),
										   places=PRECISION)

	def test_sum_conditionals(self):
		"""The conditional probability of all symbols should sum to one"""
		for desc, probs in pr.items():
			seq = sample_seq[desc]

			for depth, prob in enumerate(probs):
				if len(seq) + 1 < exp2(depth):
					m = PTWd(depth, kt.KTModel, symbols=(1, 0), sequence=list(seq))
					p1 = m.conditional_prob(1, log_form = True)
					p2 = m.conditional_prob(0, log_form = True)
					self.assertAlmostEqual(log_sum_exp([p1, p2]), 0.0,
										   msg = "{0}: {1}".format(desc, seq),
										   places=PRECISION)
					p1 = exp2(p1)
					p2 = exp2(p2)
					self.assertAlmostEqual(p1+p2, 1.0,
										   msg = "{0}: {1}".format(desc, seq),
										   places=PRECISION)


	def test_empty_conditional(self):
		"""The conditional probability over empty should be the same as the direct prob"""
		symbols = [0, 1]
		for s in symbols:
			for d in range(1, 5):
				m = PTWd(d, kt.KTModel)
				p1 = m.conditional_prob(s, log_form=False)
				p2 = PTWd.calculate_prob(kt.KTModel, [s], symbols, log_form=False)
				self.assertEqual(p1, p2, 
								 msg = "Depth {0}: sym {1}".format(d, s))

	def test_prod_conditional(self):
		"""The current prob * cond prob should be the same as extending seq"""
		for desc, probs in pr.items():
			seq = sample_seq[desc]
			symbols = (1, 0)
			for depth, prob in enumerate(probs):
				if len(seq) + 1 < exp2(depth):
					for s in symbols:
						m = PTWd(depth, kt.KTModel, symbols=(1, 0), sequence=list(seq))
						cp = m.conditional_prob(s, log_form = True)
						fp = cp + m.prob
						m.update(s)
						ep = m.prob
						msg = "{0}: {1}".format(desc, seq),
						self.assertAlmostEqual(fp, ep,
											   msg=msg,
											   places=PRECISION)
'''