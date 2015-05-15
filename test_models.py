import unittest
from learners import *
import ctw

def check_cond_sum(model, alphabet=(0, 1)):
	return approx(sum([model.predict(a) for a in alphabet]), 1)

def check_update_loss(model, sym=0):
	cur_loss = model.total_loss
	pred_loss = model.predict(sym)
	model.update(sym)
	delta = model.total_loss - cur_loss
	return approx(delta, pred_loss, precision=6)

def check_matching_predictions(*models, sym=0):
	for i, m in enumerate(models[1:]):
		if m.predict(sym) != models[i].predict(sym):
			print("{} does not match {}".format(m.predict(sym), 
			                                    models[i].predict(sym)))
			return False
		else:
			return True


def test_setup():
	pt = PTW(KTEstimator, depth=2)
	assert len(pt._models) == 0
	assert len(pt._losses) == 0
	assert pt.predict(0) == 0.5
	assert pt.predict(1) == 0.5

def test_first_steps():
	pt = PTW(KTEstimator, depth=1)
	pt.update(0)
	assert len(pt._models) == 1
	assert len(pt._losses) == 1
	assert pt._models[0].predict(0) == 0.75
	assert pt._losses[0] == -log(0.5)
	assert pt.total_loss == pt._losses[0]
	pred = pt.predict(0)
	log_pred = pt.log_predict(0)
	print("Predicting a second zero {}".format(pred))
	assert approx(pred, 0.625)
	first_loss = pt.total_loss
	pt.update(0)
	second_loss = pt.total_loss
	#assert second_loss - first_loss == log_pred
	assert len(pt._models) == 1
	assert len(pt._losses) == 1
	
def test_interesting_points():
	p = PTW(KTEstimator, depth=12)
	for _ in range(32):
		p.update(0)
	assert check_cond_sum(p)
	assert len(p._models) == 1
	assert p.predict(0) > p.predict(1)
	assert check_update_loss(p, sym=1)
	assert len(p._models) == 2
	for _ in range(30):
		p.update(1)
	assert p.predict(1) > p._models[0].predict(1)


def test_first_steps_depth():
	p = PTW(KTEstimator, depth=5)
	p.update(0)
	assert p._losses[0] == -log(0.5)
	assert p._models[0].predict(0) == 0.75
	assert p.predict(0) < 0.75 #but we don't get all the way there

def test_prob_sum():
	# probabilities of a discrete alphabet should sum to one
	pt = PTW(KTEstimator, depth=12)
	assert check_cond_sum(pt)
	pt.update(0)
	assert check_cond_sum(pt)
	pt.update(1)
	assert check_cond_sum(pt)
	
	
def test_model_update():
	# the total loss having seen a symbol should equal the loss for predicting
	# the signal
	pt = PTW(KTEstimator, depth=12)
	assert check_update_loss(pt, 0)
	assert check_update_loss(pt, 0)
	assert check_update_loss(pt, 1)
	assert check_update_loss(pt, 1)
	

def test_improved_model():
	# the probability of seeing a symbol should be greater once we've seen 
	# a symbol
	pt = PTW(KTEstimator, depth=12)
	for _ in range(10):
		p0 = pt.predict(0)
		pt.update(0)
		assert pt.predict(0) > p0
	p1 = pt.predict(1)
	assert check_cond_sum(pt)
	pt.update(1)
	assert check_cond_sum(pt)
	assert pt.predict(1) > p1

def test_ptw_cost():
	# there should be a small penalty to PTW if no switches have occurred
	pt = PTW(KTEstimator, depth=12)
	kt = KTEstimator()
	
	for i in range(12):
		pt.update(0)
		kt.update(0)
		assert pt.predict(0) < kt.predict(0)
		assert pt.predict(0) > pt.predict(1)
	assert check_cond_sum(kt)
	assert check_cond_sum(pt)
	

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
	def total_loss(self):
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


def test_compare_kt():
	mKT = ctw.KT()
	aKT = KTEstimator()
	for _ in range(16):
		mKT.update(0)
		aKT.update(0)
	assert abs(mKT.total_loss) == abs(aKT.total_loss)
	assert mKT.predict(0) == aKT.predict(0)
	assert mKT.predict(1) == aKT.predict(1)
	assert check_cond_sum(aKT)

def test_compare_ptw():
	mPTW = ctw.PTW(4, Base=ctw.KT)
	aPTW = PTW(KTEstimator, depth=4)
	assert check_matching_predictions(mPTW, aPTW)
	mPTW.update(0)
	aPTW.update(0)
	assert check_matching_predictions(mPTW, aPTW)
	mPTW.update(0)
	aPTW.update(0)
	assert mPTW.log_prob == aPTW.total_loss


'''
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