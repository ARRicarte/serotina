from .. import *
import numpy as np
import constants

def test_notime():
	m = np.random.random()
	a = np.random.random()
	f = np.random.random()
	output = integrateAccretion(m, a, f, 0, spinTracking=True, f_EddCrit=3e-2)
	np.testing.assert_almost_equal(output[0], m)
	np.testing.assert_almost_equal(output[1], a)
	np.testing.assert_almost_equal(output[2], 0)

def test_fixed_nospinning():
	f = 1.0
	a = 0.0
	m = 1.0
	t = constants.t_Sal / constants.yr / 1e9
	output = integrateAccretion(m, a, f, t, spinTracking=False, f_EddCrit=None)
	np.testing.assert_almost_equal(output[0], np.exp(1 * 0.9 / 0.1))
	assert output[1] == a
