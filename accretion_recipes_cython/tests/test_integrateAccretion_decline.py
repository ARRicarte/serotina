from .. import *
import numpy as np
import constants

def test_analytic():
	m = 1.0
	a = 0.0
	ts = 1.0
	t = 1.0
	td = 0.0
	tf = 1.0
	t_Edd = constants.t_Sal / constants.yr / 1e9
	output = integrateAccretion_decline(m, a, ts, t, td, tf, spinTracking=False)
	np.testing.assert_almost_equal(output[0], np.exp(0.9 / 0.1 * ((2)**-1 - (3)**-1) / t_Edd))

def test_infinite_limit():
	m = 1.0
        a = 0.0
        ts = np.inf
        t = 1.0
        td = 0.0
        tf = 1.0
        t_Edd = constants.t_Sal / constants.yr / 1e9
        output = integrateAccretion_decline(m, a, ts, t, td, tf, spinTracking=False)
        np.testing.assert_almost_equal(output[0], np.exp(0.9 / 0.1 * ((2)**-1) / t_Edd))
