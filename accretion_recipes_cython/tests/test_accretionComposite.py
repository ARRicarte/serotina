from .. import *
import numpy as np

def test_quasarOnly():
	m = 1.0
	a = 0.0
	ts = 1.0
	t = 1.0
	out_comp = accretionComposite(m, a, ts, t, spinTracking=True)
	out_qua = integrateAccretion(m, a, 1.0, ts, spinTracking=True)
	for i in range(3):
		np.testing.assert_almost_equal(out_comp[i], out_qua[i])

def test_steadyOnly():
	m = 1.0
        a = 0.0
        ts = 1.0
        t = 1.0
	f_min = 1e-3
        out_comp = accretionComposite(m, a, ts, t, spinTracking=True, f_EddMin=f_min, maxQuasarMass=0)
        out_qua = integrateAccretion(m, a, f_min, ts, spinTracking=True)
        for i in range(3):
                np.testing.assert_almost_equal(out_comp[i], out_qua[i])

def test_declineOnly():
	m = 1.0
        a = 0.0
        ts = 1.0
        t = 2.0
	td = 1.0
	tf = 1.0
	out_comp = accretionComposite(m, a, ts, t, spinTracking=True, maxQuasarMass=0, t_fEdd=tf, t_decline=td)
	out_dec = integrateAccretion_decline(m, a, ts, t-ts, td, tf, spinTracking=True)
	for i in range(3):
		np.testing.assert_almost_equal(out_comp[i], out_dec[i])
