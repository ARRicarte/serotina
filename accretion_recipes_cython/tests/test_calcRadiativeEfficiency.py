from .. import *
import numpy as np

def test_default():
	assert np.array(calcRadiativeEfficiency(np.random.random(), np.random.random(), spinDependence=False, f_EddCrit=None)) == 0.1

def test_upwithspin():
	e0 = np.array(calcRadiativeEfficiency(0, 1, spinDependence=True))
	eclose = np.array(calcRadiativeEfficiency(1, 1, mu=1, spinDependence=True))
	efar = np.array(calcRadiativeEfficiency(1, 1, mu=-1, spinDependence=True))
	assert eclose > e0
	assert e0 > efar

def test_fEddCrit():
	e0 = np.array(calcRadiativeEfficiency(0, 1e-3, spinDependence=False, f_EddCrit=None))
	emod = np.array(calcRadiativeEfficiency(0, 1e-3, spinDependence=False, f_EddCrit=1e-2))
	assert emod == e0 * (1e-3/1e-2)
