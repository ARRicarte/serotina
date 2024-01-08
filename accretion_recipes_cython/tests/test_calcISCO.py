from .. import *

def test_Schwarzschild():
	assert np.array(calcISCO(0)) == 6

def test_maximal_corotating():
	assert np.array(calcISCO(1, mu=1)) == 1

def test_maximal_counterrotating():
	assert np.array(calcISCO(1, mu=-1)) == 9
