from .. import *
import numpy as np

def test_spinMax():
	assert np.array(calcSpinEvolutionFromAccretion(0, np.sqrt(6), mu=1, spinMax=1)) == 1
	assert np.array(calcSpinEvolutionFromAccretion(0, np.sqrt(6)*0.9, mu=1, spinMax=1)) < 1

def test_spinFlip():
	assert np.array(calcSpinEvolutionFromAccretion(1, 3, mu=-1, spinMax=1)) == 1
	assert np.array(calcSpinEvolutionFromAccretion(1, 3*0.9, mu=-1, spinMax=1)) < 1

def test_spinDown():
	assert np.array(calcSpinEvolutionFromAccretion(1, 1.5**0.5, mu=-1, spinMax=1)) == 0
	assert np.array(calcSpinEvolutionFromAccretion(1, 1.5**0.5*0.9, mu=-1, spinMax=1)) > 0
