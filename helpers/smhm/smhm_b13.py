"""
ARR:  02.27.17

Model from Behroozi, Wechsler, and Conroy (2013).
"""

import numpy as np

#NOTE:  Past z=8, only the characteristic mass is allowed to evolve.

#Characteristic mass coefficients
M10 = 11.514
M1a = -1.793
M1z = -0.251

#Ratio at charactistic mass
eps0 = -1.777
epsa = -0.006
epsz = 0.0
epsa2 = -0.119

#Low mass power law slope
alpha0 = -1.412
alphaa = 0.731

#Transition or something?
delta0 = 3.508
deltaa = 2.608
deltaz = -0.043

#High mass power law slope
gamma0 = 0.316
gammaa = 1.319
gammaz = 0.279

def nu(a):
	return np.exp(-4*a**2)

def logCharacteristicMass(z):
	a = 1.0 / (1.0 + z)
	return M10 + (M1a*(a - 1) + M1z*z) * nu(a)

def logEpsilon(z):
	z = np.minimum(z, 8)
	a = 1.0 / (1.0 + z)
	return eps0 + (epsa*(a-1) + epsz*z) * nu(a) + epsa2*(a-1)

def alpha(z):
	z = np.minimum(z, 8)
	a = 1.0 / (1.0 + z)
	return alpha0 + (alphaa * (a-1)) * nu(a)

def delta(z):
	z = np.minimum(z, 8)
	a = 1.0 / (1.0 + z)
	return delta0 + (deltaa * (a-1) + deltaz*z) * nu(a)

def gamma(z):
	z = np.minimum(z, 8)
	a = 1.0 / (1.0 + z)
	return gamma0 + (gammaa * (a-1) + gammaz*z) * nu(a)

def funct(x, z):
	return -np.log10(10**(alpha(z)*x) + 1) + delta(z) * (np.log10(1.0+np.exp(x)))**gamma(z) / (1.0 + np.exp(10**(-x)))

def logMstar(logMhalo, z):
	logeps = logEpsilon(z)
	logM1 = logCharacteristicMass(z)

	return logeps + logM1 + funct(logMhalo - logM1, z) - funct(0, z)

def Mstar(Mhalo, z):
	logeps = logEpsilon(z)
        logM1 = logCharacteristicMass(z)

	return 10**(logeps + logM1 + funct(np.log10(Mhalo) - logM1, z) - funct(0, z))
