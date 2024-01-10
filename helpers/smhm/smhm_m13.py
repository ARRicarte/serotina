"""
ARR:  06.08.17

Model from Moster, Naab, and White (2013).
"""

import numpy as np

#Characteristic mass coefficients
M10 = 11.590
M11 = 1.195

#Ratio at charactistic mass
N10 = 0.0351
N11 = -0.0247

#Low mass power law slope
beta10 = 1.376
beta11 = -0.826

#High mass power law slope
gamma10 = 0.608
gamma11 = 0.329

def M1(z):
	return M10 + M11 * z / (1.0+z)

def N(z):
	return N10 + N11 * z / (1.0+z)

def beta(z):
	return beta10 + beta11 * z / (1.0+z)

def gamma(z):
	return gamma10 + gamma11 * z / (1.0+z)

def logMstar(logMhalo, z):
	M_h = 10**logMhalo
	M1_z = 10**M1(z)
	beta_z = beta(z)
	gamma_z = gamma(z)
	N_z = N(z)

	return np.log10(2 * N_z * M_h * ((M_h/M1_z)**-beta_z + (M_h/M1_z)**gamma_z)**-1)

def Mstar(M_h, z):
	M1_z = 10**M1(z)
	beta_z = beta(z)
	gamma_z = gamma(z)
	N_z = N(z)

	return 2 * N_z * M_h * ((M_h/M1_z)**-beta_z + (M_h/M1_z)**gamma_z)**-1
