"""
ARR:  05.25.17

Calculates the absorption fractions as a function of Lx and z
"""

import numpy as np
from .. import constants
from scipy.integrate import quad

#For most of this, see Ueda+ 2014

def fracAbsorbedAt4375(z, psi0=0.43, a1=0.48):
	"""
	The fraction of absorbed AGN at 43.75 erg/s
	"""

	#It is assumed that this only evolves to z=2.
	return psi0 * (1+np.minimum(z,2))**a1

def fracAbsorbedOfComptonThin(Lx, z, psi_max=0.84, psi_min=0.2, beta=0.24):
	"""
	The fraction of obscured objects among only those which are Compton thin.

	Assuming that Lx is coming as solar luminosities and needs to be converted to erg/s
	"""

	psi_4375 = fracAbsorbedAt4375(z)
	return np.minimum(psi_max, np.maximum(psi_4375-beta*(np.log10(Lx*constants.L_sun*1e7)-43.75), psi_min))

def fracTypeI(Lx, z):
	"""
	Unobscured fraction, including the fact that some objects are Compton thick.
	"""

	psi = fracAbsorbedOfComptonThin(Lx, z)
	return 2.0 / ((2.0 + psi)*(1.0 + psi))

def LtoLx(L_bol):
	"""
	From Hopkins+2007.  L_bol needs to be in solar luminosities.
	"""

	return L_bol / (10.83*(L_bol/1e10)**0.28 + 6.08*(L_bol/1e10)**-0.02)

def lognormal(x, sigma):
	"""
	dp(x)/dlnx
	"""
	return 1.0 / sigma / np.sqrt(2*np.pi) * np.exp(-np.log(x)**2/2/sigma**2)

def typeIProbability(L_bol, z, numberOfDexToConvolve=0.0):
	if numberOfDexToConvolve == 0:
		return fracTypeI(LtoLx(L_bol), z)
	else:
		#Slow as hell.
		probArray = np.zeros(L_bol.shape)
		for i in range(len(L_bol)):
			probArray[i] = quad(lambda logmultiplier: fracTypeI(LtoLx(L_bol[i]*np.exp(logmultiplier)), z[i])*lognormal(np.exp(logmultiplier), \
			numberOfDexToConvolve*np.log(10)), -4*numberOfDexToConvolve*np.log(10), 4*numberOfDexToConvolve*np.log(10))[0]
		return probArray
