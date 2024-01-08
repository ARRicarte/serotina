import numpy as np
from .. import cosmology
from .. import constants
from scipy.interpolate import interp1d
import absorptionFractions as af

def lf_M1450(M1450, alpha=-2.04, Mstar=-25.8, beta=-2.8, Phistar=4.06e-9):
	"""
	Onoue et al., (2017) Suprime Cam Measurement
	"""

	return Phistar / (10**(0.4*(alpha+1)*(M1450-Mstar)) + 10**(0.4*(beta+1)*(M1450-Mstar)))

def dLB_dLbol(Lbol, c1=6.25, c2=9.00, k1=-0.37, k2=-0.012):
	"""
	Derivative of fitting function from Hopkins et al., (2007)
	"""
	Lscaled = Lbol/1e10

	return 1.0 - Lscaled * (c1*k1 * Lscaled**(k1-1) + c2*k2 * Lscaled**(k2-1)) / \
	(c1 * Lscaled**k1 + c2 * Lscaled**k2)

def bolometricCorrectionFunction(Lbol, c1=6.25, c2=9.00, k1=-0.37, k2=-0.012):
	Lscaled = Lbol/1e10
	return Lbol / (c1 * Lscaled**k1 + c2 * Lscaled**k2)

#Need to do this via interpolation because there's no analytic form to go the other way around.
_Lbol_interp = np.logspace(8,15,1000)
_L1450_interp = bolometricCorrectionFunction(_Lbol_interp)
_loginterp = interp1d(np.log10(_L1450_interp), np.log10(_Lbol_interp), fill_value='extrapolate')
def L1450_toLbol(L1450):
	return 10**_loginterp(np.log10(L1450))

def M1450_to_Lsun(M1450, z=6):
	"""
	Do a bunch of conversions to get from M1450 to Lsun units.
	"""

	#Getting monochromatic luminosity
	L1450 = 10**(-0.4*(M1450 - 4.74))
	
	#Bolometric correction needs to be done via interpolation.
	return L1450_toLbol(L1450)

def compute_lf_bolometric(Mlimits=(-18,-29), alpha=-2.04, Mstar=-25.8, beta=-2.8, Phistar=4.06e-9, f_46=0.26, beta_obs=0.082):
	"""
	Carry out a series of derivatives in order to compute LF in units of h^3 Mpc^-3 dex^-1
	"""

	#Convert x-axis
	M1450 = np.linspace(Mlimits[0],Mlimits[1],1000)
	Lbol = M1450_to_Lsun(M1450)

	#Account for a luminosity-dependent observable fraction
	#Note:  This was the old Hopkins parameterization.
	#observableFraction = f_46 * (Lbol / (1e46 * constants.erg / constants.L_sun))**beta_obs

	#Using Ueda N_H distributions.
	observableFraction = af.typeIProbability(Lbol, 6)

	lf_mags = lf_M1450(M1450, alpha=alpha, Mstar=Mstar, beta=beta, Phistar=Phistar)
	dM_dlogL = 2.5
	dlogLB_dlogLBol = dLB_dLbol(Lbol)
	return Lbol, lf_mags * dM_dlogL * dlogLB_dlogLBol / cosmology.h**3 / observableFraction
