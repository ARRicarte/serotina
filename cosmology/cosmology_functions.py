import numpy as np
import cosmology
import cPickle as pickle
from scipy.interpolate import interp1d
from scipy.integrate import quad
from ..constants import constants
import os
currentPath = os.path.abspath(os.path.dirname(__file__)) + '/'

#TODO:  Some functions are repeated in sam_functions for some reason.  Reduce redundancy.

"""
Time and redshift conversions.
"""

with open(currentPath + "../lookup_tables/tofz.pkl", 'rb') as myfile:
	data = pickle.load(myfile)
tarray = data['tarray']
zarray = data['zarray']

z2t_interp = interp1d(zarray, tarray)
t2z_interp = interp1d(tarray/1e9, zarray)

"""
Tabulated luminosity distances
"""

with open(currentPath + "../lookup_tables/luminosityDistances.pkl", 'rb') as myfile:
	data = pickle.load(myfile)
z_ld = data['redshifts']
ld = data['luminosityDistances']

#Output in Mpc
computeLuminosityDistance = interp1d(z_ld, ld)

def computeComovingDistance(z):
	return computeLuminosityDistance(z) / (1.0 + z)

def computeAngularDistance(z):
	return computeLuminosityDistance(z) / (1.0 + z)**2

def E(z):
	"""
	A common function is cosmology.

	H(z) = H_0 * E(z)
	"""
	return np.sqrt(cosmology.Omega_m * (1.0+z)**3 + cosmology.Omega_l)

def z2t(z):
	"""
	Compute time as a function of redshift.  Results have been tabulated for the Planck2015 cosmology.

	Returns time in Gyr.
	"""
	return z2t_interp(z)/1e9

def t2z(t):
	"""
	Compute redshift as a function of time.
	"""
	return t2z_interp(t)

def computeH(z):
	"""
	Same traditional units as H_0:  km/s/Mpc.

	Assumes Omega_m + Omega_l = 1 to save time.
	"""

	return cosmology.H_0 * (cosmology.Omega_m * (1.0 + z)**3 + cosmology.Omega_l)**0.5

def computeLuminosityDistance_slow(z):
	"""
	Compute the luminosity distance to some redshift by numerically evaluating the integral.   You probably don't want this.

	So as not to waste time, this only works for flat universes.

	Output in Mpc.
	"""
	d_H = constants.c * cosmology.t_H * constants.yr
	d_c = d_H * quad(lambda zprime: (cosmology.Omega_m*(1+zprime)**3 + cosmology.Omega_l)**-0.5, 0, z)[0]
	return d_c * (1 + z) / (1e6 * constants.pc)

def computedVdz(z):
	"""
	The amount of observable volume in a given redshift bin in cubic Mpc
	"""
	H_inverseSeconds = computeH(z) * 1e3 / (1e6 * constants.pc)
	return 4 * np.pi * (constants.c / (1e6 * constants.pc)) * computeLuminosityDistance(z)**2 / H_inverseSeconds / (1.0 + z)**2	

def computedzdt(z):
	"""
	Returned in inverse yr
	"""
	H_inverseYears = computeH(z) * 1e3 / (1e6 * constants.pc) * constants.yr
	return H_inverseYears * (1.0 + z)
