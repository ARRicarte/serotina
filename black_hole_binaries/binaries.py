"""
None of this is currently used.
"""

import pickle
import numpy as np
from .. import constants
currentPath = os.path.abspath(os.path.dirname(__file__)) + '/'

with open(currentPath + '../lookup_tables/binaryDecayTable.pkl', 'rb') as myfile:
	interpTable = pickle.load(myfile, encoding='latin1')
tau = interpTable['taus']
alpha = interpTable['alphas']
tau2alpha = InterpolatedUnivariateSpline(tau, alpha)
alpha2tau = InterpolatedUnivariateSpline(alpha, tau)

def calcHardeningRadius(m2, sigma_DM):
	"""
	Calculate the radius at which a binary becomes hard.  This is the radius used to initialize binaries
	that have coalesced due to dynamical friction.

	Assuming m2 in solar masses, sigma_DM in km/s.  Returning a_h in pc.
	"""
	return constants.G * (m2*constants.M_sun) / 4 / (sigma_DM*1e3)**2 / constants.pc

def calcDecay(a_now, Delta_t, sigma_DM, m1, m2, H=15.0, J=1.0):
	"""
	Calculate the separation that a MBHB should have after some time elapses after its formation.

	Assuming a in pc, Delta_t in Gyr, sigma_DM in km/s, m1 and m2 in solar masses.
	Returns a_new in pc
	"""
	a_h = calcHardeningRadius(m2, sigma_DM) * constants.pc
	t_0 = 9 * np.pi * constants.G**2 * J**2 * ((m1+m2)*constants.M_sun)**2 / (8 * H * (sigma_DM*1e3)**5 * a_h)

	alpha_now = a_h/(a_now*constants.pc)
	tau_now = alpha2tau(alpha_now)
	tau_new = tau_now + (Delta_t*1e9*constants.yr)/t_0
	alpha_new = tau2alpha(tau_new)
	return a_h / alpha_new / constants.pc

def calcTransitionRadius(a_now, sigma_DM, m1, m2, H=15.0, J=1.0):
	"""
	Calculate the radius at which gravitational radiation should take over.  Assuming circular orbit.  Condition is given
	by the radius at which da/dt is the same for both stellar scattering and gravitational radiation.
	
	Units are as in calcDecay()
	"""
	a_h = calcHardeningRadius(m2, sigma_DM) * constants.pc
	r_core = 3.0/4.0/(sigma_DM*1e3)**2 * constants.G * J * ((m1+m2)*constants.M_sun) * np.log(a_h/a_now)
	rho_core = (sigma_DM*1e3)**2 / (2*np.pi*constants.G*r_core**2)
	return ((64 * constants.G**2 * (sigma_DM*1e3) * (m1 * m2 * (m1+m2) * constants.M_sun**3)) / (5 * constants.c**5 * H * rho_core))**0.2 / constants.pc

