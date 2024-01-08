import numpy as np
from .. import constants

def calcEddingtonLuminosity(m_bh):
	"""
	Returns the Eddington luminosity in units of solar luminosities.  Mass should be in solar masses.
	"""

	return 4 * np.pi * constants.G * m_bh * constants.M_sun * constants.m_p * constants.c / constants.sigma_T / constants.L_sun
