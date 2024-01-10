import numpy as np
from . import sam_functions as sf
from ..cosmology import cosmology_functions as cf
from .. import constants

def calcLynxLimit(redshift, fluxLimit_cgs=1e-19, f_x=0.1):
	"""
	Compute the limiting bolometric luminosity that Lynx can probe as a function of redshift.

	fluxLimit_cgs interpreted in ergs per second per centimeter squared
	f_x is the fraction of flux emitted in this band	

	output in solar luminosities
	"""

	fluxLimit_mks = fluxLimit_cgs * constants.erg / (1e-2)**2
	luminosityDistances = cf.computeLuminosityDistance(redshift) * constants.pc * 1e6

	L_bol = 4 * np.pi * luminosityDistances**2 * fluxLimit_mks / f_x
	return L_bol / constants.L_sun

def calcAxisLimit(redshift, fluxLimit_cgs=3e-18, f_x=0.1):
	"""
	Compute the limiting bolometric luminosity that Lynx can probe as a function of redshift.

	fluxLimit_cgs interpreted in ergs per second per centimeter squared
	f_x is the fraction of flux emitted in this band	

	output in solar luminosities
	"""

	fluxLimit_mks = fluxLimit_cgs * constants.erg / (1e-2)**2
	luminosityDistances = cf.computeLuminosityDistance(redshift) * constants.pc * 1e6

	L_bol = 4 * np.pi * luminosityDistances**2 * fluxLimit_mks / f_x
	return L_bol / constants.L_sun

def calcChandraLimit(redshift, fluxLimit_cgs=1e-17, f_x=0.1):
	"""
	Compute the limiting bolometric luminosity that Lynx can probe as a function of redshift.

	fluxLimit_cgs interpreted in ergs per second per centimeter squared
	f_x is the fraction of flux emitted in this band	

	output in solar luminosities
	"""

	fluxLimit_mks = fluxLimit_cgs * constants.erg / (1e-2)**2
	luminosityDistances = cf.computeLuminosityDistance(redshift) * constants.pc * 1e6

	L_bol = 4 * np.pi * luminosityDistances**2 * fluxLimit_mks / f_x
	return L_bol / constants.L_sun

