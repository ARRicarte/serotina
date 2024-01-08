from .calcISCO import *

def calcRadiativeEfficiency(a, f_Edd, mu=1, f_EddCrit=None, spinDependence=True):
	"""
	Calculate the radiative efficiency of a BH based on its spin and Eddington ratio
	""" 

	if spinDependence: 
		#Set the radiative efficiency based on the ISCO location
		efficiency = 1.0 - (1.0 - 2.0/3.0/calcISCO(a, mu=mu))**0.5
	else:   
		#Assume thin disk
		efficiency = 0.1
	if (f_EddCrit is not None) & (f_Edd < f_EddCrit):
		#Assuming that disk properties change at low Eddington ratios via Merloni & Heinz (2008)
		efficiency *= (f_Edd/f_EddCrit)

	return efficiency

