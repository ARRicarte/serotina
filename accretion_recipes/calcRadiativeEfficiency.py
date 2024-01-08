import numpy as np
from .calcISCO import *

def calcRadiativeEfficiency(spinMagnitude, f_Edd, alignment, f_EddCrit=3e-2, includeSpinDependence=True, includeHotTransition=False, fiducialDefault=0.1):
	"""
	Calculate the radiative efficiency of a BH accretion disk based on its spin and Eddington ratio.

	Only a thin disk model is included now.
	"""

	f_Edd = np.atleast_1d(f_Edd)
	alignment = np.atleast_1d(alignment)
	spin = np.atleast_1d(spinMagnitude) * (-1)**(1+alignment)
	assert len(spin) == len(f_Edd)

	if includeSpinDependence:
		#Set the radiative efficiency based on the ISCO location
		isco = calcISCO(spin)
		efficiency = 1.0 - (1.0 - 2.0/3.0/isco)**0.5
	else:
		#If you don't care about spin, just return the default.
		efficiency = np.full(len(spin), fiducialDefault)

	if includeHotTransition:
		#NOTE: Good place for improvement.

		#Assuming that disk properties change at low Eddington ratios via Merloni & Heinz (2008).  This is pretty arbitrary honestly.
		efficiency *= (f_Edd/f_EddCrit)

	return efficiency

