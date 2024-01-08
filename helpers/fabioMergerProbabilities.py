"""
ARR:  05.07.21

BH merger probability function based on Tremmel et al. (2018).  
Depends only on stellar mass, not black hole mass.
For use in Chen et al. 2021
"""

import numpy as np

def fabioMergerProbabilities(stellarMass, stellarMassRatio, overallNorm=0.8, stellarMassSlope=1.26, stellarMassRatioSlope=0.57, useStellarMassDependence=False):

	output = overallNorm * stellarMassRatio**stellarMassRatioSlope
	if useStellarMassDependence:
		output *= (np.log10(stellarMass) / 12.0)**stellarMassSlope

	#Just make sure the probability is between 0 and 1.
	output = np.minimum(1, output)
	output = np.maximum(0, output)
	return output
