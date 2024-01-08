import numpy as np
cimport numpy as cnp
from calcISCO import *

def calcRadiativeEfficiency(a, f_Edd, mu=1, f_EddCrit=None, spinDependence=True, spinlessDefault=0.1):
	"""
	Calculate the radiative efficiency of a BH based on its spin and Eddington ratio
	"""

	cdef cnp.float64_t[:] a_view = np.ascontiguousarray(a, dtype=np.float64)
	cdef int npts = len(a_view)
	cdef cnp.float64_t[:] f_Edd_view = np.ascontiguousarray(f_Edd, dtype=np.float64)
	cdef cnp.float64_t[:] efficiency = np.zeros(npts, dtype=np.float64)
	cdef int i
	cdef cnp.float64_t[:] isco = np.zeros(npts, dtype=np.float64)

	if spinDependence:
		#Set the radiative efficiency based on the ISCO location
		isco = calcISCO(a, mu=mu)
		for i in range(0,npts):
			efficiency[i] = 1.0 - (1.0 - 2.0/3.0/isco[i])**0.5
	else:
		#Assume thin disk
		for i in range(0,npts):
			efficiency[i] = spinlessDefault

	if f_EddCrit is not None:
		#Assuming that disk properties change at low Eddington ratios via Merloni & Heinz (2008)
		for i in range(0,npts):
			if f_Edd_view[i] < f_EddCrit:
				efficiency[i] *= (f_Edd_view[i]/f_EddCrit)

	return efficiency
