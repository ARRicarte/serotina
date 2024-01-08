import numpy as np
cimport numpy as cnp 

def calcISCO(a, mu=1):
	"""
	Return the ISCO in geometrized units.
	"""

	cdef cnp.float64_t[:] a_view = np.ascontiguousarray(a, dtype=np.float64)
	cdef int npts = len(a_view)
	cdef cnp.int_t[:] mu_view = np.zeros(npts, dtype=np.int)
	if hasattr(mu, "__len__"):
		mu_view = np.ascontiguousarray(mu)
	else:
		mu_view = np.full(npts, mu, dtype=int)
	cdef cnp.float64_t[:] z1 = np.zeros(npts, dtype=np.float64)
	cdef cnp.float64_t[:] z2 = np.zeros(npts, dtype=np.float64)
	cdef cnp.float64_t[:] output = np.zeros(npts, dtype=np.float64)
	cdef int i

	for i in range(0, npts):
		z1[i] = 1.0 + (1.0-a_view[i]**2)**(1.0/3.0) * ((1.0+a_view[i])**(1.0/3.0) + (1.0-a_view[i])**(1.0/3.0))
		z2[i] = (3*a_view[i]**2 + z1[i]**2)**0.5
		output[i] = 3 + z2[i] - mu_view[i]*((3-z1[i])*(3+z1[i]+2*z2[i]))**0.5

	return output
