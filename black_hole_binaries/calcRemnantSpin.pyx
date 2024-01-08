import numpy as np
cimport numpy as cnp
from libc.math cimport cos, sin, sqrt

#Fitted coefficients
cdef cnp.float64_t s4 = -0.129
cdef cnp.float64_t s5 = -0.384
cdef cnp.float64_t t0 = -2.686
cdef cnp.float64_t t2 = -3.454
cdef cnp.float64_t t3 = 2.353

def calcRemnantSpin(m1, m2, a1, a2, theta1=None, theta2=None, phi1=None, phi2=None, spinMax=0.998):
	"""
	Rezzolla et al., (2008)

	Assumptions:  mass loss to GW is negligible, spin vector is sum of initial spins
	plus a vector l that is aligned with the orbit.
	"""

	cdef cnp.float64_t[:] m1_view = np.ascontiguousarray(m1)
	cdef cnp.float64_t[:] m2_view = np.ascontiguousarray(m2)
	cdef cnp.float64_t[:] a1_view = np.ascontiguousarray(a1)
	cdef cnp.float64_t[:] a2_view = np.ascontiguousarray(a2)
	cdef int npts = len(m1_view)
	cdef cnp.float64_t[:] theta1_view = np.zeros(npts)
	cdef cnp.float64_t[:] theta2_view = np.zeros(npts)
	cdef cnp.float64_t[:] phi1_view = np.zeros(npts)
	cdef cnp.float64_t[:] phi2_view = np.zeros(npts)

	cdef cnp.float64_t[:,:] a_vector_1 = np.zeros((npts,3))
	cdef cnp.float64_t[:,:] a_vector_2 = np.zeros((npts,3))
	cdef cnp.float64_t[:] cosAlpha = np.zeros(npts)
	cdef cnp.float64_t[:] cosBeta = np.zeros(npts)
	cdef cnp.float64_t[:] cosGamma  = np.zeros(npts)
	cdef cnp.float64_t[:] q = np.zeros(npts)
	cdef cnp.float64_t[:] nu = np.zeros(npts)
	cdef cnp.float64_t[:] l = np.zeros(npts)
	cdef cnp.float64_t[:] underRadical = np.zeros(npts) 
	cdef cnp.float64_t[:] a_fin = np.zeros(npts)

	cdef int i
	if theta1 is None:
		theta1_view = 2*np.pi*np.random.random(npts)
	else:
		theta1_view = np.ascontiguousarray(theta1)
	if theta2 is None:
		theta2_view = 2*np.pi*np.random.random(npts)
	else:
		theta2_view = np.ascontiguousarray(theta2)
	if phi1 is None:
		phi1_view = 2*np.pi*np.random.random(npts)
	else:
		phi1_view = np.ascontiguousarray(phi1)
	if phi2 is None:
		phi2_view = 2*np.pi*np.random.random(npts)
	else:
		phi2_view = np.ascontiguousarray(phi2)

	#Transform angles
	for i in range(npts):
		a_vector_1[i,0] = a1_view[i] * cos(theta1_view[i]) * cos(phi1_view[i])
		a_vector_1[i,1] = a1_view[i] * cos(theta1_view[i]) * sin(phi1_view[i])
		a_vector_1[i,2] = a1_view[i] * sin(theta1_view[i])
		a_vector_2[i,0] = a2_view[i] * cos(theta2_view[i]) * cos(phi2_view[i])
		a_vector_2[i,1] = a2_view[i] * cos(theta2_view[i]) * sin(phi2_view[i])
		a_vector_2[i,2] = a2_view[i] * sin(theta2_view[i])

	#alpha = angle between a1 and a2, beta = angle between a1 and l, gamma = angle between a2 and l

	for i in range(npts):
		if a1_view[i]*a2_view[i] != 0:
			cosAlpha[i] = (a_vector_1[i,0]*a_vector_2[i,0] + a_vector_1[i,1]*a_vector_2[i,1] + a_vector_1[i,2]*a_vector_2[i,2]) / a1_view[i] / a2_view[i]
		else:
			cosAlpha[i] = 0
		cosBeta[i] = cos(theta1_view[i])
		cosGamma[i] = cos(theta2_view[i])
	
	#Determine mass ratio
	for i in range(npts):
		q[i] = m2_view[i] / m1_view[i]
		try:
			assert q[i] <= 1
		except:
			raise ValueError, "m1 must always be greater than or equal to m2."
		nu[i] = q[i] / (1.0 + q[i])**2

	#Magnitude of vector of orbital angular momentum
	for i in range(npts):
		l[i] = s4 / (1+q[i]**2)**2 * (a1_view[i]**2 + a2_view[i]**2*q[i]**4 + 2*a1_view[i]*a2_view[i]*q[i]**2*cosAlpha[i]) + \
		(s5*nu[i] + t0 + 2)/(1+q[i]**2) * (a1_view[i]*cosBeta[i] + a2_view[i]*q[i]**2*cosGamma[i]) + \
		2*sqrt(3) + t2*nu[i] + t3*nu[i]**2

	#Final equation
	for i in range(npts):
		underRadical[i] = a1_view[i]**2 + a2_view[i]**2*q[i]**4 + 2*a2_view[i]*a1_view[i]*q[i]**2*cosAlpha[i] + \
		2*(a1_view[i]*cosBeta[i] + a2_view[i]*q[i]**2*cosGamma[i])*l[i]*q[i] + l[i]**2*q[i]**2
		if underRadical[i] > 0:
			a_fin[i] = 1.0 / (1+q[i])**2 * sqrt(underRadical[i])
		else:
			#Sometimes I have a negative number under the radical.  For safety, I just set spin to the average...
			#I'm guessing that the fitting function happens to break with the parameters given.
			a_fin[i] = (a1_view[i]+a2_view[i]/2)
		if a_fin[i] > spinMax:
			a_fin[i] = spinMax

	return np.array(a_fin)

