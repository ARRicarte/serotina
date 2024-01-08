from calcISCO import *
cimport numpy as cnp
from libc.math cimport fmax, fmin, fabs

def calcSpinEvolutionFromAccretion(spinNow, m_ratio, mu=1, spinMax=0.998):
	"""
	Assuming that m_ratio is the amount that the black hole grew.
	"""

	#Calculate the ISCO radius, depending on whether we're prograde or retrograde
	cdef cnp.float64_t[:] a_view = np.ascontiguousarray(spinNow, dtype=np.float64)
	cdef cnp.float64_t[:] q_view = np.ascontiguousarray(m_ratio, dtype=np.float64)
	cdef int npts = len(a_view)
	cdef cnp.float64_t[:] r_ISCO = np.zeros(npts, dtype=np.float64)
	cdef cnp.float64_t[:] newSpin = np.zeros(npts, dtype=np.float64)
	cdef int i	

	r_ISCO = calcISCO(a_view, mu=mu)
	for i in range(0,npts):
		#Update spin, limited to the Thorne limit.
		if q_view[i] >= r_ISCO[i]**0.5:
			newSpin[i] = spinMax
		else:
			newSpin[i] = fmax(fmin(fabs(r_ISCO[i]**0.5 / 3 / q_view[i] * \
			(4 - (3 * r_ISCO[i] / q_view[i]**2 - 2)**0.5)), spinMax), 0)

	return newSpin
