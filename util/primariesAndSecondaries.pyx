import numpy as np
cimport numpy as cnp

def primariesAndSecondaries(ids, masses):
	"""
	Sort an n by 2 array of ids by the masses of each pair.
	"""

	cdef cnp.int_t[:,:] id_view = np.ascontiguousarray(ids)
	cdef cnp.float64_t[:,:] m_view = np.ascontiguousarray(masses)
	cdef int npts = len(id_view)
	cdef cnp.int_t[:] primaries = np.zeros(npts, dtype=int)
	cdef cnp.int_t[:] secondaries = np.zeros(npts, dtype=int)
	cdef int i

	for i in range(npts):
		if m_view[i,1] > m_view[i,0]:
			primaries[i] = id_view[i,1]
			secondaries[i] = id_view[i,0]
		else:
			primaries[i] = id_view[i,0]
			secondaries[i] = id_view[i,1]

	return np.array(primaries), np.array(secondaries)
