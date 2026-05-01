import numpy as np
cimport numpy as cnp

def primariesAndSecondaries(ids, masses):
	"""
	Sort an n by 2 array of ids by the masses of each pair.
	"""

	cdef cnp.int64_t[:,:] id_view = np.ascontiguousarray(ids, dtype=np.int64)
	cdef cnp.float64_t[:,:] m_view = np.ascontiguousarray(masses, dtype=np.float64)
	cdef int npts = len(id_view)
	cdef cnp.int64_t[:] primaries = np.zeros(npts, dtype=np.int64)
	cdef cnp.int64_t[:] secondaries = np.zeros(npts, dtype=np.int64)
	cdef int i

	for i in range(npts):
		if m_view[i,1] > m_view[i,0]:
			primaries[i] = id_view[i,1]
			secondaries[i] = id_view[i,0]
		else:
			primaries[i] = id_view[i,0]
			secondaries[i] = id_view[i,1]

	return np.array(primaries), np.array(secondaries)
