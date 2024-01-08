"""
ARR:  10.24.16

Return the pairs of duplicates in an array.
"""

import numpy as np
cimport numpy as cnp

def findDuplicates(array):

	cdef cnp.int_t[:] arr_view = np.ascontiguousarray(array)
	cdef int npts = len(arr_view)
	cdef int i
	cdef int j
	cdef cnp.int_t[:,:] pairs = np.zeros((npts,2), dtype=int)
	cdef int n_pairs = 0
	cdef cnp.int_t[:] seenBefore = np.zeros(npts, dtype=int)
	cdef cnp.int_t[:] inAPair = np.zeros(npts, dtype=int)

	for i in range(npts):
		for j in range(i):
			if seenBefore[j]:
				continue
			if arr_view[i] == arr_view[j]:
				seenBefore[i] = 1
				pairs[n_pairs,0] = j
				pairs[n_pairs,1] = i
				inAPair[i] = 1
				inAPair[j] = 1
				n_pairs += 1

	return np.array(inAPair).astype(bool), np.array(pairs)[:n_pairs,:]
