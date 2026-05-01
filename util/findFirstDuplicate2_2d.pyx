import numpy as np
cimport numpy as cnp

def findFirstDuplicate2_2d(arr1, arr2):
	"""
	Find the first index of the arrays that has a value repeated so far in either array.
	"""

	cdef cnp.int64_t[:,:] arr1_view = np.ascontiguousarray(arr1, dtype=np.int64)
	cdef cnp.int64_t[:,:] arr2_view = np.ascontiguousarray(arr2, dtype=np.int64)

	if not ((arr1.shape[0] == arr2.shape[0]) and (arr1.shape[1] == 2) and (arr2.shape[1] == 2)):
		raise ValueError("Two arrays must be n by 2.")

	cdef int i
	cdef int j
	cdef int npts = arr1.shape[0]

	for i in range(npts):
		if (arr1_view[i,0] == arr2_view[i,0]) & (arr1_view[i,1] == arr2_view[i,1]):
			return i
		else:
			for j in range(i):
				if ((arr1_view[i,0] == arr1_view[j,0]) & (arr1_view[i,0] == arr1_view[j,0])) | \
				((arr1_view[i,0] == arr2_view[j,0]) & (arr1_view[i,1] == arr2_view[j,1])) | \
				((arr2_view[i,0] == arr1_view[j,0]) & (arr2_view[i,1] == arr1_view[j,1])) | \
				((arr2_view[i,0] == arr2_view[j,0]) & (arr2_view[i,1] == arr2_view[j,1])):
					return i
	return -1
