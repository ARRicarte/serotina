import numpy as np
cimport numpy as cnp

def findFirstDuplicate2(arr1, arr2):
	"""
	Find the first index of the arrays that has a value repeated so far in either array.
	"""

	cdef cnp.int_t[:] arr1_view = np.ascontiguousarray(arr1)
	cdef cnp.int_t[:] arr2_view = np.ascontiguousarray(arr2)
	try:
		assert len(arr1_view) == len(arr2_view)
	except:
		raise ValueError, "Two arrays must have the same lengths."

	cdef int i
	cdef int j
	cdef int npts = len(arr1_view)	

	for i in range(npts):
		if arr1_view[i] == arr2_view[i]:
			return i
		else:
			for j in range(i):
				if (arr1_view[i] == arr1_view[j]) | (arr1_view[i] == arr2_view[j]) | (arr2_view[i] == arr1_view[j]) | (arr2_view[i] == arr2_view[j]):
					return i
	return -1
