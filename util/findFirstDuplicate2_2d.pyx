import numpy as np
cimport numpy as cnp

def findFirstDuplicate2_2d(arr1, arr2):
	"""
	Find the first index of the arrays that has a value repeated so far in either array.
	"""

	cdef cnp.int_t[:,:] arr1_view = np.ascontiguousarray(arr1)
	cdef cnp.int_t[:,:] arr2_view = np.ascontiguousarray(arr2)

	try:
		assert (len(arr1[:,0]) == len(arr2[:,1])) & (len(arr1[0,:])==2) & (len(arr2[0,:])==2)
	except:
		raise ValueError, "Two arrays must be n by 2."

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
