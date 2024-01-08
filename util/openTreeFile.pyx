"""
DEPRECATED:  Now saving binary files instead of text files.

Custom program to open text files more quickly.

This isn't even Cython, but compiling as Cython improves speed by 20%.
"""

import numpy as np

def openTreeFile(FileObj, comment='#', dtype=np.float):
	rows = []
	for line in FileObj:
		words = line.split()
		try:
			rows.append([float(word) for word in words])
		except ValueError:
			if words[0][0] == comment:
				continue
	return np.array(rows)
