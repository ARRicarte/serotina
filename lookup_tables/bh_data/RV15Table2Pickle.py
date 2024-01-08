"""
ARR: 06.08.16

Convert Reines and Volonteri (2015) tables into one pickle
"""

import numpy as np
import cPickle as pickle

#Input and output files
text_blagn = 'RV15_BLAGN.txt'
text_other = 'RV15_other.txt'
outfile = 'RV15.pkl'

#I've decided to save only name, M_* and M_BH
keys = ['name', 'logM_*', 'logM_BH', 'measureType']
name = []
logM_star = []
logM_BH = []
measureType = []

#Read in data one row at a time.
with open(text_blagn, 'r') as myfile:
	#Skip header
	fileWithoutHeader = myfile.readlines()[29:]
	for line in fileWithoutHeader:
		measureType.append(0)
		name.append(line[13:32].strip())
		logM_star.append(float(line[72:77]))
		logM_BH.append(float(line[79:83]))

with open(text_other, 'r') as myfile:
	#Skip header
	fileWithoutHeader = myfile.readlines()[35:]
	for line in fileWithoutHeader:
		measureType.append(int(line[0]))
		name.append(line[2:16].strip())
		logM_star.append(float(line[19:24]))
		logM_BH.append(float(line[25:30]))

outDict = dict(zip(keys, [name, logM_star, logM_BH, measureType]))
with open(outfile, 'w') as myfile:
	pickle.dump(outDict, myfile)
