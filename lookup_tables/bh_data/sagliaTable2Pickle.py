"""
ARR: 06.08.16

Convert ugly Saglia ascii table into a pickled dictionary.
"""

import cPickle as pickle
import numpy as np

#Free parameters
textfile = 'Saglia16.txt'
outfile = 'Saglia16.pkl'

#Read input
table = np.loadtxt(textfile, dtype=str)

#Set up output
keys = ['name', 'morphology', 'isSINFONI', 'bulgeType', 'isMerger', 'barType', 'bulgeMassType', 'modelingType', \
'distance', 'logM_BH', 'logsigma', 'logM_Bu', 'logrho_h', 'logr_h', 'R_e', 'unit']
units = ['', '', '', '', '', '', '', '', '\mathrm{Mpc}', 'M_\odot', '\mathrm{km} \, \mathrm{s}^{-1}', \
'M_\odot', 'M_\odot \, \mathrm{kpc}^{-3}', '\mathrm{kpc}', "\mathrm{arcsec}"]
name = []
morphology = []
isSINFONI = []
bulgeType = []
isMerger = []
barType = []
bulgeMassType = []
modelingType = []
distance = []
logM_BH = []
logsigma = []
logM_Bu = []
logrho_h = []
logr_h = []
R_e = []

#Massage rows into nice format
for row in range(len(table)):
	name.append(table[row][0].astype(str))
	morphology.append(table[row][1].astype(str))
	isSINFONI.append(table[row][2].astype(int))
	bulgeType.append(table[row][3].astype(int))
	isMerger.append(table[row][4].astype(int))
	barType.append(table[row][5].astype(float))
	bulgeMassType.append(table[row][6].astype(int))
	modelingType.append(table[row][7].astype(int))
	distance.append(table[row][[8,10]].astype((float,float)).tolist())
	logM_BH.append(table[row][[11,13]].astype((float,float)).tolist())
	logsigma.append(table[row][[14,16]].astype((float,float)).tolist())
	logM_Bu.append(table[row][[17,19]].astype((float,float)).tolist())
	logrho_h.append(table[row][[20,22]].astype((float,float)).tolist())
	logr_h.append(table[row][[23,25]].astype((float,float)).tolist())
	R_e.append(table[row][26].astype(float))

#Create pickle
outDict = dict(zip(keys, [name, morphology, isSINFONI, bulgeType, isMerger, barType, bulgeMassType, modelingType, distance, \
logM_BH, logsigma, logM_Bu, logrho_h, logr_h, R_e, units]))
with open(outfile, 'w') as myfile:
	pickle.dump(outDict, myfile)
