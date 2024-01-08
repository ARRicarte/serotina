"""
ARR:  11.04.16

Converts the Kelly & Shen table into a friendlier format with NumPy arrays
"""

import cPickle as pickle
import numpy as np

tableFile = "massFunctions.txt"
outFile = "KellyShen2013.pkl"

with open(tableFile, 'r') as myfile:
	table = np.loadtxt(myfile, skiprows=22)

#Determine where redshift breaks occur.
zbreaks = np.where(np.diff(table[:,0]) != 0)[0] + 1
zbreaks = list(zbreaks)
zbreaks.insert(0,0)
zbreaks.append(table.shape[0])

#This information is repeated.
log_m_bh = table[zbreaks[0]:zbreaks[1],1]
redshifts = table[zbreaks[:-1],0]

logPhi = {}
logPhiRange = {}
for z_index in range(len(zbreaks)-1):
	logPhi[redshifts[z_index]] = table[zbreaks[z_index]:zbreaks[z_index+1],3]
	logPhiRange[redshifts[z_index]] = table[zbreaks[z_index]:zbreaks[z_index+1],[2,4]]

outDict = {'log_m_bh': log_m_bh, 'redshifts': redshifts, 'logPhi': logPhi, 'logPhiRange': logPhiRange}
with open(outFile, 'w') as myfile:
	pickle.dump(outDict, myfile)
