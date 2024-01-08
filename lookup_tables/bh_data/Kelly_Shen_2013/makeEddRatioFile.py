"""
ARR:  01.17.16

Converts the Kelly & Shen table into a friendlier format with NumPy arrays
"""

import cPickle as pickle
import numpy as np
from scipy.integrate import simps

tableFile = 'eddRatioDistributions.txt'
outFile = 'KellyShen2013_fEdd.pkl'

with open(tableFile, 'r') as myfile:
        table = np.loadtxt(myfile, skiprows=20)

#Determine where redshift breaks occur.
zbreaks = np.where(np.diff(table[:,0]) != 0)[0] + 1
zbreaks = list(zbreaks)
zbreaks.insert(0,0)
zbreaks.append(table.shape[0])

#This information is repeated.
log_f_Edd = table[zbreaks[0]:zbreaks[1],1]
redshifts = table[zbreaks[:-1],0]

logPhi = {}
logPhiRange = {}
for z_index in range(len(zbreaks)-1):
        logPhi[redshifts[z_index]] = table[zbreaks[z_index]:zbreaks[z_index+1],3]
        logPhiRange[redshifts[z_index]] = table[zbreaks[z_index]:zbreaks[z_index+1],[2,4]]

logAverageDistribution = np.average(np.array([logPhi[z] for z in redshifts]), axis=0)
normalizer = simps(log_f_Edd, 10**logAverageDistribution)
averageDistribution = 10**logAverageDistribution / normalizer

outDict = {'log_f_Edd': log_f_Edd, 'redshifts': redshifts, 'logPhi': logPhi, 'logPhiRange': logPhiRange, \
'averageDistribution': averageDistribution}
with open(outFile, 'w') as myfile:
        pickle.dump(outDict, myfile)
