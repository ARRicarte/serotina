"""
Taking the CDF of all the 'extra' delays
"""

import numpy as np
import cPickle as pickle

#Load data and just get every column for the hell of it
data = np.loadtxt('./BH_pair_times_for_angelo_epsConst.txt')

#We're using the difference between the actual merger time and that predicted by BK.
delayTime = data[:,6]
redshift = data[:,2]
Mstar = np.maximum(data[:,-4], data[:,-3])
stellarMassRatio = data[:,-3]/data[:,-4]
stellarMassRatio[stellarMassRatio>1] = 1.0/stellarMassRatio[stellarMassRatio>1]
Mvir = np.maximum(data[:,-2], data[:,-1])
virialMassRatio = data[:,-1]/data[:,-2]
virialMassRatio[virialMassRatio>1] = 1.0/virialMassRatio[virialMassRatio>1]

#Modify this filter if you want
generalFilter = redshift>2
delayTime = delayTime[generalFilter]
xbins = np.linspace(0,np.max(delayTime),1000)
cdf = np.array([np.sum(delayTime<=x).astype(float)/len(delayTime) for x in xbins])

#Make a pickle
outDict = {'delayTime': xbins, 'CDF': cdf}
with open('mergerDelayTimeDistribution.pkl', 'w') as myfile:
	pickle.dump(outDict, myfile, protocol=2)
