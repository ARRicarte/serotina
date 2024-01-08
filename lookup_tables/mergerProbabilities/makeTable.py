import cPickle as pickle
import numpy as np

outname = 'mergerProbabilityTable.pkl'
massRatioEdges = np.array([0.03, 0.1, 0.3, 1.0])
logStellarMassEdges = np.array([9.0,9.5,10.0,10.5,11.0,11.5])
mergerProbability = np.array(\
[[0.0, 0.0, 0.02, 0.05, 0.0], \
[0.15, 0.19, 0.28, 0.33, 0.50], \
[0.47, 0.51, 0.37, 0.74, 0.67]])
mergerProbabilityErrors = np.array(\
[[0.0, 0.0, 0.02, 0.05, 0.0], \
[0.04, 0.06, 0.08, 0.11, 0.19], \
[0.09, 0.12, 0.10, 0.20, 0.47]])

outDict = {'massRatioEdges': massRatioEdges, 'logStellarMassEdges': logStellarMassEdges, \
'mergerProbability': mergerProbability}
with open(outname, 'w') as myfile:
	pickle.dump(outDict, myfile, protocol=2)
