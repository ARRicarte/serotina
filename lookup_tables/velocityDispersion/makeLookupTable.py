"""
ARR: 06.08.17

Makes a look-up table for sigma as a function of halo mass and redshift
"""

import numpy as np
import velocityDispersion as vd
import cPickle as pickle

def makeLookupTable(outfile='velocityDispersion.pkl', logHaloMass=np.linspace(6,15,100), redshift=np.linspace(0,20,200)):

	sigmaTable = np.zeros((len(logHaloMass), len(redshift)))
	for z_index in range(len(redshift)):
		sigmaTable[:,z_index] = vd.sigma_ap(10**logHaloMass, redshift[z_index])

	outDict = {'sigma': sigmaTable, 'logHaloMass': logHaloMass, 'redshift': redshift}
	with open(outfile, 'w') as myfile:
		pickle.dump(outDict, myfile, protocol=2)

if __name__ == '__main__':
	makeLookupTable()
