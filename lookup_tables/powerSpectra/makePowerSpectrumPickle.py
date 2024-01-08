import os
import numpy as np
import cPickle as pickle

#Input info
directory = './files/'
outname = 'powerSpectra.pkl'

#Find files
files = np.array(os.listdir(directory))
redshifts = np.array([float(thing.split()[-1].split('.txt')[0]) for thing in files])

#Things get read in a random order; let's sort by redshift
rightOrder = np.argsort(redshifts)
files = files[rightOrder]
redshifts = redshifts[rightOrder]

#Storage
karray = None
powerArrays = []

#Read files, make arrays
for f_index in range(len(files)):
	data = np.loadtxt(directory+files[f_index])
	if karray is None:
		karray = data[:,0]
	powerArrays.append(data[:,1])

powerArrays = np.array(powerArrays)

#Make output
note = 'Computed with HMFCalcwith default settings (Planck cosmology).  Units are h/Mpc for k, and Mpc^3/h^3 for P.'
dictionary = {'redshift': redshifts, 'powerSpectrum': powerArrays, 'Note': note, \
'k': karray}
with open(outname, 'w') as myfile:
	pickle.dump(dictionary, myfile, protocol=2)

