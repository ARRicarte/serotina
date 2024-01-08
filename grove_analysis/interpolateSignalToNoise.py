import numpy as np
import cPickle as pickle
from scipy.interpolate import RegularGridInterpolator
import os
currentPath = os.path.abspath(os.path.dirname(__file__)) + '/'

pickleFile = currentPath + '../lookup_tables/lisa_signal_to_noise_bs2020.pkl'

with open(pickleFile, 'r') as myfile:
	dictionary = pickle.load(myfile)

m = dictionary['masses']
q = dictionary['massRatios']
z = dictionary['redshifts']
sn = dictionary['signalToNoise']

_logInterpSignalToNoise = RegularGridInterpolator((np.log10(m),np.log10(q),np.log10(z)), sn, \
bounds_error=False, fill_value=None)

def interpSignalToNoise(m,q,z):
	return _logInterpSignalToNoise((np.log10(m),np.log10(q),np.log10(z)))
