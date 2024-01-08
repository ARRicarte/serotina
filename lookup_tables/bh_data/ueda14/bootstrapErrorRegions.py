"""
ARR:  06.27.17

Bootstrap error regions for Ueda+14 bolometric luminosity functions
"""

import numpy as np
import constants
import cosmology
import cPickle as pickle

#These are the values with error bars
A = (3.26, 0.08)
logLstar = (45.48, 0.09)
gamma1 = (0.89, 0.03)
gamma2 = (2.13, 0.07)
p1 = (5.33, 0.15)
beta1 = (0.46, 0.15)
logLa1 = (46.20, 0.04)

#Below is copy/pasted from sam_functions
def lumFunctUeda(luminosities, redshift, A=3.26, logLstar=45.48, gamma1=0.89, gamma2=2.13, p1=5.33, p2=-1.5, p3=-6.2, beta1=0.46, zc1=1.86, zc2=3.0, \
        logLa1=46.20, logLa2=45.67, alpha1=0.29, alpha2=-0.1):
        """
        Return luminosity function values in h^3 Mpc^-3

        redshift must be a scalar
        """

        #Assuming luminosities are solar.  Converting to erg/s
        luminosities = luminosities * constants.L_sun * 1e7

        #Normalization at z=0
        realNorm = A * 1e-6 * (cosmology.h/0.7)**3

        #Redshift evolution has a long, annoying parametrization
        z1array = zc1 * (luminosities/10**logLa1)**alpha1
        z1array[luminosities > 10**logLa1] = zc1
        z2array = zc2 * (luminosities/10**logLa2)**alpha2
        z2array[luminosities > 10**logLa2] = zc2

        #First slope is also luminosity dependent
        realp1 = p1 + beta1*(np.log10(luminosities)-logLa2)

        evolutionFactor = (1.0+redshift)**realp1
        part2Mask = (z1array < redshift) & (z2array >= redshift)
        part3Mask = z2array < redshift
        evolutionFactor[part2Mask] = (1.0+z1array[part2Mask])**realp1[part2Mask] * ((1.0+redshift)/(1.0+z1array[part2Mask]))**p2
        evolutionFactor[part3Mask] = (1.0+z1array[part3Mask])**realp1[part3Mask] * ((1.0+z2array[part3Mask])/(1.0+z1array[part3Mask]))**p2 * ((1.0+redshift)/(1.0+z2array[part3Mask]))**p3

        Lstar = 10**logLstar
        return realNorm * ((luminosities/Lstar)**gamma1 + (luminosities/Lstar)**gamma2)**-1 * evolutionFactor

#Saved arrays
relevantLuminosities = np.logspace(8,16,1000)
relevantRedshifts = np.array([0, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0])
outfile = './ueda14.pkl'
n_bootstrap = 10000

lf_ranges = np.zeros((len(relevantRedshifts),len(relevantLuminosities),2))
for z_index in range(len(relevantRedshifts)):
	print "z = {0}".format(relevantRedshifts[z_index])
	boots = np.zeros((len(relevantLuminosities),n_bootstrap))
	for boot in range(n_bootstrap):
		A_boot = A[0] + A[1] * np.random.normal()
		logLstar_boot = logLstar[0] + logLstar[1] * np.random.normal()
		gamma1_boot = gamma1[0] + gamma1[1] * np.random.normal()
		gamma2_boot = gamma2[0] + gamma2[1] * np.random.normal()
		p1_boot = p1[0] + p1[1] * np.random.normal()
		beta1_boot = beta1[0] + beta1[1] * np.random.normal()
		logLa1_boot = logLa1[0] + logLa1[1] * np.random.normal()
		boots[:,boot] = lumFunctUeda(relevantLuminosities, relevantRedshifts[z_index], A=A_boot, logLstar=logLstar_boot, \
		gamma1=gamma1_boot, gamma2=gamma2_boot, p1=p1_boot, beta1=beta1_boot, logLa1=logLa1_boot)

	lf_ranges[z_index,:,:] = np.transpose(np.percentile(boots, (14,86), axis=1))
	
dictionary = {'luminosity': relevantLuminosities, 'redshift': relevantRedshifts, 'luminosityFunction': lf_ranges}

with open(outfile, 'w') as myfile:
	pickle.dump(dictionary, myfile, protocol=2)
