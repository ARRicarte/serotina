"""
ARR: 05.17.17

Based on calcMassFunctions, just taking one aspect and allowing multiple ensembles as argument.
"""

from __future__ import division
import numpy as np
import os
import gzip
import cPickle as pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from .. import sam_functions as sf
from .. import cosmology
from .. import constants
from scipy.interpolate import RectBivariateSpline
currentPath = os.path.abspath(os.path.dirname(__file__)) + '/'

def calcHaloMassFunctions(ensemble, redshiftSlices=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], \
	n_mass=23, logRange=(10.6,15), n_sample=20, n_bootstrap=10000, plottedPercentiles=[14,86], \
	logMassBins=np.linspace(6,15,37), z_host=0):

	"""
	Create mass functions
	"""

	#Storage for reading
	halo_masses = [np.array([]) for dummy in range(len(redshiftSlices))]
	halo_weights = [np.array([]) for dummy in range(len(redshiftSlices))]
	halo_fileIndices = [np.array([]) for dummy in range(len(redshiftSlices))]

	samplingFactor = float(n_mass) / (logRange[1] - logRange[0])
	ensemble_files = [file for file in os.listdir(ensemble) if file[-5:]=='.pklz']
	print "Reading from {0}".format(ensemble)
	
	for f_index in range(len(ensemble_files)):
		file = ensemble_files[f_index]
		print file
		hostHaloMass = 10**float(file.split('_')[-1].split('m')[1].split('n')[0])
		nHalos = int(file.split('_')[-1].split('n')[1].split('.')[0])
	
		#Unpack data
		with gzip.open(ensemble+'/'+file, 'r') as myfile:
			megaDict = pickle.load(myfile)
		uniqueRedshifts = np.unique(megaDict['redshift'])

		for z_index in range(len(redshiftSlices)):
			closestRedshift = uniqueRedshifts[np.argmin(np.abs(uniqueRedshifts - redshiftSlices[z_index]))]
			if np.abs(closestRedshift - redshiftSlices[z_index]) > 0.1:
				print "Warning:  Wanted z={0}, but we're using z={1}.".format(redshiftSlices[z_index], closestRedshift)

			#Weight by z=0 abundance
			weight = sf.calcNumberDensity(hostHaloMass, z_host) / nHalos / samplingFactor
			redshiftMask = (megaDict['redshift'] == closestRedshift)

			halo_weights[z_index] = np.concatenate((halo_weights[z_index], np.full(np.sum(redshiftMask), weight)))
			halo_masses[z_index] = np.concatenate((halo_masses[z_index], megaDict['m_halo'][redshiftMask]))
			halo_fileIndices[z_index] = np.concatenate((halo_fileIndices[z_index], megaDict['treeIndex'][redshiftMask]))

	#Storage for bootstrapping
	massFunctionPieces = np.zeros((len(logMassBins)-1,len(redshiftSlices),n_sample))
	massFunctions = np.zeros((len(logMassBins)-1,len(redshiftSlices),n_bootstrap))
	massFunctionRange = np.zeros((len(logMassBins)-1,len(redshiftSlices),2))

	#Compute the mass function from this 2D histogram
	for z_index in range(len(redshiftSlices)):
		print "z = {0}:".format(redshiftSlices[z_index])
		closestRedshift = uniqueRedshifts[np.argmin(np.abs(uniqueRedshifts - redshiftSlices[z_index]))]

		#Collect the pieces for bootstrapping
		print "   Creating bootstrap pieces."
		for treeIndex in range(n_sample):
			massFunctionPieces[:,z_index,treeIndex] = np.histogram(np.log10(halo_masses[z_index][halo_fileIndices[z_index]==treeIndex+1]), bins=logMassBins, \
			weights=halo_weights[z_index][halo_fileIndices[z_index]==treeIndex+1]/np.diff(logMassBins)[0])[0]

		#Bootstrapping!
		print "   Computing mass function by bootstrapping.  Number of samples = {0}.".format(n_bootstrap)
		for boot in range(n_bootstrap):
			randomFileNumbers = np.random.randint(0, n_sample, n_sample)
			massFunctions[:,z_index,boot] = np.sum(massFunctionPieces[:,z_index,randomFileNumbers], axis=1)

		#Finally, find the central 68% of all values.
		massFunctionRange[:,z_index,:] = np.transpose(np.percentile(massFunctions[:,z_index,:], plottedPercentiles, axis=1))

	return logMassBins, massFunctionRange

def plotMassFunctions(massFuncts, labels=None, colors=None, redshiftSlices=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], figsize=(8,8), \
	figShape=None, xlim=(5e6,1e15), ylim=(1e-6,1e3), output=None):
	"""
	Plot the output of the previous function
	"""

	#Initialize figure
	if figShape is None:
		figShape = (np.ceil(float(len(redshiftSlices))/3).astype(int), 3)
	fig, axarr = plt.subplots(figShape[0], figShape[1], figsize=figsize, sharex=True, sharey=True)

	#Reshaping because subplots is dumb
	if figShape[0] == 1:
                axarr = axarr.reshape(1, len(axarr))
        elif figShape[1] == 1:
                axarr = axarr.reshape(len(axarr), 1)

	for e_index in range(len(massFuncts)):
		logMassBins, massFunctionRange = massFuncts[e_index]
		for z_index in range(len(redshiftSlices)):
			i = int(z_index / figShape[1])
			j = z_index % figShape[1]
			xaxis = 10**(0.5*(logMassBins[:-1]+logMassBins[1:]))
			axarr[i,j].fill_between(xaxis, massFunctionRange[:,z_index,0], massFunctionRange[:,z_index,1], label=labels[e_index], \
			color=colors[e_index], alpha=0.6)

			#If this is the last time around, put up the analytic halo mass functions and do some formatting.
			if e_index == len(massFuncts)-1:

				axarr[i,j].plot(xaxis, sf.calcNumberDensity(xaxis, redshiftSlices[z_index]), color='r', lw=2, ls='--', label='Analytic')
				axarr[i,j].text(xlim[0]*1.4,ylim[1]*1e-1,r'$z = {0}$'.format(redshiftSlices[z_index]), fontsize=12)
				if (i==0) & (j==axarr.shape[1]-1):
					axarr[i,j].legend(frameon=False, fontsize=10, loc='upper right')

	#Loop again just for final formatting.
	for i in range(axarr.shape[0]):
		for j in range(axarr.shape[1]):
			axarr[i,j].set_xscale('log')
			axarr[i,j].set_yscale('log')
			if i==figShape[0]-1:
				axarr[i,j].set_xlabel(r'$M_h \ [M_\odot]$', fontsize=12)
			if j==0:
				axarr[i,j].set_ylabel(r'$dN/d\log M_h \ [h^{3} \, \mathrm{Mpc}^{-3}]$', fontsize=12)
			axarr[i,j].set_xlim(xlim[0], xlim[1])
			axarr[i,j].set_ylim(ylim[0], ylim[1])

	fig.tight_layout()
	fig.subplots_adjust(hspace=0,wspace=0)
	if output is not None:
		fig.savefig(output)
	else:
		fig.show()

def estimateHaloCompleteness(massFuncts, redshiftSlices=[3.0], threshold=0.5, alphaBeta=(8.32,5.35), outfile=None):
	"""
	Find the halo mass above which the discrepancy is above than some value
	"""

	logMassBins, massFunctionRange = massFuncts[0]

	haloLimits = np.zeros(len(redshiftSlices))
	holeLimits = np.zeros(len(redshiftSlices))
	for z_index in range(len(redshiftSlices)):
		centralMasses = 10**(0.5*(logMassBins[:-1]+logMassBins[1:]))
		correctMassFunction = sf.calcNumberDensity(centralMasses, redshiftSlices[z_index])
		inferredMassFunction = np.average(massFunctionRange[:,z_index,:], axis=1)
		ratio = inferredMassFunction / correctMassFunction
		badBins = ratio < threshold
		offenders = np.where(np.diff(badBins.astype(int)) == 1)[0]
		if len(offenders) > 0:
			firstOffender = offenders[0] + 1
			incompleteHaloMass = centralMasses[firstOffender]
			incompleteBHMass = 10**(alphaBeta[0] + alphaBeta[1] * np.log10(sf.velocityDispersion(incompleteHaloMass, redshiftSlices[z_index])/200))
			haloLimits[z_index] = incompleteHaloMass
			holeLimits[z_index] = incompleteBHMass

	if outfile is not None:
		outDict = {'redshift': redshiftSlices, 'haloLimits': haloLimits, 'holeLimits': holeLimits}
		with open(outfile, 'w') as myfile:
			pickle.dump(outDict, myfile, protocol=2)
	else:
		return haloLimits, holeLimits

def haloToBlackHoleMassFunctions(haloMassFuncts, redshiftSlices=[3.0], alphaBeta=(8.49,4.38)):
	"""
	Convert halo mass functions into black hole mass functions

	Not doing uncertainties
	"""

	logMassBins, massFunctionRange = haloMassFuncts[0]
	centralHaloMasses = 10**(0.5*(logMassBins[1:]+logMassBins[:-1]))
	haloMassFunctions = np.transpose(np.average(massFunctionRange, axis=2))
	sigmas = np.array([sf.velocityDispersion(centralHaloMasses, z) for z in redshiftSlices])
	holeMasses = 10**(alphaBeta[0] + alphaBeta[1] * np.log10(sigmas/200.0))
	
	#This section is dedicated to estimating a derivative
        logMh = np.log10(centralHaloMasses)
        logsigma = np.log10(sigmas)
        dlogsigma_dlogMh = np.zeros((len(redshiftSlices), len(centralHaloMasses)))
        dlogsigma_dlogMh[:,1:-1] = (logsigma[:,2:] - logsigma[:,:-2]) / (logMh[2:] - logMh[:-2])
        dlogsigma_dlogMh[:,0] = (logsigma[:,1] - logsigma[:,0]) / (logMh[1] - logMh[0])
        dlogsigma_dlogMh[:,-1] = (logsigma[:,-1] - logsigma[:,-2]) / (logMh[-1] - logMh[-2])

	#Finally, the black hole mass function:
        holeMassFunctions = haloMassFunctions / dlogsigma_dlogMh / alphaBeta[1]

	#Resample so that the black hole masses are the same across redshift and evenly sampled
	logEvenHoleMasses = np.linspace(4, 11, 100)
	evenHoleMassFunctions = np.zeros((holeMassFunctions.shape[0],len(logEvenHoleMasses)))
	for z_index in range(holeMassFunctions.shape[0]):
		evenHoleMassFunctions[z_index,:] = np.interp(logEvenHoleMasses, np.log10(holeMasses[z_index,:]), \
		holeMassFunctions[z_index,:], right=0)

	return 10**logEvenHoleMasses, evenHoleMassFunctions

def estimateBlackHoleCompleteness(blackHoleMassFuncts, redshiftSlices=[3.0], threshold=0.5, alphaBeta=(8.49,4.38), outfile=None, \
	trueFunctions='./analyticMassFunctions.pkl', dexScatter=0.3):
	"""
	Find the black hole mass above which the discrepancy due to incompleteness is above some value
	"""

	centralMasses, massFunctionRange = blackHoleMassFuncts

	#Analytic black hole mass functions
	with open(trueFunctions, 'r') as myfile:
                dictionary = pickle.load(myfile)

        m_bh = dictionary['m_bh']
        numberDensity = dictionary['numberDensity']
	redshift = dictionary['redshift']

	#Also resample this so that black hole masses are the same across redshift and evenly sampled
	logEvenHoleMasses = np.log10(centralMasses)
	evenAnalyticMassFunctions = np.zeros((numberDensity.shape[0],logEvenHoleMasses.shape[0]))
	for z_index in range(len(redshift)):
		evenAnalyticMassFunctions[z_index,:] = np.interp(logEvenHoleMasses, np.log10(m_bh[z_index,:]), numberDensity[z_index,:], right=0)

	analyticMassFunction = RectBivariateSpline(redshift, logEvenHoleMasses, evenAnalyticMassFunctions)

        holeLimits = np.zeros(len(redshiftSlices))
	dexPerBin = np.diff(np.log10(centralMasses))[0]
	nBinsToConvolve = dexScatter / dexPerBin
	kernel = sf.makeGaussianSmoothingKernel(nBinsToConvolve)
        for z_index in range(len(redshiftSlices)):
		correctMassFunction = np.convolve(analyticMassFunction(redshiftSlices[z_index], np.log10(centralMasses))[0,:], kernel, mode='same')
		inferredMassFunction = np.convolve(massFunctionRange[z_index,:], kernel, mode='same')
                ratio = inferredMassFunction / correctMassFunction
                badBins = ratio < threshold
                offenders = np.where(np.diff(badBins.astype(int)) == 1)[0]
                if len(offenders) > 0:
                        firstOffender = offenders[0] + 1
                        incompleteHoleMass = centralMasses[firstOffender]
                        holeLimits[z_index] = incompleteHoleMass

        if outfile is not None:
                outDict = {'redshift': redshiftSlices, 'holeLimits': holeLimits}
                with open(outfile, 'w') as myfile:
                        pickle.dump(outDict, myfile, protocol=2)
        else:
                return redshiftSlices, holeLimits

if __name__ == '__main__':
	ensembles = ['blq_dcbh_a8.45b5.0_haloRatios_newSeeding_superEdd_061418/']
	labels = ['SAM']
	colors = ['b']

	#Uncomment to plot comparison with HMFCalc
	redshiftSlices = [0,20.0]; xlim=(5e6,1e15); ylim=(1e-6,1e3); figsize=(8,10)
	massFuncts = [calcHaloMassFunctions(e, redshiftSlices=redshiftSlices, logMassBins=np.linspace(np.log10(5e6),15,37)) for e in ensembles]
	plotMassFunctions(massFuncts, labels=labels, colors=colors, redshiftSlices=redshiftSlices, figsize=figsize, xlim=xlim, ylim=ylim)

	#Uncomment to make completeness file
	'''
	redshiftSlices = [0.1, 0.2, 0.4, 0.5, 0.6, 1.0, 1.6, 2.0, 2.15, 3.0, 3.2, 4.0, 4.75, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 14.0]
	massFuncts = [calcHaloMassFunctions(e, redshiftSlices=redshiftSlices) for e in ensembles]
	#estimateHaloCompleteness(massFuncts, redshiftSlices=redshiftSlices, threshold=0.5, outfile='./completeness_50.pkl')
	blackHoleMassFunction = haloToBlackHoleMassFunctions(massFuncts, redshiftSlices=redshiftSlices)
	estimateBlackHoleCompleteness(blackHoleMassFunction, redshiftSlices=redshiftSlices, outfile='./bh_completeness_50_convolved0.3.pkl', dexScatter=0.3)
	'''
