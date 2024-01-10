"""
ARR: 04.19.17

Based on calcMassFunctions, just taking one aspect and allowing multiple ensembles as argument.
"""

import numpy as np
import os
import gzip
import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from ..helpers import sam_functions as sf
from .. import cosmology
from .. import constants
from ..helpers import absorptionFractions as af
from scipy.integrate import quad
from scipy.interpolate import interp1d
currentPath = os.path.abspath(os.path.dirname(__file__)) + '/'

def calcMassFunctions(ensemble, redshiftSlices=[0.6, 1.0, 2.0, 3.0, 4.0, 5.0], \
	n_mass=20, logRange=(11,15), n_sample=15, n_bootstrap=10000, plottedPercentiles=[14,86], \
	logBHMassBins=np.linspace(2,11,41), z_host=0, eddMin=0, weightByObscuration=False, numberOfDexToConvolve=0.0):

	"""
	Create mass functions
	"""

	#Storage for reading
	bh_masses = [np.array([]) for dummy in range(len(redshiftSlices))]
	bh_weights = [np.array([]) for dummy in range(len(redshiftSlices))]
	bh_fileIndices = [np.array([]) for dummy in range(len(redshiftSlices))]

	samplingFactor = float(n_mass) / (logRange[1] - logRange[0])
	ensemble_files = [file for file in os.listdir(ensemble) if file[-5:]=='.pklz']
	print("Reading from {0}".format(ensemble))
	
	for f_index in range(len(ensemble_files)):
		file = ensemble_files[f_index]
		hostHaloMass = 10**float(file.split('_')[-1].split('m')[1].split('n')[0])
		nHalos = int(file.split('_')[-1].split('n')[1].split('.')[0])
	
		#Weight by z=0 abundance
		weight = sf.calcNumberDensity(hostHaloMass, z_host) / nHalos / samplingFactor

		#Unpack data
		with gzip.open(ensemble+'/'+file, 'rb') as myfile:
			megaDict = pickle.load(myfile)
		uniqueRedshifts = np.unique(megaDict['redshift'])

		bhMask = (megaDict['m_bh'] > 0)
		for z_index in range(len(redshiftSlices)):
			closestRedshift = uniqueRedshifts[np.argmin(np.abs(uniqueRedshifts - redshiftSlices[z_index]))]
			if np.abs(closestRedshift - redshiftSlices[z_index]) > 0.1:
				print("Warning:  Wanted z={0}, but we're using z={1}.".format(redshiftSlices[z_index], closestRedshift))

			redshiftMask = (megaDict['redshift'] == closestRedshift)
			eddRatios = megaDict['L_bol'] / sf.eddingtonLum(megaDict['m_bh'])
			eddRatioMask = (eddRatios >= eddMin)
			combinedMask = redshiftMask & bhMask & eddRatioMask
			if weightByObscuration:
				weights = weight * af.typeIProbability(megaDict['L_bol'][combinedMask], np.full(np.sum(combinedMask), closestRedshift), numberOfDexToConvolve=numberOfDexToConvolve)
				weights[megaDict['L_bol'][combinedMask] == 0] = 0
			else:
				weights = np.full(np.sum(combinedMask), weight)

			bh_weights[z_index] = np.concatenate((bh_weights[z_index], weights))
			bh_masses[z_index] = np.concatenate((bh_masses[z_index], megaDict['m_bh'][combinedMask]))
			bh_fileIndices[z_index] = np.concatenate((bh_fileIndices[z_index], megaDict['treeIndex'][combinedMask]))

	#Storage for bootstrapping
	massFunctionPieces = np.zeros((len(logBHMassBins)-1,len(redshiftSlices),n_sample))
	massFunctions = np.zeros((len(logBHMassBins)-1,len(redshiftSlices),n_bootstrap))
	massFunctionRange = np.zeros((len(logBHMassBins)-1,len(redshiftSlices),2))

	#Compute the mass function from this 2D histogram
	for z_index in range(len(redshiftSlices)):
		print("z = {0}:".format(redshiftSlices[z_index]))
		closestRedshift = uniqueRedshifts[np.argmin(np.abs(uniqueRedshifts - redshiftSlices[z_index]))]

		#Collect the pieces for bootstrapping
		print("   Creating bootstrap pieces.")
		for treeIndex in range(n_sample):
			massFunctionPieces[:,z_index,treeIndex] = np.histogram(np.log10(bh_masses[z_index][bh_fileIndices[z_index]==treeIndex+1]), bins=logBHMassBins, \
			weights=bh_weights[z_index][bh_fileIndices[z_index]==treeIndex+1]/np.diff(logBHMassBins)[0])[0]

		#Bootstrapping!
		print("   Computing mass function by bootstrapping.  Number of samples = {0}.".format(n_bootstrap))
		for boot in range(n_bootstrap):
			randomFileNumbers = np.random.randint(0, n_sample, n_sample)
			massFunctions[:,z_index,boot] = np.sum(massFunctionPieces[:,z_index,randomFileNumbers], axis=1)

		#Finally, find the central 68% of all values.
		massFunctionRange[:,z_index,:] = np.transpose(np.percentile(massFunctions[:,z_index,:], plottedPercentiles, axis=1))

	return logBHMassBins, massFunctionRange

def plotMassFunctions(massFuncts, labels=None, colors=None, redshiftSlices=[0.6, 1.0, 2.0, 3.0, 4.0, 5.0], figsize=(8,6), \
	figShape=None, xlim=(5e6,2e10), ylim=(1e-6,1e-1), labelData=True, output=None, includeQuasars=False, quasarScaling=1, \
	includeMH=True, treeLimits=None, showAnalytic=False, showCompleteness=False, numberOfDexToConvolve=0.0, completenessFile='./bh_completeness_50_convolved0.3.pkl'):
	"""
	Plot the output of the previous function
	"""

	#Initialize figure
	if figShape is None:
		figShape = (np.ceil(float(len(redshiftSlices))/3).astype(int), min(3,len(redshiftSlices)))
	fig, axarr = plt.subplots(figShape[0], figShape[1], figsize=figsize, sharex=True, sharey=True)

	#Reshaping because subplots is dumb
	axarr = np.atleast_2d(axarr)
	if figShape[0] == 1:
		axarr = axarr.reshape(1, len(axarr))
	elif figShape[1] == 1:
		axarr = axarr.reshape(len(axarr), 1)

	#Merloni & Heinz data
	mh_redshifts = np.array([0.6, 1.0, 2.0, 3.0, 4.0, 5.0])
	mh_path = currentPath + '../lookup_tables/bh_data/Merloni_Heinz_2008/area_data/'

	if includeQuasars:
		with open(currentPath + "../lookup_tables/bh_data/Kelly_Shen_2013/KellyShen2013.pkl", 'rb') as myfile:
			quasarObservations = pickle.load(myfile, encoding='latin1')
			z_ks13 = quasarObservations['redshifts']

	if showCompleteness:
		with open(completenessFile, 'rb') as myfile:
			completenessTable = pickle.load(myfile, encoding='latin1')

	for e_index in range(len(massFuncts)):
		logBHMassBins, massFunctionRange = massFuncts[e_index]
		for z_index in range(len(redshiftSlices)):
			i = int(z_index / figShape[1])
			j = z_index % figShape[1]
			xaxis = 10**(0.5*(logBHMassBins[:-1]+logBHMassBins[1:]))

			#Convolve with a lognormal
			dexPerBin = np.diff(logBHMassBins)[0]
			convolvedWidth = numberOfDexToConvolve / dexPerBin
			kernel = sf.makeGaussianSmoothingKernel(convolvedWidth)
			top = np.convolve(massFunctionRange[:,z_index,1], kernel, mode='same')
			bottom = np.convolve(massFunctionRange[:,z_index,0], kernel, mode='same')
			axarr[i,j].fill_between(xaxis, bottom, top, label=labels[e_index], \
			color=colors[e_index], alpha=0.7)

			#If this is the last time around, put up the MH data points and do some formatting.
			if e_index == len(massFuncts)-1:
				if includeMH:
					closestRedshift = mh_redshifts[np.argmin(np.abs(mh_redshifts-redshiftSlices[z_index]))]
					if np.abs(closestRedshift - redshiftSlices[z_index]) < 0.2:
						#Complicated code to read data and create a polygon
						with open(mh_path+'mh08_{0:2.1f}_top.dat'.format(closestRedshift), 'r') as myfile:
							data_top = np.loadtxt(myfile)
						with open(mh_path+'mh08_{0:2.1f}_bottom.dat'.format(closestRedshift), 'r') as myfile:
							data_bottom = np.loadtxt(myfile)
						poly_x = np.concatenate((10**data_top[:,0],np.flipud(10**data_bottom[:,0])))
						poly_y = np.concatenate((data_top[:,1]/cosmology.h**3,np.flipud(data_bottom[:,1]/cosmology.h**3)))
						polygon = [Polygon(zip(poly_x,poly_y))]
						axarr[i,j].add_collection(PatchCollection(polygon, alpha=0.7, color='k', hatch='/'))
						
						#For some reason, labeling doesn't work in ax.add_collection().  This is my work-around.  Also good in case no data in this panel.
						if labelData:
							axarr[i,j].fill_between([], [], [], alpha=0.7, color='k', label='MH08', hatch='/')

				if redshiftSlices[z_index]==0:
					#Complicated code to read data and create a polygon
					with open(currentPath + '../lookup_tables/bh_data/shankar09/shankar09_top.dat', 'r') as myfile:
						data_top = np.loadtxt(myfile)
					with open(currentPath + '../lookup_tables/bh_data/shankar09/shankar09_bottom.dat', 'r') as myfile:
						data_bottom = np.loadtxt(myfile)
					poly_x = np.concatenate((10**data_top[:,0],np.flipud(10**data_bottom[:,0])))
					poly_y = np.concatenate((data_top[:,1]/cosmology.h**3,np.flipud(data_bottom[:,1]/cosmology.h**3)))
					polygon = [Polygon(zip(poly_x,poly_y))]
					axarr[i,j].add_collection(PatchCollection(polygon, alpha=0.7, color='k', hatch='/'))
					axarr[i,j].fill_between([], [], [], alpha=0.7, color='k', label='Shankar+09', hatch='/')

				if showAnalytic:
					amf_mass, amf_density = sf.analyticMassFunction(redshiftSlices[z_index])
					m_bh_even = np.logspace(5, 10, 100)
					numberDensity_even = np.interp(np.log10(m_bh_even), np.log10(amf_mass), amf_density)
					axarr[i,j].plot(m_bh_even, sf.convolve(numberDensity_even, numberOfDexToConvolve, np.diff(np.log10(m_bh_even))[0]), lw=2, color='red', ls='--', label='Analytic')

				#Add in the closest redshift bin from KS13
				if includeQuasars:
					closestRedshift_ks13 = z_ks13[np.argmin(np.abs(z_ks13-redshiftSlices[z_index]))]
					if np.abs(closestRedshift_ks13 - redshiftSlices[z_index]) < 0.5:
						if quasarScaling != 1:
							quasarLabel = 'KS13x{0}'.format(quasarScaling)
						else:
							quasarLabel = 'KS13'
						axarr[i,j].fill_between(10**quasarObservations['log_m_bh'], quasarScaling*10**quasarObservations['logPhiRange'][closestRedshift_ks13][:,0]/cosmology.h**3, \
						quasarScaling*10**quasarObservations['logPhiRange'][closestRedshift_ks13][:,1]/cosmology.h**3, alpha=0.7, color='k', \
						label=quasarLabel, hatch='/')

				axarr[i,j].text(xlim[0]*1.4,ylim[1]*4e-1,r'$z = {0}$'.format(redshiftSlices[z_index]), fontsize=12)
				if includeQuasars:
					if (i==figShape[0]-1) & (j==figShape[1]-1):
						axarr[i,j].legend(frameon=False, fontsize=9, loc='upper right')
				elif includeMH:
					if (i==0) & (j==0):
						axarr[i,j].legend(frameon=False, fontsize=9, loc='lower left')

				if treeLimits is not None:
					axarr[i,j].plot([treeLimits[z_index][0],treeLimits[z_index][0]], [1e-99,1e99], lw=2, ls=':', color='orange')

				if showCompleteness:
					closestRedshiftIndex_completeness = np.argmin(np.abs(np.array(redshiftSlices[z_index]) - completenessTable['redshift']))
					m_max = completenessTable['holeLimits'][closestRedshiftIndex_completeness]
					axarr[i,j].plot([m_max,m_max], [1e-99,1e99], lw=2, ls=':', color='orange')

	#Loop again just for final formatting.
	for i in range(axarr.shape[0]):
		for j in range(axarr.shape[1]):
			axarr[i,j].set_xscale('log')
			axarr[i,j].set_yscale('log')
			if i==figShape[0]-1:
				axarr[i,j].set_xlabel(r'$M_\bullet \ [M_\odot]$', fontsize=12)
			if j==0:
				axarr[i,j].set_ylabel(r'$dN/d\log M_\bullet \ [h^{3} \, \mathrm{Mpc}^{-3}]$', fontsize=12)
			axarr[i,j].set_xlim(xlim[0], xlim[1])
			axarr[i,j].set_ylim(ylim[0], ylim[1])

	fig.tight_layout()
	fig.subplots_adjust(hspace=0,wspace=0)
	if output is not None:
		fig.savefig(output)
	else:
		fig.show()

def computeTreeLimits(ensemble, redshiftSlices, alphaBeta=(8.32,5.35), biggestFile='m15.00n15.pklz', z_host=0, \
	n_mass=20, logRange=(11,15)):
	"""
	DEPRECATED:  Now computing limits based on HMF

	Compute the maximum mass and the minimum abundance probed by the merger tree at a list of redshifts.
	"""

	samplingFactor = float(n_mass) / (logRange[1] - logRange[0])

	#Unpack data
	with gzip.open(ensemble+'/'+biggestFile, 'rb') as myfile:
		megaDict = pickle.load(myfile)
	uniqueRedshifts = np.unique(megaDict['redshift'])
	hostHaloMass = 10**float(biggestFile.split('_')[-1].split('m')[1].split('n')[0])
	nHalos = int(biggestFile.split('_')[-1].split('n')[1].split('.')[0])
	limits = []

	for z_index in range(len(redshiftSlices)):
		closestRedshift = uniqueRedshifts[np.argmin(np.abs(uniqueRedshifts - redshiftSlices[z_index]))]

		redshiftMask = megaDict['redshift'] == closestRedshift
		biggestHaloAtRedshift = np.max(megaDict['m_halo'][redshiftMask])
		correspondingBlackHoleMass = 10**(alphaBeta[0] + alphaBeta[1] * np.log10(sf.velocityDispersion(biggestHaloAtRedshift, closestRedshift)/200.0))
		abundanceOfOne = sf.calcNumberDensity(hostHaloMass, z_host) / nHalos / samplingFactor
		limits.append([correspondingBlackHoleMass, abundanceOfOne])

	return limits

def computeMassDensity(massFuncts, redshiftSlices=[0], z=0, numberOfDexToConvolve=0.3):
	densities = []
	for mf in massFuncts:
		logBHMassBins, massFunctionRange = mf
		logBHMassBins = 0.5 * (logBHMassBins[1:] + logBHMassBins[:-1])
		dexPerBin = np.diff(logBHMassBins)[0]
		convolutionFactor = np.exp(0.5 * (numberOfDexToConvolve * np.log(10))**2)
		n_function = interp1d(logBHMassBins, np.average(massFunctionRange[:,0,:], axis=1) * cosmology.h**3 * convolutionFactor)
		integral = quad(lambda logm: 10**logm*n_function(logm), logBHMassBins[0], logBHMassBins[-1])
		print("The log of the local mass density is {0:3.2f} solar masses per cubic Mpc.".format(np.log10(integral[0])))
		densities.append(np.log10(integral[0]))
	return densities

if __name__ == '__main__':

	'''
	ensembles = ['retuned_dcbh_070517', 'random_dcbh_nomergers_072417', 'mainSequence_dcbh_071717']
	colors = ['b', 'c', 'r']
	labels = ['Fiducial', 'NoMerge', 'AGNMS']
	'''
	ensembles = ['agnms_dcbh_072717']
	colors = ['r']
	labels = ['AGNMS']

	#Uncomment for comparison with Merloni & Heinz (2008)
	#redshiftSlices = [0.6, 1.0, 2.0, 3.0, 4.0, 5.0]; xlim=(5e6,2e10); ylim=(1e-6,1e-1); figsize=(8,5)
	redshiftSlices = [0]; xlim=(1e6,3e9); ylim=(1e-5,1e-1); figsize=(5,5)

	#Uncomment for JWST era
	#redshiftSlices = [6.0, 7.0, 8.0, 9.0, 10.0]; xlim=(5e4,1e9); ylim=(1e-6,1e0); figsize=(8,5)

	#Uncomment for BLQs
	#redshiftSlices = [0.6, 1.0, 1.6, 2.15, 3.2, 4.75]; xlim=(1e8,1e11); ylim=(2e-8,3e-3); figsize=(8,5)

	#massFuncts = [calcMassFunctions(e, redshiftSlices=redshiftSlices, weightByObscuration=True, numberOfDexToConvolve=0.0, eddMin=1e-4) for e in ensembles]
	#plotMassFunctions(massFuncts, labels=labels, colors=colors, redshiftSlices=redshiftSlices, figsize=figsize, xlim=xlim, ylim=ylim, \
	#includeQuasars=True, includeMH=True, showAnalytic=True)

	massFuncts = [calcMassFunctions(e, redshiftSlices=redshiftSlices, weightByObscuration=False) for e in ensembles]
	plotMassFunctions(massFuncts, labels=labels, colors=colors, redshiftSlices=redshiftSlices, figsize=figsize, xlim=xlim, ylim=ylim, \
	includeQuasars=False, includeMH=True, showAnalytic=True, numberOfDexToConvolve=0.3)
	#computeMassDensity(massFuncts)
