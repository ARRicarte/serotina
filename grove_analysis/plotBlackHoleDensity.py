"""
ARR:  05.23.17

Plots the mass assembly history of black holes of different masses in the SAM
"""

import numpy as np
from ..helpers import sam_functions as sf
import os
import matplotlib.pyplot as plt
import pickle
import gzip
from . import addDensityLimits as adl
from .. import constants
from .. import cosmology
currentPath = os.path.abspath(os.path.dirname(__file__)) + '/'

def computeBlackHoleDensity(ensemble, n_mass=20, logRange=(11,15), n_sample=15, \
	n_bootstrap=10000, percentileRange=[14,86], z_host=0):

	"""
	Create average assembly histories for SMBHs in halos of different masses
	"""

	ensemble_files = [file for file in os.listdir(ensemble) if file[-5:]=='.pklz']
	ensembleHostMasses = np.array([10**float(file.split('_')[-1].split('m')[1].split('n')[0]) for file in ensemble_files])
	print("Reading from {0}".format(ensemble))

	samplingFactor = float(n_mass) / (logRange[1] - logRange[0])
	storageSet = False

	for f_index in range(len(ensemble_files)):
		file = ensemble_files[f_index]
		hostHaloMass = 10**float(file.split('_')[-1].split('m')[1].split('n')[0])
		n_tree = int(file.split('_')[-1].split('n')[1].split('.')[0])
		#NOTE:  n_tree factor removed because averaging is used during bootstrapping instead of summing.
		weight = sf.calcHaloNumberDensity(hostHaloMass, z_host) / samplingFactor * cosmology.h**3

		#Unpack data
		with gzip.open(ensemble+'/'+file, 'r') as myfile:
			megaDict = pickle.load(myfile)

		#Create some storage arrays
		if not storageSet:
			uniqueRedshifts = np.flipud(np.unique(megaDict['redshift']))
			n_z = len(uniqueRedshifts)
			weights = [np.array([]) for dummy in range(n_z)]
			masses = [np.array([]) for dummy in range(n_z)]
			accretedMasses = [np.array([]) for dummy in range(n_z)]
			totalLuminosities = [np.array([]) for dummy in range(n_z)]
			fileIndices = [np.array([]) for dummy in range(n_z)]
			storageSet = True

		#Assemble data for each redshift
		hasBHMask = megaDict['m_bh'] > 0
		for z_index in range(len(uniqueRedshifts)):
			redshiftMask = megaDict['redshift'] == uniqueRedshifts[z_index]
			combinedMask = redshiftMask & hasBHMask
			weights[z_index] = np.concatenate((weights[z_index], np.full(np.sum(combinedMask), weight)))
			masses[z_index] = np.concatenate((masses[z_index], megaDict['m_bh'][combinedMask]))
			accretedMasses[z_index] = np.concatenate((accretedMasses[z_index], megaDict['m_burst'][combinedMask]+megaDict['m_steady'][combinedMask]))
			totalLuminosities[z_index] = np.concatenate((totalLuminosities[z_index], megaDict['L_bol'][combinedMask]))
			fileIndices[z_index] = np.concatenate((fileIndices[z_index], megaDict['treeIndex'][combinedMask]))

	#Compute density for each file index
	densityPieces = np.zeros((len(uniqueRedshifts),n_tree))
	accretedDensityPieces = np.zeros((len(uniqueRedshifts),n_tree))
	luminosityDensityPieces = np.zeros((len(uniqueRedshifts),n_tree))
	for z_index in range(len(uniqueRedshifts)):
		for treeIndex in range(1,n_tree+1):
			treeIndexMask = fileIndices[z_index] == treeIndex
			densityPieces[z_index,treeIndex-1] = np.log10(np.sum(masses[z_index][treeIndexMask] * weights[z_index][treeIndexMask]))
			accretedDensityPieces[z_index,treeIndex-1] = np.log10(np.sum(accretedMasses[z_index][treeIndexMask] * weights[z_index][treeIndexMask]))
			luminosityDensityPieces[z_index,treeIndex-1] = np.log10(np.sum(totalLuminosities[z_index][treeIndexMask] * weights[z_index][treeIndexMask]))

	#Do bootstrapping
	bootstrapsDensity = np.zeros((len(uniqueRedshifts),n_bootstrap))
	bootstrapsAccretedDensity = np.zeros((len(uniqueRedshifts),n_bootstrap))
	bootstrapsLuminosity = np.zeros((len(uniqueRedshifts),n_bootstrap))

	bootstrapRangesDensity = np.zeros((len(uniqueRedshifts),2))
	bootstrapRangesAccretedDensity = np.zeros((len(uniqueRedshifts),2))
	bootstrapRangesLuminosity = np.zeros((len(uniqueRedshifts),2))

	for boot in range(n_bootstrap):
		randomIndices = np.random.randint(0, n_tree, n_tree)
		bootstrapsDensity[:,boot] = np.average(densityPieces[:,randomIndices], axis=1)
		bootstrapsAccretedDensity[:,boot] = np.average(accretedDensityPieces[:,randomIndices], axis=1)
		bootstrapsLuminosity[:,boot] = np.average(luminosityDensityPieces[:,randomIndices], axis=1)
		
	bootstrapRangesDensity[:,:] = np.transpose(np.percentile(bootstrapsDensity, percentileRange, axis=1))
	bootstrapRangesAccretedDensity[:,:] = np.transpose(np.percentile(bootstrapsAccretedDensity, percentileRange, axis=1))
	bootstrapRangesLuminosity[:,:] = np.transpose(np.percentile(bootstrapsLuminosity, percentileRange, axis=1))

	return uniqueRedshifts, bootstrapRangesDensity, bootstrapRangesAccretedDensity, bootstrapRangesLuminosity

def plotBlackHoleDensities(redshifts, densitiesTotal=None, figsize=(4,4), output=None, colors=['b'], labels=['Light'], xlim=(0,15), ylim=(1,6.5), \
	addObservations=True, densitiesAccreted=None, numberOfDexToConvolve=0.3, showYue=False):

	fig, ax = plt.subplots(figsize=figsize)

	if addObservations:
		adl.addDensityLimits(ax)

	#This is a theoretical prediction from a DCBH model.
	if showYue:
		with open("./Yue2013/Yue13_totalBHDensity.dat", 'r') as myfile:
			dataYue = np.loadtxt(myfile)
		ax.plot(dataYue[:,0], dataYue[:,1], lw=3, color='forestgreen', ls='--', label='Y13 (Tot.)')

	#This is the mean of a lognormal
	convolutionCoefficient = np.exp(0.5 * (numberOfDexToConvolve * np.log(10))**2)
	logc = np.log10(convolutionCoefficient)
	for e_index in range(len(densitiesAccreted)):
		if densitiesTotal is not None:
			ax.plot(redshifts, np.average(np.vstack((logc+densitiesTotal[e_index][:,0], logc+densitiesTotal[e_index][:,1])), axis=0), color=colors[e_index], \
			ls='--', lw=1, label=labels[e_index]+' (Tot.)')
		if densitiesAccreted is not None:
			ax.fill_between(redshifts, logc+densitiesAccreted[e_index][:,0], logc+densitiesAccreted[e_index][:,1], \
			color=colors[e_index], alpha=1.0, label=labels[e_index]+' (Acc.)')

	ax.set_xlabel('$z$', fontsize=12)
	ax.set_ylabel(r'$\log_{10} \rho \ [M_\odot \mathrm{Mpc}^{-3}]$', fontsize=12)
	ax.legend(loc='lower left', frameon=False, ncol=1, fontsize=8)
	ax.set_xlim(xlim[0],xlim[1])
	ax.set_ylim(ylim[0],ylim[1])
	
	fig.tight_layout()

	if output is not None:
		fig.savefig(output)
	else:
		fig.show()

def plotAccretionDensities(redshifts, luminosityDensities, figsize=(4,4), output=None, colors=['b'], labels=['Light'], xlim=(0,10), \
	numberOfDexToConvolve=0.3, radiativeEfficiency=0.1):

	fig, ax = plt.subplots(figsize=figsize)

	#This is the mean of a lognormal
	convolutionCoefficient = np.exp(0.5 * (numberOfDexToConvolve * np.log(10))**2)
	logc = np.log10(convolutionCoefficient)
	for e_index in range(len(luminosityDensities)):
		logUnitConversion = np.log10(constants.L_sun / constants.M_sun / radiativeEfficiency / constants.c**2 * constants.yr)
		bottom = logc+luminosityDensities[e_index][:,0] + logUnitConversion
		top = logc+luminosityDensities[e_index][:,1] + logUnitConversion
		ax.fill_between(redshifts, bottom, top, color=colors[e_index], \
		label=labels[e_index], alpha=0.7)

	ax.set_xlabel('$z$', fontsize=12)
	ax.set_ylabel(r'$\log_{10} \dot{\rho} \ [M_\odot \; \mathrm{yr}^{-1} \; \mathrm{Mpc}^{-3}]$', fontsize=12)
	ax.set_xlim(0,15)
	ax.set_ylim(-7,-3.5)
	ax.legend(loc='lower left', frameon=False)

	fig.tight_layout()

	if output is not None:
		fig.savefig(output)
	else:
		fig.show()

if __name__ == '__main__':
	'''
	ensembles = ['powerLaw_popIII_021318', 'agnms_popIII_021518', 'powerLaw_dcbh_112917', 'agnms_dcbh_120617']
	colors = ['r', 'orange', 'purple', 'c']
	labels = ['Light-PL', 'Light-MS', 'Heavy-PL', 'Heavy-MS']
	'''
	ensembles = ['blq_dcbh_pmerge0.1_z4_072018', 'blq_popIII_pmerge0.1_z4_072018']
	colors = ['firebrick', 'royalblue']
	labels = ['Heavy', 'Light']

	numberOfDexToConvolve = 0.3

	'''
	densities = []
	densitiesAccreted = []
	luminosityDensities = []
	for e_index in range(len(ensembles)):
		redshifts, ranges, rangesAccreted, rangesLumDensity = computeBlackHoleDensity(ensembles[e_index], z_host=4, logRange=(10,13.4), n_mass=35, n_sample=100)
		#redshifts, ranges, rangesAccreted, rangesLumDensity = computeBlackHoleDensity(ensembles[e_index], z_host=0)
		densities.append(ranges)
		densitiesAccreted.append(rangesAccreted)
		luminosityDensities.append(rangesLumDensity)
	'''
	#plotBlackHoleDensities(redshifts, densitiesTotal=None, densitiesAccreted=densitiesAccreted, labels=labels, colors=colors, numberOfDexToConvolve=0.3, showYue=False)
	plotBlackHoleDensities(redshifts, densitiesTotal=densities, densitiesAccreted=densitiesAccreted, labels=labels, colors=colors, numberOfDexToConvolve=0.3, showYue=True, xlim=(0,17), ylim=(1.3,6.7))
	#plotAccretionDensities(redshifts, luminosityDensities, labels=labels, colors=colors, numberOfDexToConvolve=0.3)
