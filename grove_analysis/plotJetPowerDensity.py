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
from .. import constants
from .. import cosmology
from ..accretion_recipes import spinEvolverMAD as jets
currentPath = os.path.abspath(os.path.dirname(__file__)) + '/'

def computeKineticPowerDensity(ensemble, n_mass=20, logRange=(11,15), n_sample=15, \
	n_bootstrap=10000, percentileRange=[14,86], z_host=0, mdot_Edd_hot_thin=3e-3, mdot_Range=[0,np.inf]):

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
			totalJetPowers = [np.array([]) for dummy in range(n_z)]
			fileIndices = [np.array([]) for dummy in range(n_z)]
			storageSet = True

		#Assemble data for each redshift
		hasBHMask = megaDict['m_bh'] > 0
		for z_index in range(len(uniqueRedshifts)):
			redshiftMask = megaDict['redshift'] == uniqueRedshifts[z_index]
			accretionMask = (megaDict['eddRatio'] >= mdot_Range[0]) & (megaDict['eddRatio'] <= mdot_Range[1])
			combinedMask = redshiftMask & hasBHMask & accretionMask
			weights[z_index] = np.concatenate((weights[z_index], np.full(np.sum(combinedMask), weight)))
			jetPower = []
			for i in np.where(combinedMask)[0]:
				jetPower.append(jets.calcJetPower(megaDict['m_bh'][i], megaDict['spin_bh'][i], megaDict['eddRatio'][i], mdot_Edd_hot_thin=mdot_Edd_hot_thin))
			jetPower = np.array(jetPower)
			totalJetPowers[z_index] = np.concatenate((totalJetPowers[z_index], jetPower))
			fileIndices[z_index] = np.concatenate((fileIndices[z_index], megaDict['treeIndex'][combinedMask]))

	#Compute density for each file index
	kineticDensityPieces = np.zeros((len(uniqueRedshifts),n_tree))
	for z_index in range(len(uniqueRedshifts)):
		for treeIndex in range(1,n_tree+1):
			treeIndexMask = fileIndices[z_index] == treeIndex
			kineticDensityPieces[z_index,treeIndex-1] = np.log10(np.sum(totalJetPowers[z_index][treeIndexMask] * weights[z_index][treeIndexMask]))

	#Do bootstrapping
	bootstrapsJetPower = np.zeros((len(uniqueRedshifts),n_bootstrap))
	bootstrapRangesJetPower = np.zeros((len(uniqueRedshifts),2))
	for boot in range(n_bootstrap):
		randomIndices = np.random.randint(0, n_tree, n_tree)
		bootstrapsJetPower[:,boot] = np.average(kineticDensityPieces[:,randomIndices], axis=1)
		
	bootstrapRangesJetPower[:,:] = np.transpose(np.percentile(bootstrapsJetPower, percentileRange, axis=1))

	return uniqueRedshifts, bootstrapRangesJetPower

def plotKineticPowerDensity(redshifts, bootstrapRangesJetPower, figsize=(4,4), output=None, colors=['b'], labels=['Light'], xlim=(0,6), ylim=(1e29,1e35), \
	numberOfDexToConvolve=0.3, radiativeEfficiency=0.1, fig_ax=None, legend=True, alpha=0.7):

	if fig_ax is None:
		fig, ax = plt.subplots(figsize=figsize)
	else:
		fig, ax = fig_ax

	#This is the mean of a lognormal
	convolutionCoefficient = np.exp(0.5 * (numberOfDexToConvolve * np.log(10))**2)
	logc = np.log10(convolutionCoefficient)
	for e_index in range(len(bootstrapRangesJetPower)):
		bottom = logc+bootstrapRangesJetPower[e_index][:,0]
		top = logc+bootstrapRangesJetPower[e_index][:,1]
		ax.fill_between(redshifts, 10**bottom, 10**top, color=colors[e_index], label=labels[e_index], alpha=alpha)

	ax.set_xlabel('$z$', fontsize=12)
	ax.set_ylabel(r'Kinetic Power Density$\ [\mathrm{W} \; \mathrm{Mpc}^{-3}]$', fontsize=12)
	ax.set_xlim(xlim)
	ax.set_ylim(ylim)
	ax.set_yscale('log')
	if legend:
		ax.legend(loc='lower left', frameon=False)

	fig.tight_layout()
	if output is not None:
		fig.savefig(output)
	elif fig_ax is None:
		fig.show()
