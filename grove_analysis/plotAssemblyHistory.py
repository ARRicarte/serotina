"""
ARR:  05.22.17

Plots the mass assembly history of black holes of different masses in the SAM
"""

import numpy as np
from .. import sam_functions as sf
import os
import matplotlib.pyplot as plt
import pickle
import gzip
currentPath = os.path.abspath(os.path.dirname(__file__)) + '/'

def computeAssemblyHistories(ensemble, hostMasses=[1e13], \
	n_bootstrap=10000, percentileRange=[14,86], normByZero=True, divideByMsigma=False, alphaBeta=(8.32,5.35)):

	"""
	Create average assembly histories for SMBHs in halos of different masses
	"""

	ensemble_files = [file for file in os.listdir(ensemble) if file[-5:]=='.pklz']
	ensembleHostMasses = np.array([10**float(file.split('_')[-1].split('m')[1].split('n')[0]) for file in ensemble_files])
	ensemble_files_of_closest_mass = [ensemble_files[np.argmin(np.abs(ensembleHostMasses - desiredMass))] for desiredMass in hostMasses]
	print("Reading from {0}".format(ensemble))

	massAssemblyHistories = [[] for mass in range(len(hostMasses))]
	usedMasses = []
	usableIndices = []

	for f_index in range(len(ensemble_files_of_closest_mass)):
		file = ensemble_files_of_closest_mass[f_index]
		hostHaloMass = 10**float(file.split('_')[-1].split('m')[1].split('n')[0])
		usedMasses.append(np.log10(hostHaloMass))
		n_tree = int(file.split('_')[-1].split('n')[1].split('.')[0])
		usableIndicesOfMass = np.ones(n_tree, dtype=bool)

		#Unpack data
		with gzip.open(ensemble+'/'+file, 'rb') as myfile:
			megaDict = pickle.load(myfile)
		uniqueRedshifts = np.flipud(np.unique(megaDict['redshift']))

		for i_tree in range(1,n_tree+1):
			treeIndexMask = megaDict['treeIndex'] == i_tree
			relevantIDs = megaDict['bh_id'][treeIndexMask]
			relevantHaloMasses = megaDict['m_halo'][treeIndexMask]
			finalHoleIndex = relevantIDs[np.argmax(relevantHaloMasses)]
			finalHoleMask = megaDict['bh_id'] == finalHoleIndex
			hasBHMask = megaDict['m_bh'] > 0
			combinedMask = treeIndexMask & finalHoleMask & hasBHMask
			if np.sum(combinedMask) != len(uniqueRedshifts):
				#Note:  This also catches the case where the host has no BH at z=0, but may have previously
				massAssemblyHistories[f_index].append(np.full(len(uniqueRedshifts), -np.inf))
				print("Warning:  Mask empty for m={0} and i={1}".format(hostMasses[f_index], i_tree))
				usableIndicesOfMass[i_tree-1] = False
				continue
			allPreviousMasses = megaDict['m_bh'][combinedMask]
			if normByZero:
				allPreviousMasses /= allPreviousMasses[-1]
			elif divideByMsigma:
				previousSigma = sf.velocityDispersion(megaDict['m_halo'][combinedMask], megaDict['redshift'][combinedMask])
				previousMsigma = 10**(alphaBeta[0] + alphaBeta[1]*np.log10(previousSigma/200.0))
				allPreviousMasses /= previousMsigma
			logAllPreviousMasses = np.log10(allPreviousMasses)
			massAssemblyHistories[f_index].append(logAllPreviousMasses)
		usableIndices.append(usableIndicesOfMass)

	usableIndices = np.array(usableIndices)
	massAssemblyHistories = np.array(massAssemblyHistories)

	#Do bootstrapping
	if n_bootstrap > 0:
		bootstraps = np.zeros((len(hostMasses),len(uniqueRedshifts),n_bootstrap))
		bootstrapRanges = np.zeros((len(hostMasses),len(uniqueRedshifts),2))
		for m_index in range(bootstraps.shape[0]):
			for boot in range(n_bootstrap):
				randomIndices = np.where(usableIndices[m_index])[0][np.random.randint(0, sum(usableIndices[m_index]), sum(usableIndices[m_index]))]
				bootstraps[m_index,:,boot] = np.average(massAssemblyHistories[m_index,randomIndices,:], axis=0)
			bootstrapRanges[m_index,:,:] = np.transpose(np.percentile(bootstraps[m_index,:,:], percentileRange, axis=1))
		return uniqueRedshifts, usedMasses, bootstrapRanges
	else:
		return uniqueRedshifts, usedMasses, massAssemblyHistories

def plotAssemblyHistories(redshifts, assemblyHistories, labels=['Light'], colors=['indigo', 'darkcyan', 'darkgreen', 'goldenrod', 'firebrick'], \
	masses=[1e11, 1e12, 1e13, 1e14, 1e15], figsize=(8,4), output=None, normalized=True, convertToTime=True):

	fig, axarr = plt.subplots(1, len(assemblyHistories), figsize=figsize, sharey=True, sharex=True)
	axarr = np.atleast_1d(axarr)

	if normalized:
		xlim = (10,0)
		ylim = (1e-4,1)
	else:
		xlim = (10,0)
		ylim = (1e2,3e10)

	if convertToTime:
		xvalues = sf.z2t(redshifts)
		xlabel = '$t_\mathrm{ABB} \ [\mathrm{Gyr}]$'
		xlim = (sf.z2t(xlim[0]), sf.z2t(xlim[1]))
	else:
		xvalues = redshifts
		xlabel = 'z'

	axtwins = []
	for panel in range(len(assemblyHistories)):

		for m_index in range(len(masses)):
			axarr[panel].fill_between(xvalues, 10**assemblyHistories[panel][m_index,:,0], 10**assemblyHistories[panel][m_index,:,1], \
			color=colors[m_index], label='$\log(M_0) = {0:2.0f}$'.format(masses[m_index]), alpha=0.7)

		axarr[panel].set_yscale('log')
		axarr[panel].set_xlabel(xlabel, fontsize=12)
		axarr[panel].set_ylim(ylim)
		axarr[panel].set_xlim(xlim)
		if panel==0:
			if normalized:
				axarr[panel].set_ylabel(r'$M_\bullet(z)/M_\bullet(0)$', fontsize=14)
			else:
				axarr[panel].set_ylabel(r'$M_\bullet(z) \ [M_\odot]$', fontsize=14)
		if panel == (len(assemblyHistories)-1):
			if normalized:
				axarr[panel].legend(loc='lower right', frameon=False, ncol=1)
			else:
				axarr[panel].legend(loc='upper left', frameon=False, ncol=2)
		if normalized:
			axarr[panel].text(1.6, 1.8e-4, labels[panel], fontsize=12)
		else:
			axarr[panel].text(1.6, 2e2, labels[panel], fontsize=12)

		if convertToTime:
			stringRedshifts = [0,0.5,1,2,3,4,6]
			redshifts = np.array(stringRedshifts)
			labeledTimes = sf.z2t(redshifts)
			axtwins.append(axarr[panel].twiny())
			axtwins[panel].set_xlim(axarr[panel].get_xlim())
			axtwins[panel].set_xticks(labeledTimes)
			axtwins[panel].set_xticklabels(stringRedshifts)
			axtwins[panel].set_xlabel('$z$', fontsize=14)

	fig.tight_layout()
	fig.subplots_adjust(hspace=0, wspace=0)

	if output is not None:
		fig.savefig(output)
	else:
		fig.show()

def plotRelativeMsigma(redshifts, assemblyHistories, labels=['Light'], colors=['indigo', 'darkcyan', 'darkgreen', 'goldenrod', 'firebrick'], \
	masses=[1e11, 1e12, 1e13, 1e14, 1e15], figsize=(8,4), output=None, ylim=(1e-1,1e1), xlim=(10,0)):

	fig, axarr = plt.subplots(1, len(assemblyHistories), figsize=figsize, sharey=True, sharex=True)
	axarr = np.atleast_1d(axarr)

	for panel in range(len(assemblyHistories)):
		for m_index in range(len(masses)):
			axarr[panel].fill_between(redshifts, 10**assemblyHistories[panel][m_index,:,0], 10**assemblyHistories[panel][m_index,:,1], \
			color=colors[m_index], label='$\log(M_0) = {0:2.0f}$'.format(masses[m_index]), alpha=0.7)

		axarr[panel].plot([xlim[0],xlim[1]], [1,1], lw=2, color='k', ls=':')
		axarr[panel].set_yscale('log')
		axarr[panel].set_xlabel('z', fontsize=12)
		axarr[panel].set_ylim(ylim)
		axarr[panel].set_xlim(xlim)
		if panel==0:
			axarr[panel].set_ylabel(r'$M_\bullet/M_\bullet(\sigma)$', fontsize=12)
			axarr[panel].legend(loc='lower left', frameon=False, ncol=2)
		axarr[panel].text(1.8, 1.3e-1, labels[panel], fontsize=12)

	fig.tight_layout()
	fig.subplots_adjust(hspace=0, wspace=0)

	if output is not None:
		fig.savefig(output)
	else:
		fig.show()


if __name__ == '__main__':
	#ensembles = ['agnms_popIII_100217', 'agnms_dcbh_072717']
	ensembles = ['powerLaw_dcbh_112917']
	labels = ['Light', 'Heavy']
	hostMasses = [1e11, 1e12, 1e13, 1e14, 1e15]
	colors = ['indigo', 'darkcyan', 'darkgreen', 'goldenrod', 'firebrick']
	normalized = True
	divideByMsigma = False

	assemblyHistories = []
	for e_index in range(len(ensembles)):
		redshifts, masses, ranges = computeAssemblyHistories(ensembles[e_index], percentileRange=[14,86], hostMasses=hostMasses, normByZero=normalized, divideByMsigma=divideByMsigma)
		assemblyHistories.append(ranges)
	plotAssemblyHistories(redshifts, assemblyHistories, labels=labels, colors=colors, masses=masses, normalized=normalized, figsize=(8,5))
	#plotRelativeMsigma(redshifts, assemblyHistories, labels=labels, colors=colors, masses=masses)
