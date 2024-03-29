"""
ARR: 01.16.24

Compute and plot spin distributions.
"""

import numpy as np
import os
import gzip
import pickle
import matplotlib.pyplot as plt
from ..helpers import sam_functions as sf
from .. import cosmology
from .. import constants
from .. import util
from ..cosmology import cosmology_functions as cf
from scipy import stats
import matplotlib.patches as patches
import warnings
currentPath = os.path.abspath(os.path.dirname(__file__)) + '/'

def computeSpinDistribution(ensemble, redshifts, Mbh_range=[0,np.inf], Mhalo_range=[0,np.inf], Mstar_range=[0,np.inf], Lbol_range=[0,np.inf], fEdd_range=[0,np.inf], \
	treeStartingRedshift=0, transformFunction=None, n_bootstrap=10000, xaxis=np.linspace(-1,1,1000), normalize=True, output=None):
	"""
	Read a grove, then produce a Gaussian KDE function of spins.
	Weighted by halo number densities.
	Some filtering can be applied.
	"""

	if not isinstance(redshifts, list):
		redshifts = [redshifts]

	spinValueList = [[] for z in redshifts]
	spinWeightList =[[] for z in redshifts]
	treeIndexList = [[] for z in redshifts]

	print("Reading data.")
	ensemble_files = [file for file in os.listdir(ensemble) if file[-5:]=='.pklz']
	for f_index in range(len(ensemble_files)):
		file = ensemble_files[f_index]
		hostHaloMass = 10**float(file.split('_')[-1].split('m')[1].split('n')[0])
		nHalos = int(file.split('_')[-1].split('n')[1].split('.')[0])
		with gzip.open(os.path.join(ensemble, file), 'rb') as myfile:
			megaDict = pickle.load(myfile)

		for z_index in range(len(redshifts)):
			#Find closest redshift
			uniqueRedshifts = np.unique(megaDict['redshift'])	
			closestRedshift = uniqueRedshifts[np.argmin(np.abs(uniqueRedshifts - redshifts[z_index]))]

			#Apply selection criteria
			mask = megaDict['redshift'] == closestRedshift

			#Pass if some keys were not saved.

			#m_star was bugged.  Skipping...
			filterKeys = ['m_bh', 'm_halo', 'L_bol', 'eddRatio']
			filterRanges = [Mbh_range, Mhalo_range, Lbol_range, fEdd_range]
			'''
			filterKeys = ['m_bh', 'm_halo', 'm_star', 'L_bol', 'eddRatio']
			filterRanges = [Mbh_range, Mhalo_range, Mstar_range, Lbol_range, fEdd_range]
			'''
			for key, filterRange in zip(filterKeys, filterRanges):
				try:
					mask = mask & (megaDict[key] >= filterRange[0]) & (megaDict[key] <= filterRange[1])
				except KeyError:
					warnings.warn(f"{key} was not saved in this file. Skipping this filter.")
					pass
			spinValueList[z_index].extend(megaDict['spin_bh'][mask])
			spinWeightList[z_index].extend(np.full(np.sum(mask), sf.calcHaloNumberDensity(hostHaloMass, treeStartingRedshift)))
			treeIndexList[z_index].extend(megaDict['treeIndex'][mask])

	distributions = []

	for z_index in range(len(redshifts)):
		#We now have values and associated weights
		spinValues = np.array(spinValueList[z_index])
		spinWeights = np.array(spinWeightList[z_index])
		treeIndices = np.array(treeIndexList[z_index])

		if len(spinValues) == 0:
			#This will happen if there was nothing passing the masking criteria.
			distributions.append(np.zeros((len(xaxis),3), dtype=float))
		else:
			if n_bootstrap <= 1:
				if transformFunction is not None:
					spinValues = transformFunction(spinValues)								
				try:
					kde = stats.gaussian_kde(spinValues, weights=spinWeights)
					savedValues = kde(xaxis)
					if normalize:
						savedValues /= np.max(savedValues)
					distributions.append(savedValues)
				except:
					savedValues = np.zeros_like(xaxis)
					if len(spinValues) == 0:
						#There is simply nothing here.  Pass, leaving all values at 0.
						pass
					elif np.all(spinValues == spinValues[0]):
						#Errors will happen when all of the values are the same.  In that case, I'm just making a delta function in the right bin.
						appropriateBin = int(np.floor((spinValues[0]-xaxis[0])/(xaxis[-1]-xaxis[0])))
						if normalize:
							savedValues[appropriateBin] = 1.0
						else:
							savedValues[appropriateBin] = 1/np.diff(xaxis)[0]  #Assuming equal bin spacing to integrate to one.
					distribution.append(savedValues)
			else:
				print("Bootstrapping.")
				storedBootstraps = np.zeros((len(xaxis),n_bootstrap))
				for n_boot in range(n_bootstrap):
					#Random subsample with replacement...
					randomIndices = np.random.randint(0, nHalos, nHalos)
					included = np.in1d(treeIndices, randomIndices)
					bootstrapSpins = spinValues[included]
					bootstrapWeights = spinWeights[included]
					selectedTreeIndices = treeIndices[included]
					for index in np.unique(randomIndices):
						#Multiply weights by the number of times they are selected.
						bootstrapWeights[selectedTreeIndices==index] *= np.count_nonzero(randomIndices==index)

					#Make a KDE
					
					#Optionally, you can provide a function to transform the spins into something else.
					#This can be useful for models where models tend to pile up at 0.998, for example.
					if transformFunction is not None:
						bootstrapSpins = transformFunction(bootstrapSpins)

					try:
						kde = stats.gaussian_kde(bootstrapSpins, weights=bootstrapWeights)
						#Store this instance.
						storedBootstraps[:,n_boot] = kde(xaxis)
						if normalize:
							storedBootstraps[:,n_boot] /= np.max(storedBootstraps[:,n_boot])
					except:
						if len(bootstrapSpins) == 0:
							#There is simply nothing here.  Pass, leaving all values at 0.
							pass
						elif np.all(bootstrapSpins == bootstrapSpins[0]):
							#Errors will happen when all of the values are the same.  In that case, I'm just making a delta function in the right bin.
							appropriateBin = int(np.floor((bootstrapSpins[0]-xaxis[0])/(xaxis[-1]-xaxis[0])))
							if normalize:
								storedBootstraps[appropriateBin,n_boot] = 1.0
							else:
								storedBootstraps[appropriateBin,n_boot] = 1/np.diff(xaxis)[0]  #Assuming equal bin spacing to integrate to one.

				#Once pieces are in place, take percentiles.
				distributions.append(np.percentile(storedBootstraps, [16,50,84], axis=1).transpose())

	if output is None:
		return xaxis, distributions
	else:
		D = {}
		D['bins'] = xaxis
		D['distributions'] = distributions
		D['ensemble'] = ensemble
		D['redshifts'] = redshifts
		with open(output, 'wb') as openFile:
			pickle.dump(D, openFile, protocol=2)

def plotSpinDistribution(xaxis, probabilityDistribution, label=None, color=None, fig_ax=None, figsize=(5,4), doFormatting=True, fontsize=10,\
	xlabel=None, ylabel=None, alpha=0.7, xlim=None, ylim=None, show=True, output=None):
	"""
	Starting with the output of computeSpinDistribution(), make a plot.
	"""

	if fig_ax is not None:
		fig, ax = fig_ax
	else:
		fig, ax = plt.subplots(1, 1, figsize=figsize)

	ax.fill_between(xaxis, probabilityDistribution[:,0], probabilityDistribution[:,2], alpha=alpha, color=color, label=label)

	if doFormatting:
		if label is not None:
			ax.legend(frameon=False)
		ax.set_xlabel(xlabel, fontsize=fontsize)
		ax.set_ylabel(ylabel, fontsize=fontsize)
		ax.set_xlim(xlim)
		ax.set_ylim(ylim)
		fig.tight_layout()

	if show:
		fig.show()
	if output is not None:
		fig.savefig(output, dpi=400)

def plotSpinDistributionGrid(listOfEnsembles, listOfRedshifts=[0], listOfLabels=None, listOfColors=None, figsize=(8,4), xaxis=np.linspace(0,1,1000), figshape=None, show=True, output=None, \
	Mbh_range=[0,np.inf], Mhalo_range=[0,np.inf], Mstar_range=[0,np.inf], Lbol_range=[0,np.inf], fEdd_range=[0,np.inf], treeStartingRedshift=0, transformFunction=None, n_bootstrap=10000, \
	xlim=None, ylim=None, xlabel=r"$a_\bullet$", ylabel="Probability", fontsize=11):

	if figshape is None:
		figshape = (1, len(listOfRedshifts))
	fig, axarr = plt.subplots(figshape[0], figshape[1], figsize=figsize)
	axarr = np.atleast_1d(axarr)

	for e_index in range(len(listOfEnsembles)):
		grove = listOfEnsembles[e_index]
		if listOfLabels is not None:
			label = listOfLabels[e_index]
		else:
			label = None
		if listOfColors is not None:
			color = listOfColors[e_index]
		else:
			color = None
		for z_index in range(len(listOfRedshifts)):
			ax = axarr[z_index]
			redshift = listOfRedshifts[z_index]

			#The real computation and plotting
			if grove.split('/')[-1].split('.')[-1] == 'pkl':
				#This is a pre-computed pickle file.  Open it and extract the information.
				with open(grove, 'rb') as openFile:
					D = pickle.load(openFile)
				xaxis = D['bins']
				allDistributions = D['distributions']
				pickleIndex = np.argmin(np.abs(np.array(D['redshifts'])-redshift))
				if np.abs(D['redshifts'][pickleIndex]-redshift) > 0.1:
					print("You asked for z={redshift}, but we're actually plotting z={D['redshifts'][pickleIndex]} from {grove}.")
				distribution = allDistributions[pickleIndex]
			else:
				#This is a folder leading to the ensembles themselves.  Compute from scratch.  This will take a while.
				xaxis, distribution = computeSpinDistribution(grove, redshift, Mbh_range=Mbh_range, Mhalo_range=Mhalo_range, Mstar_range=Mstar_range, Lbol_range=Lbol_range, fEdd_range=fEdd_range, \
				treeStartingRedshift=treeStartingRedshift, transformFunction=transformFunction, n_bootstrap=n_bootstrap, xaxis=xaxis)
				distribution = distribution[0]
			plotSpinDistribution(xaxis, distribution, fig_ax=(fig,ax), doFormatting=False, color=color, label=label, show=False)

	for z_index in range(len(listOfRedshifts)):
		#Formatting
		ax = axarr[z_index]
		ax.set_xlim(xlim)
		ax.set_ylim(ylim)
		ax.set_xlabel(xlabel, fontsize=fontsize)
		if z_index == 0:
			ax.set_ylabel(ylabel, fontsize=fontsize)
			if listOfLabels is not None:
				ax.legend(frameon=False, fontsize=fontsize)
		else:
			ax.set_yticklabels([])
		ax.text(0.05, 0.95, f"z={listOfRedshifts[z_index]:1.1f}", fontsize=fontsize, ha='left', va='top', transform=ax.transAxes)
	
	fig.tight_layout()
	if show:
		fig.show()
	if output is not None:
		fig.savefig(output, dpi=400)
