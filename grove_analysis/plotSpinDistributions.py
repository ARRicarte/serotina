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
	treeStartingRedshift=0, transformFunction=None, n_bootstrap=10000, xaxis=np.linspace(-1,1,1000), normalize=True, compute_kde=True, output=None, bins=np.linspace(-1,1,13)):
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

			filterKeys = ['m_bh', 'm_halo', 'm_star', 'L_bol', 'eddRatio']
			filterRanges = [Mbh_range, Mhalo_range, Mstar_range, Lbol_range, fEdd_range]
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
			if compute_kde:
				distributions.append(np.full((len(xaxis),3), np.nan))
			else:
				distributions.append(np.full((len(bins)-1,3), np.nan))
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
				if compute_kde:
					storedBootstraps = np.zeros((len(xaxis),n_bootstrap))
				else:
					storedBootstraps = np.zeros((len(bins)-1,n_bootstrap))
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

					#Optionally, you can provide a function to transform the spins into something else.
					#This can be useful for models where models tend to pile up at 0.998, for example.
					if transformFunction is not None:
						bootstrapSpins = transformFunction(bootstrapSpins)

					try:
						if compute_kde:
							kde = stats.gaussian_kde(bootstrapSpins, weights=bootstrapWeights)
							#Store this instance.
							storedBootstraps[:,n_boot] = kde(xaxis)
						else:
							storedBootstraps[:,n_boot] = np.histogram(bootstrapSpins, bins=bins, weights=bootstrapWeights, density=True)[0]
						if normalize:
							storedBootstraps[:,n_boot] /= np.max(storedBootstraps[:,n_boot])
					except:
						if len(bootstrapSpins) == 0:
							#There is simply nothing here.  Set values to nan.
							storedBootstraps[:,n_boot] = np.nan
						elif np.all(bootstrapSpins == bootstrapSpins[0]):
							#Errors will happen when all of the values are the same.  In that case, I'm just making a delta function in the right bin.
							if compute_kde:
								appropriateBin = int(np.floor((bootstrapSpins[0]-xaxis[0])/(xaxis[-1]-xaxis[0])))
								if normalize:
									storedBootstraps[appropriateBin,n_boot] = 1.0
								else:
									storedBootstraps[appropriateBin,n_boot] = 1/np.diff(xaxis)[0]  #Assuming equal bin spacing to integrate to one.
							else:
								appropriateBin = int(np.floor((bootstrapSpins[0]-bins[0])/(bins[-1]-bins[0])))
								if normalize:
									storedBootstraps[appropriateBin,n_boot] = 1.0
								else:
									storedBootstraps[appropriateBin,n_boot] = 1/np.diff(bins)[0]  #Assuming equal bin spacing to integrate to one.

				#Once pieces are in place, take percentiles.
				distributions.append(np.nanpercentile(storedBootstraps, [16,50,84], axis=1).transpose())

	if output is None:
		return xaxis, distributions
	else:
		D = {}
		if compute_kde:
			D['bins'] = xaxis
		else:
			D['bins'] = bins
		D['distributions'] = distributions
		D['ensemble'] = ensemble
		D['redshifts'] = redshifts
		D['is_kde'] = compute_kde
		with open(output, 'wb') as openFile:
			pickle.dump(D, openFile, protocol=2)

def plotSpinDistribution(xaxis, probabilityDistribution, label=None, color=None, fig_ax=None, figsize=(5,4), doFormatting=True, fontsize=10,\
	xlabel=None, ylabel=None, alpha=0.7, xlim=None, ylim=None, show=True, output=None, histogram=False, renormalize=False):
	"""
	Starting with the output of computeSpinDistribution(), make a plot.
	"""

	if fig_ax is not None:
		fig, ax = fig_ax
	else:
		fig, ax = plt.subplots(1, 1, figsize=figsize)

	if len(probabilityDistribution.shape) == 2:
		if renormalize:
			probabilityDistribution /= np.max(probabilityDistribution[:,1])
		if histogram:
			ax.bar(xaxis[:-1], probabilityDistribution[:,2]-probabilityDistribution[:,0], bottom=probabilityDistribution[:,0], align='edge', width=np.diff(xaxis), alpha=alpha, color=color, label=label)
			ax.bar(xaxis[:-1], probabilityDistribution[:,1], lw=2, edgecolor=color, align='edge', facecolor='none', width=np.diff(xaxis))
		else:
			ax.fill_between(xaxis, probabilityDistribution[:,0], probabilityDistribution[:,2], alpha=alpha, color=color, label=label)
	elif len(probabilityDistribution.shape) == 1:
		if renormalize:
			probabilityDistribution /= np.max(probabilityDistribution)
		if histogram:
			ax.bar(xaxis[:-1], probabilityDistribution, lw=2, color=color, label=label, align='edge', facecolor='none')
		else:
			ax.plot(xaxis, probabilityDistribution, lw=2, color=color, label=label)

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
		plt.close(fig)

def plotSpinDistributionGrid(listOfEnsembles, listOfRedshifts=[0], listOfLabels=None, listOfColors=None, figsize=(8,4), xaxis=np.linspace(0,1,1000), figshape=None, show=True, output=None, \
	Mbh_range=[0,np.inf], Mhalo_range=[0,np.inf], Mstar_range=[0,np.inf], Lbol_range=[0,np.inf], fEdd_range=[0,np.inf], treeStartingRedshift=0, transformFunction=None, n_bootstrap=10000, \
	xlim=None, ylim=None, xlabel=r"$a_\bullet$", ylabel="$P/P_\mathrm{max}$", fontsize=11, renormalize=False, compute_kde=True):

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
				histogram = not D['is_kde']
			else:
				#This is a folder leading to the ensembles themselves.  Compute from scratch.  This will take a while.
				xaxis, distribution = computeSpinDistribution(grove, redshift, Mbh_range=Mbh_range, Mhalo_range=Mhalo_range, Mstar_range=Mstar_range, Lbol_range=Lbol_range, fEdd_range=fEdd_range, \
				treeStartingRedshift=treeStartingRedshift, transformFunction=transformFunction, n_bootstrap=n_bootstrap, xaxis=xaxis, compute_kde=compute_kde)
				distribution = distribution[0]
				histogram = not compute_kde
			plotSpinDistribution(xaxis, distribution, fig_ax=(fig,ax), doFormatting=False, color=color, label=label, show=False, histogram=histogram, renormalize=renormalize)

	for z_index in range(len(listOfRedshifts)):
		#Formatting
		ax = axarr[z_index]
		ax.set_xlim(xlim)
		ax.set_ylim(ylim)
		ax.set_xlabel(xlabel, fontsize=fontsize)
		ax.plot([0,0], ylim, lw=1, ls=':', zorder=-1, color='k')
		if z_index == 0:
			ax.set_ylabel(ylabel, fontsize=fontsize)
			if listOfLabels is not None:
				ax.legend(frameon=False, fontsize=fontsize)
		else:
			ax.set_yticklabels([])
		ax.text(0.05, 0.95, f"z={listOfRedshifts[z_index]:1.1f}", fontsize=fontsize, ha='left', va='top', transform=ax.transAxes)
	
	fig.tight_layout()
	if output is not None:
		fig.savefig(output, dpi=400)
	else:
		fig.show()
