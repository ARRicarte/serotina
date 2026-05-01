"""
ARR: 05.14.24

Compute and plot jet power distributions.
Assumes MAD formulae of RNC23.
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
from scipy.interpolate import griddata
import matplotlib.patches as patches
import warnings
currentPath = os.path.abspath(os.path.dirname(__file__)) + '/'

def r_isco(spin):
	"""
	Innermost stable circular orbit, bearing in mind prograde vs. retrograde.
	"""

	Z1 = 1.0 + (1.0 - spin**2)**(1.0/3.0) * ((1.0 + spin)**(1.0/3.0) + (1.0 - spin)**(1.0/3.0))
	Z2 = np.sqrt(3*spin**2 + Z1**2)
	r = 3 + Z2 - np.sign(spin)*np.sqrt((3.0 - Z1) * (3.0 + Z1 + 2*Z2))  #Note the addition of sign(spin) that's not in Shapiro's paper.
	return r

def radiativeEfficiencyThin(spin):
	"""
	Used to define the Eddington ratio.
	"""

	return 1.0 - np.sqrt(1.0 - 2.0/3.0/r_isco(spin))

def calcEddingtonAccretionRate(mass, spin):
	"""
	Returns the Eddington limit of accretion in units of solar masses per year.
	"""

	efficiency = radiativeEfficiencyThin(spin)
	return 4 * np.pi * constants.G * mass * constants.m_p / (efficiency * constants.sigma_T * constants.c) * constants.yr

def spinToHorizon(a):
	"""
	Computes the horizon radius for a given spin value.
	"""

	return 1.0 + np.sqrt(1.0 - a**2)

def Omega_H(spin):
	"""
	Rotation velocity of hole.
	"""
	return np.abs(spin) / (2*spinToHorizon(spin))

def jetEfficiencyFunction(phi, spin, kappa=0.05):
	"""
	Formula from Tchekhovskoy et al. (2010)
	"""

	horizonAngularVelocity = Omega_H(spin)
	return kappa / (4*np.pi) * phi**2 * horizonAngularVelocity**2 * (1. + 1.38*horizonAngularVelocity**2 - 9.2*horizonAngularVelocity**4)

def calcMaximumMagnetization(spin):
	"""
	Fitting function we made up in Narayan et al. 2022
	"""

	return -20.2*spin**3 - 14.9*spin**2 + 34*spin + 52.6

def calcMagnetization(spin, eddingtonRatio, parameters=[1.29, 1.88], fEdd_ADAF_threshold=3e-3):
	"""
	Newly introduced model function.
	"""

	output = calcMaximumMagnetization(spin)
	nonHot = eddingtonRatio > fEdd_ADAF_threshold
	if np.any(nonHot):
		evolutionFactor = (eddingtonRatio[nonHot]/parameters[1])**parameters[0]
		output[nonHot] *= evolutionFactor / (1.0 + evolutionFactor)
	return output

def etaEM(spin, eddingtonRatio, fEdd_ADAF_threshold=3e-3):
	"""
	Assume models for magnetization and jet efficiency.
	"""

	return jetEfficiencyFunction(calcMagnetization(spin, eddingtonRatio, fEdd_ADAF_threshold=fEdd_ADAF_threshold), spin)

def calcJetPower(mass, spin, eddingtonRatio, fEdd_ADAF_threshold=3e-3):
	"""
	Use a fitting function for phi and connect to BZ.
	"""

	#Simply eta * Mdot * c^2
	return etaEM(spin, eddingtonRatio, fEdd_ADAF_threshold=fEdd_ADAF_threshold) * eddingtonRatio * calcEddingtonAccretionRate(mass, spin) * constants.c**2 * constants.M_sun / constants.yr

def computeJetPowerDistribution(ensemble, redshifts, Mbh_range=[0,np.inf], Mhalo_range=[0,np.inf], Mstar_range=[0,np.inf], Lbol_range=[0,np.inf], fEdd_range=[0,np.inf], \
	treeStartingRedshift=0, n_bootstrap=10000, xaxis=np.linspace(40,47,1000), normalize=True, compute_kde=True, output=None, bins=np.linspace(40,47,15), efficiency=False, fEdd_ADAF_threshold=3e-3):
	"""
	Read a grove, then produce a Gaussian KDE function of spins.
	Weighted by halo number densities.
	Some filtering can be applied.
	"""

	if not isinstance(redshifts, list):
		redshifts = [redshifts]

	jetPowerList = [[] for z in redshifts]
	weightList =[[] for z in redshifts]
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
			spins = megaDict['spin_bh'][mask]
			eddingtonRatios = megaDict['eddRatio'][mask]
			masses = megaDict['m_bh'][mask]
			if efficiency:
				jetPowerList[z_index].extend(np.log10(etaEM(spins, eddingtonRatios)))
			else:
				jetPowerList[z_index].extend(np.log10(calcJetPower(masses, spins, eddingtonRatios, fEdd_ADAF_threshold=fEdd_ADAF_threshold) / constants.erg))
			weightList[z_index].extend(np.full(np.sum(mask), sf.calcHaloNumberDensity(hostHaloMass, treeStartingRedshift)))
			treeIndexList[z_index].extend(megaDict['treeIndex'][mask])

	distributions = []

	for z_index in range(len(redshifts)):
		#We now have values and associated weights
		jetPowers = np.array(jetPowerList[z_index])
		#Should deal with 0 jet power.
		jetPowers[~np.isfinite(jetPowers)] = 1e-99
		weights = np.array(weightList[z_index])
		treeIndices = np.array(treeIndexList[z_index])

		if len(jetPowers) == 0:
			#This will happen if there was nothing passing the masking criteria.
			if compute_kde:
				distributions.append(np.full((len(xaxis),3), np.nan))
			else:
				distributions.append(np.full((len(bins)-1,3), np.nan))
		else:
			if n_bootstrap <= 1:
				try:
					kde = stats.gaussian_kde(jetPowers, weights=weights)
					savedValues = kde(xaxis)
					if normalize:
						savedValues /= np.max(savedValues)
					distributions.append(savedValues)
				except:
					savedValues = np.zeros_like(xaxis)
					if len(jetPowers) == 0:
						#There is simply nothing here.  Pass, leaving all values at 0.
						pass
					elif np.all(jetPowers == jetPowers[0]):
						#Errors will happen when all of the values are the same.  In that case, I'm just making a delta function in the right bin.
						appropriateBin = int(np.floor((jetPowers[0]-xaxis[0])/(xaxis[-1]-xaxis[0])))
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
					bootstrapJets = jetPowers[included]
					bootstrapWeights = weights[included]
					selectedTreeIndices = treeIndices[included]
					for index in np.unique(randomIndices):
						#Multiply weights by the number of times they are selected.
						bootstrapWeights[selectedTreeIndices==index] *= np.count_nonzero(randomIndices==index)

					try:
						if compute_kde:
							kde = stats.gaussian_kde(bootstrapJets, weights=bootstrapWeights)
							#Store this instance.
							storedBootstraps[:,n_boot] = kde(xaxis)
						else:
							storedBootstraps[:,n_boot] = np.histogram(bootstrapJets, bins=bins, weights=bootstrapWeights, density=False)[0]
						if normalize:
							storedBootstraps[:,n_boot] /= np.max(storedBootstraps[:,n_boot])
					except:
						if len(bootstrapJets) == 0:
							#There is simply nothing here.  Set values to nan.
							storedBootstraps[:,n_boot] = np.nan
						elif np.all(bootstrapJets == bootstrapJets[0]):
							#Errors will happen when all of the values are the same.  In that case, I'm just making a delta function in the right bin.
							if compute_kde:
								appropriateBin = int(np.floor((bootstrapJets[0]-xaxis[0])/(xaxis[-1]-xaxis[0])))
								if normalize:
									storedBootstraps[appropriateBin,n_boot] = 1.0
								else:
									storedBootstraps[appropriateBin,n_boot] = 1/np.diff(xaxis)[0]  #Assuming equal bin spacing to integrate to one.
							else:
								appropriateBin = int(np.floor((bootstrapJets[0]-bins[0])/(bins[-1]-bins[0])))
								if normalize:
									storedBootstraps[appropriateBin,n_boot] = 1.0
								else:
									storedBootstraps[appropriateBin,n_boot] = 1/np.diff(bins)[0]  #Assuming equal bin spacing to integrate to one.

				#Once pieces are in place, take percentiles.
				distributions.append(np.nanpercentile(storedBootstraps, [16,50,84], axis=1).transpose())

	if output is None:
		if compute_kde:
			return xaxis, distributions
		else:
			return bins, distributions
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

def plotJetPowerDistribution(xaxis, probabilityDistribution, label=None, color=None, fig_ax=None, figsize=(5,4), doFormatting=True, fontsize=10,\
	xlabel=None, ylabel=None, alpha=0.7, xlim=None, ylim=None, show=True, output=None, histogram=False, renormalize=False, efficiency=False):
	"""
	Starting with the output of computeJetPowerDistribution(), make a plot.
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
			ax.step(xaxis[:-1], probabilityDistribution[:,1], lw=2, color=color, where='post')
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

def plotJetPowerDistributionGrid(listOfEnsembles, listOfRedshifts=[0], listOfLabels=None, listOfColors=None, figsize=(8,4), xaxis=np.linspace(40,47,1000), bins=np.linspace(40,47,15), figshape=None, show=True, output=None, \
	Mbh_range=[0,np.inf], Mhalo_range=[0,np.inf], Mstar_range=[0,np.inf], Lbol_range=[0,np.inf], fEdd_range=[0,np.inf], treeStartingRedshift=0, n_bootstrap=10000, \
	xlim=None, ylim=None, xlabel=r"$\log_{10}(P_\mathrm{jet} \ [\mathrm{erg}\;\mathrm{s}^{-1}])$", ylabel="$P/P_\mathrm{max}$", fontsize=11, renormalize=False, compute_kde=True, efficiency=False, fEdd_ADAF_threshold=3e-3):

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
				xaxis, distribution = computeJetPowerDistribution(grove, redshift, Mbh_range=Mbh_range, Mhalo_range=Mhalo_range, Mstar_range=Mstar_range, Lbol_range=Lbol_range, fEdd_range=fEdd_range, \
				treeStartingRedshift=treeStartingRedshift, n_bootstrap=n_bootstrap, xaxis=xaxis, compute_kde=compute_kde, bins=bins, efficiency=efficiency, fEdd_ADAF_threshold=fEdd_ADAF_threshold)
				distribution = distribution[0]
				histogram = not compute_kde
			plotJetPowerDistribution(xaxis, distribution, fig_ax=(fig,ax), doFormatting=False, color=color, label=label, show=False, histogram=histogram, renormalize=renormalize)

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
	if output is not None:
		fig.savefig(output, dpi=400)
	else:
		fig.show()

def plotRadioLoudness(ensemble, redshifts, Mbh_range=[0,np.inf], Mhalo_range=[0,np.inf], Mstar_range=[0,np.inf], Lbol_range=[0,np.inf], fEdd_range=[0,np.inf], fig_axarr=None, figsize=(8,3.2), label=None, color=None, cmap=None, \
	xlim=(1e-5,1e1), ylim=(8e-4,1e1), output=None, minimumEfficiency=1e-3, formatting=True, scatter=True, nbin=30, contoursEnclose=[0.9,0.99], treeStartingRedshift=0):
	"""
	Assume that radio loudness is related to jet efficiency vs. Eddington ratio.
	"""

	if fig_axarr is None:
		fig, axarr = plt.subplots(1, len(redshifts), figsize=figsize)
	else:
		fig, axarr = fig_axarr
	axarr = np.atleast_1d(axarr)

	if not isinstance(redshifts, list):
		redshifts = [redshifts]

	eddingtonRatios = [np.array([]) for z in redshifts]
	efficiencies = [np.array([]) for z in redshifts]
	weightList = [np.array([]) for z in redshifts]
	
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
			spins = megaDict['spin_bh'][mask]
			eddingtonRatios[z_index] = np.concatenate([eddingtonRatios[z_index], megaDict['eddRatio'][mask]])
			efficiencies[z_index] = np.concatenate([efficiencies[z_index], etaEM(spins, megaDict['eddRatio'][mask])])
			weightList[z_index] = np.concatenate([weightList[z_index], np.full(np.sum(mask), sf.calcHaloNumberDensity(hostHaloMass, treeStartingRedshift))])

	for z_index in range(len(redshifts)):
		if scatter:
			f = eddingtonRatios[z_index]
			eta = efficiencies[z_index]
			inefficient = eta < minimumEfficiency
			axarr[z_index].scatter(f[~inefficient], efficiencies[~inefficient], color=color, label=label, marker='o', s=20, alpha=0.7)	
			axarr[z_index].scatter(f[inefficient], [minimumEfficiency]*np.sum(inefficient), color=color, marker='v', s=20, alpha=0.7)
		else:
			f = eddingtonRatios[z_index]
			eta = efficiencies[z_index]
			eta[eta<minimumEfficiency] = minimumEfficiency
			weights = weightList[z_index]

			binnedData, xedges, yedges = np.histogram2d(f, eta, bins=[np.logspace(np.log10(xlim[0]),np.log10(xlim[1]),nbin+1),np.logspace(np.log10(ylim[0]),np.log10(ylim[1]),nbin+1)], density=False, weights=weights)
			Zflat = binnedData.flatten()
			Zflat.sort()
			Zflat = np.flipud(Zflat)
			totalZ = np.sum(Zflat)
			cumsumZ = np.cumsum(Zflat)
			levelsf = np.flipud(np.interp(np.concatenate([[0],contoursEnclose]), cumsumZ / totalZ, Zflat))
			levels = np.flipud(np.interp(contoursEnclose, cumsumZ / totalZ, Zflat))
			axarr[z_index].contourf(0.5*(xedges[1:]+xedges[:-1]), 0.5*(yedges[1:]+yedges[:-1]), binnedData.transpose(), colors=color, cmap=cmap, levels=levelsf)
			axarr[z_index].contour(0.5*(xedges[1:]+xedges[:-1]), 0.5*(yedges[1:]+yedges[:-1]), binnedData.transpose(), colors='k', levels=levels, linewidths=1.5)

	if formatting:
		for col in range(len(redshifts)):
			ax = axarr[col]
			ax.set_xlim(xlim)
			ax.set_ylim(ylim)
			ax.set_xscale('log')
			ax.set_yscale('log')
			ax.set_xlabel(r"$f_\mathrm{Edd}$", fontsize=12)
			if col == 0:
				ax.set_ylabel(r"$P_\mathrm{Jet}/\dot{M}_\bullet c^2$", fontsize=12)
			else:
				ax.set_yticklabels([])
			if len(redshifts) > 1:
				ax.text(0.97, 0.97, f"$z={redshifts[col]:3.1f}$", fontsize=12, transform=ax.transAxes, ha='right', va='top')
		ax.legend(frameon=False, fontsize=12)
		fig.tight_layout()

	if output is None:
		if formatting:
			fig.show()
	else:
		fig.savefig(output)
		plt.close(fig)
