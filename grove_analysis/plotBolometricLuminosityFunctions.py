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
from ..helpers.calcLimitingLuminosities import *
from ..helpers import z6_lf
from ..cosmology import cosmology_functions as cf
from ..helpers import absorptionFractions as af
from scipy.integrate import simps
import matplotlib.patches as patches
currentPath = os.path.abspath(os.path.dirname(__file__)) + '/'

def calcBolometricLuminosityFunctions(ensemble, redshiftSlices=[0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], \
	n_mass=20, logRange=(11,15), n_sample=15, n_bootstrap=10000, plottedPercentiles=[14,86], \
	logAGNLumBins=np.linspace(6,15,41), z_host=0):

	"""
	Create unconvolved luminosity functions
	"""

	#Storage for reading
	agn_luminosities = [np.array([]) for dummy in range(len(redshiftSlices))]
	agn_weights = [np.array([]) for dummy in range(len(redshiftSlices))]
	agn_fileIndices = [np.array([]) for dummy in range(len(redshiftSlices))]

	samplingFactor = float(n_mass) / (logRange[1] - logRange[0])
	ensemble_files = [file for file in os.listdir(ensemble) if file[-5:]=='.pklz']
	print("Reading from {0}".format(ensemble))
	
	for f_index in range(len(ensemble_files)):
		file = ensemble_files[f_index]
		hostHaloMass = 10**float(file.split('_')[-1].split('m')[1].split('n')[0])
		nHalos = int(file.split('_')[-1].split('n')[1].split('.')[0])
	
		#Unpack data
		with gzip.open(ensemble+'/'+file, 'rb') as myfile:
			megaDict = pickle.load(myfile)
		uniqueRedshifts = np.unique(megaDict['redshift'])

		luminousMask = (megaDict['L_bol'] > 0)
		for z_index in range(len(redshiftSlices)):
			closestRedshift = uniqueRedshifts[np.argmin(np.abs(uniqueRedshifts - redshiftSlices[z_index]))]
			if np.abs(closestRedshift - redshiftSlices[z_index]) > 0.1:
				print("Warning:  Wanted z={0}, but we're using z={1}.".format(redshiftSlices[z_index], closestRedshift))
			redshiftMask = (megaDict['redshift'] == closestRedshift)
			combinedMask = redshiftMask & luminousMask
			if np.any(combinedMask):
				#Weight by z=0 abundance
				weight = sf.calcNumberDensity(hostHaloMass, z_host) / nHalos / samplingFactor

				agn_weights[z_index] = np.concatenate((agn_weights[z_index], np.full(np.sum(combinedMask), weight)))
				agn_luminosities[z_index] = np.concatenate((agn_luminosities[z_index], megaDict['L_bol'][combinedMask]))
				agn_fileIndices[z_index] = np.concatenate((agn_fileIndices[z_index], megaDict['treeIndex'][combinedMask]))

	#Storage for bootstrapping
	agnLumFunctionPieces = np.zeros((len(logAGNLumBins)-1,len(redshiftSlices),n_sample))
	agnLumFunctions = np.zeros((len(logAGNLumBins)-1,len(redshiftSlices),n_bootstrap))
	agnLumFunctionRange = np.zeros((len(logAGNLumBins)-1,len(redshiftSlices),2))

	#Compute the luminosity function from this 2D histogram
	for z_index in range(len(redshiftSlices)):
		print("z = {0}:".format(redshiftSlices[z_index]))
		closestRedshift = uniqueRedshifts[np.argmin(np.abs(uniqueRedshifts - redshiftSlices[z_index]))]

		#Collect the pieces for bootstrapping
		print("   Creating bootstrap pieces.")
		for treeIndex in range(n_sample):
			agnLumFunctionPieces[:,z_index,treeIndex] = np.histogram(np.log10(agn_luminosities[z_index][agn_fileIndices[z_index]==treeIndex+1]), bins=logAGNLumBins, \
			weights=agn_weights[z_index][agn_fileIndices[z_index]==treeIndex+1]/np.diff(logAGNLumBins)[0])[0]

		#Bootstrapping!
		print("   Computing luminosity function by bootstrapping.  Number of samples = {0}.".format(n_bootstrap))
		for boot in range(n_bootstrap):
			randomFileNumbers = np.random.randint(0, n_sample, n_sample)
			agnLumFunctions[:,z_index,boot] = np.sum(agnLumFunctionPieces[:,z_index,randomFileNumbers], axis=1)

		#Finally, find the central 68% of all values.
		agnLumFunctionRange[:,z_index,:] = np.transpose(np.percentile(agnLumFunctions[:,z_index,:], plottedPercentiles, axis=1))

	return logAGNLumBins, agnLumFunctionRange

def plotLuminosityFunctions(lumFuncts, labels=None, colors=None, redshiftSlices=[0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], figsize=(8,8), \
	figShape=None, plotConvolved=True, plotUnconvolved=True, numberOfDexToConvolve=0.4, xlim=(1e9,1e15), ylim=(1e-9,1e0), output=None, includeUeda=True, \
	showCompleteness=False, showLynx=True, showOnoue=True, showAxis=False, completenessFile='./bh_completeness_50_convolved0.3.pkl'):
	"""
	Plot the output of the previous function
	"""

	#Initialize figure
	if figShape is None:
		if len(redshiftSlices) >= 3:
			figShape = (np.ceil(float(len(redshiftSlices))/3).astype(int), 3)
		else:
			figShape = (1,len(redshiftSlices))
	fig, axarr = plt.subplots(figShape[0], figShape[1], figsize=figsize, sharex=False, sharey=True)

	#Annoyingly, if figShape[i] == 1, then axarr doesn't have enough dimensions.
	if not hasattr(axarr, '__len__'):
		axarr = np.array([axarr])
	if figShape[0] == 1:
		axarr = axarr.reshape(1, len(axarr))
	elif figShape[1] == 1:
		axarr = axarr.reshape(len(axarr), 1)

	#Read in Hopkins data
	with open(currentPath + '../lookup_tables/bh_data/bol_lf_point_dump.dat', 'r') as myfile:
		hopkinsDataPoints = np.loadtxt(myfile)
	z_hop = hopkinsDataPoints[:,0]
	unique_z_hop = np.unique(z_hop)

	with open(currentPath + '../lookup_tables/bh_data/ueda14/ueda14.pkl', 'rb') as myfile:
		uedaRanges = pickle.load(myfile, encoding='latin1')
	z_ueda = uedaRanges['redshift']
	lum_ueda = uedaRanges['luminosity']
	numberDensity_ueda = uedaRanges['luminosityFunction']

	if showCompleteness:
		with open(completenessFile, 'rb') as myfile:
			completenessTable = pickle.load(myfile, encoding='latin1')

	if showLynx:
		lynxLimit = calcLynxLimit(redshiftSlices)

	if showAxis:
		axisLimit = calcAxisLimit(redshiftSlices)

	for e_index in range(len(lumFuncts)):
		logAGNLumBins, agnLumFunctionRange = lumFuncts[e_index]
		for z_index in range(len(redshiftSlices)):
			i = int(z_index / figShape[1])
			j = z_index % figShape[1]
			xaxis = 10**(0.5*(logAGNLumBins[:-1]+logAGNLumBins[1:]))
			if plotUnconvolved:
				axarr[i,j].fill_between(xaxis, agnLumFunctionRange[:,z_index,0], agnLumFunctionRange[:,z_index,1], label=labels[e_index]+' Unc.', \
				edgecolor='k', alpha=1.0, hatch='//', facecolor='none')
			if plotConvolved:
				nBinsToConvolve = numberOfDexToConvolve / np.diff(logAGNLumBins)[0]
				kernel = sf.makeGaussianSmoothingKernel(nBinsToConvolve)
				convolvedBottom = np.convolve(agnLumFunctionRange[:,z_index,0], kernel, mode='same')
				convolvedTop = np.convolve(agnLumFunctionRange[:,z_index,1], kernel, mode='same')
				axarr[i,j].fill_between(xaxis, convolvedBottom, convolvedTop, label=labels[e_index], \
				color=colors[e_index], alpha=0.7)

			#If this is the last time around, but up the Hopkins data points and do some formatting.
			if e_index == len(lumFuncts)-1:
				closest_z = unique_z_hop[np.argmin(np.abs(unique_z_hop-redshiftSlices[z_index]))]
				if np.abs(closest_z - redshiftSlices[z_index]) < 0.1:
					x_hop = 10**hopkinsDataPoints[z_hop==closest_z,1]
					y_hop = 10**hopkinsDataPoints[z_hop==closest_z,2]/cosmology.h**3
					err_hop = [y_hop*(1.0-10**(-hopkinsDataPoints[z_hop==closest_z,3])), y_hop*(10**hopkinsDataPoints[z_hop==closest_z,3] - 1.0)]
					axarr[i,j].errorbar(x_hop, y_hop, yerr=err_hop, ecolor='k', fmt='none', color='k', capsize=2)
				axarr[i,j].errorbar([0], [0], yerr=[0], ecolor='k', fmt='none', color='k', capsize=2, label='H07')
				
				if (includeUeda) & (redshiftSlices[z_index] <= 5):
					#plottedUedaLums = np.logspace(8,16,100)
					#axarr[i,j].plot(plottedUedaLums, sf.lumFunctUeda(plottedUedaLums, redshiftSlices[z_index]), lw=2, color='slategrey')
					closest_z_ueda_index = np.argmin(np.abs(redshiftSlices[z_index]-z_ueda))
					axarr[i,j].fill_between(lum_ueda, numberDensity_ueda[closest_z_ueda_index,:,0]/cosmology.h**3, numberDensity_ueda[closest_z_ueda_index,:,1]/cosmology.h**3, \
					color='k', alpha=0.7, hatch='/')
					axarr[i,j].fill_between([], [], [], lw=2, color='k', label='U14', alpha=0.7)

				if showCompleteness:
					closestRedshiftIndex_completeness = np.argmin(np.abs(np.array(redshiftSlices[z_index]) - completenessTable['redshift']))
					m_max = completenessTable['holeLimits'][closestRedshiftIndex_completeness]
					l_max = sf.eddingtonLum(m_max)
					axarr[i,j].plot([l_max,l_max], [1e-99,1e99], lw=2, ls=':', color='orange')
					if (i==figShape[0]-1) & (j==figShape[1]-1):
						axarr[i,j].text(l_max*1.3, 10**np.average(np.log10(ylim)), 'SAM Incomplete', rotation=90, fontsize=10, color='orange')

				if showLynx:
					axarr[i,j].plot([lynxLimit[z_index],lynxLimit[z_index]], [1e-99,1e99], lw=1, ls='--', color='dodgerblue')
					if (i==figShape[0]-1) & (j==figShape[1]-1):
						axarr[i,j].text(lynxLimit[z_index]*1.3, 10**np.average(np.log10(ylim)-1.5), 'Lynx Limit', rotation=90, fontsize=10, color='dodgerblue')
				if showAxis:
					axarr[i,j].plot([axisLimit[z_index],axisLimit[z_index]], [1e-99,1e99], lw=1, ls='--', color='indigo')
					if (i==figShape[0]-1) & (j==figShape[1]-1):
						axarr[i,j].text(axisLimit[z_index]*1.3, 10**np.average(np.log10(ylim)-1.5), 'AXIS 32Ms', rotation=90, fontsize=10, color='indigo')

				if showOnoue & (6 in redshiftSlices):
					axarr[i,j].fill_between([], [], [], color='k', alpha=0.7, label='O17')
					if redshiftSlices[z_index]==6:
						data_onoue1 = z6_lf.compute_lf_bolometric(alpha=-2.04,Mstar=-25.8,Phistar=4.06e-9)
						data_onoue2 = z6_lf.compute_lf_bolometric(alpha=-1.98,Mstar=-25.7,Phistar=4.53e-9)
						axarr[i,j].fill_between(data_onoue1[0], data_onoue1[1], data_onoue2[1], color='k', alpha=0.7)

				axarr[i,j].text(xlim[1]*3e-3,ylim[1]*1e-1,r'$z = {0}$'.format(redshiftSlices[z_index]), fontsize=12)
				if (i==figShape[0]-1) & (j==0):
					axarr[i,j].legend(frameon=False, fontsize=9, loc='lower left')

	#Loop again for final formatting
	for i in range(axarr.shape[0]):
		for j in range(axarr.shape[1]):
			axarr[i,j].set_xscale('log')
			axarr[i,j].set_yscale('log')
			if j==0:
				axarr[i,j].set_ylabel(r'$dn/d\log L_\bullet \ [h^{3} \, \mathrm{Mpc}^{-3}]$', fontsize=12)
			axarr[i,j].set_xlim(xlim[0],xlim[1])
			axarr[i,j].set_ylim(ylim[0],ylim[1])
			if i==figShape[0]-1:
				axarr[i,j].set_xlabel(r'$L_\bullet \ [L_\odot]$', fontsize=12)
				#X-tick formatting...
				currentTicks = axarr[i,j].get_xticks()
				editedTicks = currentTicks[(currentTicks <= xlim[1]) & (currentTicks >= xlim[0])]
				if j>0:
					editedTicks = editedTicks[editedTicks>xlim[0]]
				axarr[i,j].set_xticks(editedTicks)
			else:
				axarr[i,j].set_xticks([])

	fig.tight_layout()
	fig.subplots_adjust(hspace=0,wspace=0)

	if output is not None:
		fig.savefig(output)
	else:
		fig.show()

def _computeExpectedNumbers(xaxis, yaxis_range):
	"""
	Used to convert ranges of d^N/dlogL/dV into observed numbers with error bars.
	"""

	if len(xaxis) == 0:
		return 0.0, 0.0
	else:
		yaxis_mean = np.mean(yaxis_range, axis=0)
		yaxis_error = 0.5 * np.squeeze(np.diff(yaxis_range, axis=0))
		expectedNumber = simps(yaxis_mean, x=xaxis)
		propagatedError = np.sqrt(simps(yaxis_error**2, x=xaxis))
		return expectedNumber, propagatedError

def plotLynxDetections(lumFuncts, labels=None, colors=None, redshiftSlices=[0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], figsize=(8,8), \
	figShape=None, numberOfDexToConvolve=0.3, xlim=(1e9,1e15), ylim=(1e-2,1e4), output=None, fieldOfView_Lynx=400.0, fieldOfView_Chandra=25.0, integrate=True):

	#Initialize figure
	if figShape is None:
		if len(redshiftSlices) >= 3:
			figShape = (np.ceil(float(len(redshiftSlices))/3).astype(int), 3)
		else:
			figShape = (1,len(redshiftSlices))
	fig, axarr = plt.subplots(figShape[0], figShape[1], figsize=figsize, sharex=False, sharey=False)

	#Annoyingly, if figShape[i] == 1, then axarr doesn't have enough dimensions.
	if not hasattr(axarr, '__len__'):
		axarr = np.array([axarr])
	if figShape[0] == 1:
		axarr = axarr.reshape(1, len(axarr))
	elif figShape[1] == 1:
		axarr = axarr.reshape(len(axarr), 1)

	lynxLimit = calcLynxLimit(redshiftSlices)
	chandraLimit = calcChandraLimit(redshiftSlices)
	fov_correction_Lynx = fieldOfView_Lynx * (np.pi / 180.0 / 60)**2 / (4*np.pi)
	fov_correction_Chandra = fieldOfView_Chandra * (np.pi / 180.0 / 60)**2 / (4*np.pi)

	for e_index in range(len(lumFuncts)):
		logAGNLumBins, agnLumFunctionRange = lumFuncts[e_index]
		if integrate:
			print(labels[e_index]+":")

		for z_index in range(len(redshiftSlices)):
			i = int(z_index / figShape[1])
			j = z_index % figShape[1]

			cosmologyWeighting = cf.computedVdz(redshiftSlices[z_index])

			xaxis = 10**(0.5*(logAGNLumBins[:-1]+logAGNLumBins[1:]))
			nBinsToConvolve = numberOfDexToConvolve / np.diff(logAGNLumBins)[0]
			kernel = sf.makeGaussianSmoothingKernel(nBinsToConvolve)
			convolvedBottom = np.convolve(agnLumFunctionRange[:,z_index,0], kernel, mode='same') * cosmology.h**3
			convolvedTop = np.convolve(agnLumFunctionRange[:,z_index,1], kernel, mode='same') * cosmology.h**3

			#NOTE:  Applying Field of View correction later, since it's different for these two instruments.
			observableFraction = af.typeIProbability(xaxis, redshiftSlices[z_index])
			observedBottom = cosmologyWeighting*observableFraction*convolvedBottom
			observedTop = cosmologyWeighting*observableFraction*convolvedTop
			
			axarr[i,j].fill_between(xaxis, fov_correction_Lynx*observedBottom, fov_correction_Lynx*observedTop, \
			label=labels[e_index], color=colors[e_index], alpha=0.7)

			if integrate:
				observedByChandra = xaxis > chandraLimit[z_index]
				observedByLynx = xaxis > lynxLimit[z_index]
				raw_numbers_cdfs = fov_correction_Chandra * np.array([observedBottom, observedTop])[:,observedByChandra]
				xrange_cdfs = np.log10(xaxis[observedByChandra])
				raw_numbers_lynx = fov_correction_Lynx * np.array([observedBottom, observedTop])[:,observedByLynx]
				xrange_lynx = np.log10(xaxis[observedByLynx])

				number_cdfs, error_cdfs = _computeExpectedNumbers(xrange_cdfs, raw_numbers_cdfs)
				number_lynx, error_lynx = _computeExpectedNumbers(xrange_lynx, raw_numbers_lynx)

				print("   z={0}, N={1} +/- {2} for CDF-S, N={3} +/- {4} for Lynx".format(redshiftSlices[z_index], number_cdfs, error_cdfs, number_lynx, error_lynx))

			#If this is the last time around, do some formatting.
			if e_index == len(lumFuncts)-1:
				axarr[i,j].text(xlim[1]*3e-3,ylim[1]*3e-1,r'$z={0}$'.format(redshiftSlices[z_index]), fontsize=12)
				axarr[i,j].plot([lynxLimit[z_index],lynxLimit[z_index]], [1e-99,1e99], lw=1, ls='--', color='dodgerblue')
				#axarr[i,j].plot([lynxLimit[z_index]*1e2,lynxLimit[z_index]*1e2], [1e-99,1e99], lw=1, ls='-.', color='violet')
				if (i==figShape[0]-1) & (j==figShape[1]-1):
					axarr[i,j].text(lynxLimit[z_index]*1.3, 10**np.average(np.log10(ylim))*5e1, 'Lynx Limit', rotation=90, fontsize=10, color='dodgerblue')
					#axarr[i,j].text(lynxLimit[z_index]*1e2*1.3, 10**np.average(np.log10(ylim))*5e1, 'CDF-S Limit', rotation=90, fontsize=10, color='violet')
				if (i==figShape[0]-1) & (j==figShape[1]-1):
					axarr[i,j].legend(frameon=False, fontsize=9, loc='lower right')

	#Loop again for final formatting
	for i in range(axarr.shape[0]):
		for j in range(axarr.shape[1]):
			axarr[i,j].set_xscale('log')
			axarr[i,j].set_yscale('log')
			if i==figShape[0]-1:
				axarr[i,j].set_xlabel(r'$L_\bullet \ [L_\odot]$', fontsize=12)
			if j==0:
				axarr[i,j].set_ylabel(r'$d^2 N_\mathrm{obs}/d\logL_\bullet dz$', fontsize=12)
			if j>0:
				axarr[i,j].set_xticks(np.logspace(9,15,4))
			if i!=figShape[0]-1:
				axarr[i,j].set_xticks([])
			axarr[i,j].set_xlim(xlim[0],xlim[1])
			axarr[i,j].set_ylim(ylim[0],ylim[1])

	fig.tight_layout()
	fig.subplots_adjust(hspace=0,wspace=0)
	for i in range(axarr.shape[0]):
		for j in range(axarr.shape[1]):
			if j > 0:
				axarr[i,j].set_yticks([])
			elif i != axarr.shape[0]-1:
				currentyTicks = axarr[i,j].get_yticks()
				newyTicks = [tick for tick in currentyTicks if ((tick > ylim[0]) & (tick <= ylim[1]))]
				axarr[i,j].set_yticks(newyTicks)

	if output is not None:
		fig.savefig(output)
	else:
		fig.show()

def plotLynxDetectionsDiscrete(lumFuncts, labels=None, colors=None, redshiftSlices=[0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], figsize=(4,4), \
	numberOfDexToConvolve=0.3, ylim=(0,1e4), output=None, fieldOfView_Lynx=400.0, fieldOfView_Chandra=25.0, spitTable=False, fluxLimit_cgs=1e-19, \
	computeAverageFlux=False):

	#Calculations
	lynxLimit = calcLynxLimit(redshiftSlices, fluxLimit_cgs=fluxLimit_cgs)
	chandraLimit = calcChandraLimit(redshiftSlices)
	fov_correction_Lynx = fieldOfView_Lynx * (np.pi / 180.0 / 60)**2 / (4*np.pi)
	fov_correction_Chandra = fieldOfView_Chandra * (np.pi / 180.0 / 60)**2 / (4*np.pi)
	detectionList_Lynx = [np.zeros((len(redshiftSlices),2)) for dummy in range(len(lumFuncts))]
	detectionList_Chandra = [np.zeros((len(redshiftSlices),2)) for dummy in range(len(lumFuncts))]

	for e_index in range(len(lumFuncts)):
		logAGNLumBins, agnLumFunctionRange = lumFuncts[e_index]
		if not spitTable:
			print(labels[e_index]+":")
		for z_index in range(len(redshiftSlices)):
			cosmologyWeighting = cf.computedVdz(redshiftSlices[z_index])

			xaxis = 10**(0.5*(logAGNLumBins[:-1]+logAGNLumBins[1:]))
			nBinsToConvolve = numberOfDexToConvolve / np.diff(logAGNLumBins)[0]
			kernel = sf.makeGaussianSmoothingKernel(nBinsToConvolve)
			convolvedBottom = np.convolve(agnLumFunctionRange[:,z_index,0], kernel, mode='same') * cosmology.h**3
			convolvedTop = np.convolve(agnLumFunctionRange[:,z_index,1], kernel, mode='same') * cosmology.h**3

			#NOTE:  Applying Field of View correction later, since it's different for these two instruments.
			observableFraction = af.typeIProbability(xaxis, redshiftSlices[z_index])
			observedBottom = cosmologyWeighting*observableFraction*convolvedBottom
			observedTop = cosmologyWeighting*observableFraction*convolvedTop

			observedByChandra = xaxis > chandraLimit[z_index]
			observedByLynx = xaxis > lynxLimit[z_index]
			raw_numbers_cdfs = fov_correction_Chandra * np.array([observedBottom, observedTop])[:,observedByChandra]
			xrange_cdfs = np.log10(xaxis[observedByChandra])
			raw_numbers_lynx = fov_correction_Lynx * np.array([observedBottom, observedTop])[:,observedByLynx]
			xrange_lynx = np.log10(xaxis[observedByLynx])

			number_cdfs, error_cdfs = _computeExpectedNumbers(xrange_cdfs, raw_numbers_cdfs)
			number_lynx, error_lynx = _computeExpectedNumbers(xrange_lynx, raw_numbers_lynx)

			detectionList_Lynx[e_index][z_index,:] = np.array([number_lynx-error_lynx,number_lynx+error_lynx])
			detectionList_Chandra[e_index][z_index,:] = np.array([number_cdfs-error_cdfs,number_cdfs+error_cdfs])
			if not spitTable:
				print("   z={0}, N={1} +/- {2} for CDF-S, N={3} +/- {4} for Lynx".format(redshiftSlices[z_index], number_cdfs, error_cdfs, number_lynx, error_lynx))

			if computeAverageFlux:
				averageLuminosity = np.sum(xaxis[observedByLynx] * np.mean(raw_numbers_lynx, axis=0)) / np.sum(np.mean(raw_numbers_lynx, axis=0)) * constants.L_sun / constants.erg
				dL = cf.computeLuminosityDistance(redshiftSlices[z_index]) * 1e6 * constants.pc * 1e2
				averageFlux = averageLuminosity / (4 * np.pi * dL**2)
				print("  The average flux at z={0} is {1:3.2e} erg/s/cm^2.".format(redshiftSlices[z_index], averageFlux))

	if spitTable:
		headerString = "Model"
		for z_index in range(len(redshiftSlices)):
			headerString += " & z = {0:2.0f}".format(redshiftSlices[z_index])
		headerString += r" \\"
		print(headerString)
		for e_index in range(len(lumFuncts)):
			outstring = labels[e_index]
			for z_index in range(len(redshiftSlices)):
				outstring += r" & ${0} \pm {1}$".format(np.mean(detectionList_Lynx[e_index][z_index,:]), np.squeeze(np.diff(detectionList_Lynx[e_index][z_index,:])))
			outstring += r" \\"
			print(outstring)

	#Plotting
	fig, ax = plt.subplots(1, 1, figsize=figsize)

	for e_index in range(len(lumFuncts)):
		for z_index in range(len(redshiftSlices)): 
			ax.add_patch(patches.Rectangle((redshiftSlices[z_index]-0.5, detectionList_Lynx[e_index][z_index,0]), 1, \
			(detectionList_Lynx[e_index][z_index,1]-detectionList_Lynx[e_index][z_index,0]), color=colors[e_index], alpha=0.7))
		ax.fill_between([], [], [], color=colors[e_index], label=labels[e_index], alpha=0.7)

	ax.legend(frameon=False)
	ax.set_xscale('linear')
	ax.set_yscale('log')
	ax.set_xlabel(r'$z$', fontsize=12)
	ax.set_ylabel(r'Number of AGN in Lynx Deep Field', fontsize=12)
	ax.set_xlim(redshiftSlices[0]-0.5,redshiftSlices[-1]+0.5)
	ax.set_ylim(ylim[0],ylim[1])

	fig.tight_layout()

	if output is not None:
		fig.savefig(output)
	else:
		fig.show()

def plotAxisDetectionsDiscrete(lumFuncts, labels=None, colors=None, redshiftSlices=[0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], figsize=(4,4), \
	numberOfDexToConvolve=0.3, ylim=(0,1e4), output=None, fieldOfView_Axis=1800.0, fieldOfView_Chandra=25.0, spitTable=False, fluxLimit_cgs=1e-19):

	#Calculations
	axisLimit = calcAxisLimit(redshiftSlices, fluxLimit_cgs=fluxLimit_cgs)
	chandraLimit = calcChandraLimit(redshiftSlices)
	fov_correction_Axis = fieldOfView_Axis * (np.pi / 180.0 / 60)**2 / (4*np.pi)
	fov_correction_Chandra = fieldOfView_Chandra * (np.pi / 180.0 / 60)**2 / (4*np.pi)
	detectionList_Axis = [np.zeros((len(redshiftSlices),2)) for dummy in range(len(lumFuncts))]
	detectionList_Chandra = [np.zeros((len(redshiftSlices),2)) for dummy in range(len(lumFuncts))]

	for e_index in range(len(lumFuncts)):
		logAGNLumBins, agnLumFunctionRange = lumFuncts[e_index]
		if not spitTable:
			print(labels[e_index]+":")
		for z_index in range(len(redshiftSlices)):
			cosmologyWeighting = cf.computedVdz(redshiftSlices[z_index])

			xaxis = 10**(0.5*(logAGNLumBins[:-1]+logAGNLumBins[1:]))
			nBinsToConvolve = numberOfDexToConvolve / np.diff(logAGNLumBins)[0]
			kernel = sf.makeGaussianSmoothingKernel(nBinsToConvolve)
			convolvedBottom = np.convolve(agnLumFunctionRange[:,z_index,0], kernel, mode='same') * cosmology.h**3
			convolvedTop = np.convolve(agnLumFunctionRange[:,z_index,1], kernel, mode='same') * cosmology.h**3

			#NOTE:  Applying Field of View correction later, since it's different for these two instruments.
			observableFraction = af.typeIProbability(xaxis, redshiftSlices[z_index])
			observedBottom = cosmologyWeighting*observableFraction*convolvedBottom
			observedTop = cosmologyWeighting*observableFraction*convolvedTop

			observedByChandra = xaxis > chandraLimit[z_index]
			observedByAxis = xaxis > axisLimit[z_index]
			raw_numbers_cdfs = fov_correction_Chandra * np.array([observedBottom, observedTop])[:,observedByChandra]
			xrange_cdfs = np.log10(xaxis[observedByChandra])
			raw_numbers_axis = fov_correction_Axis * np.array([observedBottom, observedTop])[:,observedByAxis]
			xrange_axis = np.log10(xaxis[observedByAxis])

			number_cdfs, error_cdfs = _computeExpectedNumbers(xrange_cdfs, raw_numbers_cdfs)
			number_axis, error_axis = _computeExpectedNumbers(xrange_axis, raw_numbers_axis)

			detectionList_Axis[e_index][z_index,:] = np.array([number_axis-error_axis,number_axis+error_axis])
			detectionList_Chandra[e_index][z_index,:] = np.array([number_cdfs-error_cdfs,number_cdfs+error_cdfs])
			if not spitTable:
				print("   z={0}, N={1} +/- {2} for CDF-S, N={3} +/- {4} for Axis".format(redshiftSlices[z_index], number_cdfs, error_cdfs, number_axis, error_axis))

	if spitTable:
		headerString = "Model"
		for z_index in range(len(redshiftSlices)):
			headerString += " & z = {0:2.0f}".format(redshiftSlices[z_index])
		headerString += r" \\"
		print(headerString)
		for e_index in range(len(lumFuncts)):
			outstring = labels[e_index]
			for z_index in range(len(redshiftSlices)):
				outstring += r" & ${0} \pm {1}$".format(np.mean(detectionList_Axis[e_index][z_index,:]), np.squeeze(np.diff(detectionList_Axis[e_index][z_index,:])))
			outstring += r" \\"
			print(outstring)

	#Plotting
	fig, ax = plt.subplots(1, 1, figsize=figsize)

	for e_index in range(len(lumFuncts)):
		for z_index in range(len(redshiftSlices)): 
			ax.add_patch(patches.Rectangle((redshiftSlices[z_index]-0.5, detectionList_Axis[e_index][z_index,0]), 1, \
			(detectionList_Axis[e_index][z_index,1]-detectionList_Axis[e_index][z_index,0]), color=colors[e_index], alpha=0.7))
		ax.fill_between([], [], [], color=colors[e_index], label=labels[e_index], alpha=0.7)

	ax.legend(frameon=False)
	ax.set_xscale('linear')
	ax.set_yscale('log')
	ax.set_xlabel(r'$z$', fontsize=12)
	ax.set_ylabel(r'Number of AGN Detected', fontsize=12)
	ax.set_xlim(redshiftSlices[0]-0.5,redshiftSlices[-1]+0.5)
	ax.set_ylim(ylim[0],ylim[1])

	fig.tight_layout()

	if output is not None:
		fig.savefig(output)
	else:
		fig.show()

def saveLuminosityFunctions_pickle(lumFuncts, outputName, labels, redshifts):
	#Just save a pickled dictionary
	dictionary = {'lumFuncts': lumFuncts, 'labels': labels, 'redshifts': redshifts}
	pickle.dump(dictionary, outputName, protocol=2)

def saveLuminosityFunctions_text(lumFuncts, outputName, labels, redshiftSlices, numberOfDexToConvolve=0.3):
	#outputName should be a folder
	if os.path.exists(outputName):
		print("Using existing folder, "+outputName)
	else:
		os.mkdir(outputName)

	#Need to make a 2D table for each model
	headerText = 'z log10(L_bol) Phi_low Phi_high'
	largeTable = np.empty((0,4))
	for e_index in range(len(lumFuncts)):
		log10_L_bol = 0.5*(lumFuncts[e_index][0][:-1] + lumFuncts[e_index][0][1:])
		nBinsToConvolve = numberOfDexToConvolve / np.diff(log10_L_bol)[0]
		kernel = sf.makeGaussianSmoothingKernel(nBinsToConvolve)
		for z_index in range(len(redshiftSlices)):
			convolvedBottom = np.convolve(lumFuncts[e_index][1][:,z_index,0], kernel, mode='same') * cosmology.h**3
			convolvedTop = np.convolve(lumFuncts[e_index][1][:,z_index,1], kernel, mode='same') * cosmology.h**3
			zarray = np.full(len(log10_L_bol), redshiftSlices[z_index])
			
			#Now all the columns exist, so I need to make them a 2d thing.
			newChunk = np.stack((zarray, log10_L_bol, convolvedBottom, convolvedTop), axis=-1)
			largeTable = np.vstack((largeTable, newChunk))

		#Outputting
		textFileName = outputName+'/'+labels[e_index]+'.txt'
		np.savetxt(textFileName, largeTable, header=headerText)

if __name__ == '__main__':
	ensembles = ['powerLaw_popIII_pmerge0.1_072018', 'agnms_popIII_pmerge0.1_072018', 'powerLaw_dcbh_pmerge0.1_072018', 'agnms_dcbh_pmerge0.1_072018']
	labels = ['Light-PL', 'Light-MS', 'Heavy-PL', 'Heavy-MS']
	colors = ['r', 'orange', 'purple', 'c']

	#Uncomment for Hopkins comparison
	redshiftSlices = [0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; xlim=(1e9,1e15); ylim=(1e-9,1e0); figsize=(8,8); plotUnconvolved=False

	#Uncomment to explore Lynx era
	#redshiftSlices = [6.0, 7.0, 8.0, 9.0, 10.0, 12.0]; figsize=(8,5); xlim=(1e7,1e15); ylim=(1e-9,1e0); plotUnconvolved=False

	#Uncomment for convolution comparison
	'''
	ensembles = ['randombumps_popIII_042017']
	labels = ['PopIII']
	colors = ['g']
	redshiftSlices = [0.2]; xlim=(1e9,1e15); ylim=(1e-9,1e0); figsize=(4,4); plotUnconvolved=True
	'''

	lumFuncts = [calcBolometricLuminosityFunctions(e, redshiftSlices=redshiftSlices, n_mass=23, n_sample=20, logRange=(10.6,15.0), z_host=0) for e in ensembles]
	#saveLuminosityFunctions_text(lumFuncts, './data_products/luminosity_functions_to6', labels, redshiftSlices)
	plotLuminosityFunctions(lumFuncts, labels=labels, colors=colors, redshiftSlices=redshiftSlices, plotUnconvolved=plotUnconvolved, figsize=figsize, \
	xlim=xlim, ylim=ylim, showCompleteness=True, numberOfDexToConvolve=0.3, showOnoue=False, showLynx=False, showAxis=False)
	#plotLynxDetections(lumFuncts, labels=labels, colors=colors, redshiftSlices=redshiftSlices, figsize=(8,5), xlim=(1e7,1e15), ylim=(1e0,1e4))
	#plotLynxDetectionsDiscrete(lumFuncts, labels=labels, colors=colors, redshiftSlices=redshiftSlices, figsize=(4,4), ylim=(0,5e3), fluxLimit_cgs=1e-18, fieldOfView_Lynx=9.7, computeAverageFlux=True, spitTable=True)
	#plotAxisDetectionsDiscrete(lumFuncts, labels=labels, colors=colors, redshiftSlices=redshiftSlices, figsize=(4,4), ylim=(0,5e3))
