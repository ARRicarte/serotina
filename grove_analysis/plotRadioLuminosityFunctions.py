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
from .. import util
from ..helpers.calcLimitingLuminosities import *
from ..helpers import z6_lf
from ..cosmology import cosmology_functions as cf
from ..helpers import absorptionFractions as af
from . import bolometricLuminosityFunctionsShen2020 as shen2020
from ..accretion_recipes import spinEvolverMAD as jets
from . import radioLuminosityFunctionsCeraj2018 as ceraj18
import matplotlib.patches as patches
currentPath = os.path.abspath(os.path.dirname(__file__)) + '/'

def jetPowerToLuminosity(power, relation='MH07'):
	"""
	Many relationships in the literature are compiled in Appendix A of Smolcic et al. (2017).  The most extreme are implemented here.

	power should be in W, values returned are W Hz^-1
	"""

	if relation == 'MH07':
		#Merloni & Heinz (2007)
		return 10**((np.log10(power)-37.39)/0.81) * 1e24
	elif relation == 'D12':
		#Daly et al. (2012)
		return 10**((np.log10(power)-35.69)/0.84) * 1e24
	elif relation == 'HB14':
		#Heckman & Best (2014)
		return 10**((np.log10(power)-37.44)/0.68) * 1e25

def calcRadioLuminosityFunctions(ensemble, redshiftSlices=[0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], madness=1.0, \
	n_mass=20, logRange=(11,15), n_sample=15, n_bootstrap=10000, plottedPercentiles=[14,86], \
	logAGNLumBins=np.linspace(21,28,29), z_host=0, relation='MH07', mdot_Edd_hot_thin=3e-3, fEdd_range=(0,np.inf), Lbol_range=(0,np.inf), P_range=(0,np.inf)):

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

		luminousMask = (megaDict['L_bol']*constants.L_sun/constants.erg > Lbol_range[0]) & (megaDict['L_bol']*constants.L_sun/constants.erg <= Lbol_range[1])
		fEddMask = (megaDict['eddRatio'] >= fEdd_range[0]) & (megaDict['eddRatio'] <= fEdd_range[1])
		for z_index in range(len(redshiftSlices)):
			closestRedshift = uniqueRedshifts[np.argmin(np.abs(uniqueRedshifts - redshiftSlices[z_index]))]
			if np.abs(closestRedshift - redshiftSlices[z_index]) > 0.1:
				print("Warning:  Wanted z={0}, but we're using z={1}.".format(redshiftSlices[z_index], closestRedshift))
			redshiftMask = (megaDict['redshift'] == closestRedshift)
			
			combinedMask = redshiftMask & luminousMask & fEddMask
			if np.any(combinedMask):
				#Weight by z=0 abundance
				weight = sf.calcHaloNumberDensity(hostHaloMass, z_host) / nHalos / samplingFactor

				jetPower = []
				for i in np.where(combinedMask)[0]:
					jetPower.append(jets.calcJetPower(megaDict['m_bh'][i], megaDict['spin_bh'][i], megaDict['eddRatio'][i], mdot_Edd_hot_thin=mdot_Edd_hot_thin))
				jetPower = np.array(jetPower)
				jetPower[(jetPower<P_range[0]) | (jetPower>P_range[1])] = 0.0
				agn_weights[z_index] = np.concatenate((agn_weights[z_index], np.full(np.sum(combinedMask), weight)))
				agn_luminosities[z_index] = np.concatenate((agn_luminosities[z_index], jetPowerToLuminosity(jetPower, relation=relation)))
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

	#madness is a duty cycle.  It's just going to rescale everything at the very end.
	return logAGNLumBins, agnLumFunctionRange*madness

def plotRadioLuminosityFunctions(lumFuncts, labels=None, colors=None, redshiftSlices=[0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], figsize=(8,8), showCeraj=True, \
	figShape=None, plotConvolved=True, plotUnconvolved=False, numberOfDexToConvolve=0.4, xlim=(1e19,1e28), ylim=(1e-8,1e-2), output=None, fig_axarr=None):
	"""
	Plot the output of the previous function
	"""

	#Initialize figure
	if fig_axarr is None:
		if figShape is None:
			if len(redshiftSlices) >= 3:
				figShape = (np.ceil(float(len(redshiftSlices))/3).astype(int), 3)
			else:
				figShape = (1,len(redshiftSlices))
		fig, axarr = plt.subplots(figShape[0], figShape[1], figsize=figsize, sharex=False, sharey=True)
	else:
		fig, axarr = fig_axarr

	#Annoyingly, if figShape[i] == 1, then axarr doesn't have enough dimensions.
	if not hasattr(axarr, '__len__'):
		axarr = np.array([axarr])
	if figShape[0] == 1:
		axarr = axarr.reshape(1, len(axarr))
	elif figShape[1] == 1:
		axarr = axarr.reshape(len(axarr), 1)

	for e_index in range(len(lumFuncts)):
		logAGNLumBins, agnLumFunctionRange = lumFuncts[e_index]
		for z_index in range(len(redshiftSlices)):
			i = int(z_index / figShape[1])
			j = z_index % figShape[1]
			xaxis = 10**(0.5*(logAGNLumBins[:-1]+logAGNLumBins[1:]))
			if plotUnconvolved:
				axarr[i,j].fill_between(xaxis, agnLumFunctionRange[:,z_index,0]*cosmology.h**3, agnLumFunctionRange[:,z_index,1]*cosmology.h**3, label=labels[e_index]+' Unc.', \
				edgecolor='k', alpha=1.0, hatch='//', facecolor='none')
			if plotConvolved:
				nBinsToConvolve = numberOfDexToConvolve / np.diff(logAGNLumBins)[0]
				kernel = util.makeGaussianSmoothingKernel(nBinsToConvolve)
				convolvedBottom = np.convolve(agnLumFunctionRange[:,z_index,0]*cosmology.h**3, kernel, mode='same')
				convolvedTop = np.convolve(agnLumFunctionRange[:,z_index,1]*cosmology.h**3, kernel, mode='same')
				axarr[i,j].fill_between(xaxis, convolvedBottom, convolvedTop, label=labels[e_index], \
				color=colors[e_index], alpha=0.7)

			#If this is the last time around, do some formatting.
			if e_index == len(lumFuncts)-1:

				if showCeraj:
					x_ceraj = np.logspace(21,29,100)
					axarr[i,j].plot(x_ceraj, ceraj18.phi(x_ceraj, redshiftSlices[z_index]), ls='-', lw=2, color='k', label='C18', zorder=-1)

				axarr[i,j].text(xlim[1]*3e-3,ylim[1]*1e-1,r'$z = {0}$'.format(redshiftSlices[z_index]), fontsize=12)
				if (i==figShape[0]-1) & (j==0):
					axarr[i,j].legend(frameon=False, fontsize=9, loc='lower left')

	#Loop again for final formatting
	for i in range(axarr.shape[0]):
		for j in range(axarr.shape[1]):
			axarr[i,j].set_xscale('log')
			axarr[i,j].set_yscale('log')
			if j==0:
				axarr[i,j].set_ylabel(r'$dn/d\log L_{\nu,1.4} \ [\mathrm{Mpc}^{-3}\;\mathrm{dex}^{-1}]$', fontsize=12)
			axarr[i,j].set_xlim(xlim[0],xlim[1])
			axarr[i,j].set_ylim(ylim[0],ylim[1])
			if i==figShape[0]-1:
				axarr[i,j].set_xlabel(r'$L_{\nu,1.4} \ [\mathrm{W}\;\mathrm{Hz}^{-1}]$', fontsize=12)
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
		kernel = util.makeGaussianSmoothingKernel(nBinsToConvolve)
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
