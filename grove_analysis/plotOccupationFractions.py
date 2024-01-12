"""
ARR:  05.15.17

Plots the occupation fractions at z=0 for a SAM ensemble.
"""

import numpy as np
import gzip
import os
import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.patches as patches
from ..helpers import sam_functions as sf
from .. import cosmology
from .. import constants
from ..helpers import smhm
from ..helpers import absorptionFractions as af
currentPath = os.path.abspath(os.path.dirname(__file__)) + '/'

def computeOccupationFractions(ensemble, haloMassRange=(10.6,15.1), nbins=23, n_bootstrap=1000, redshifts=[0], \
	weightByMass=False, alphaBeta=(8.45,5.0), takeLog=True, minimumMass=0, includeSatellites=False):
	"""
	Calculate the occupation fractions of the ensemble
	"""
	
	ensemble_files = [file for file in os.listdir(ensemble) if file[-5:]=='.pklz']
	print("Reading from {0}".format(ensemble))

	#Storage
	logStellarMasses = [np.array([]) for z in redshifts]
	blackHoleMasses = [np.array([]) for z in redshifts]
	expectedMasses = [np.array([]) for z in redshifts]

	for f_index in range(len(ensemble_files)):
		file = ensemble_files[f_index]

		#Unpack data
		with gzip.open(ensemble+'/'+file, 'r') as myfile:
			megaDict = pickle.load(myfile)

		redshiftsAvailable = np.unique(megaDict['redshift'])
		for z_index in range(len(redshifts)):
			redshift = redshifts[z_index]
			closestRedshift = redshiftsAvailable[np.argmin(np.abs(redshiftsAvailable-redshift))]
			selection = (megaDict['redshift']==closestRedshift)
			if not includeSatellites:
				selection = selection & (megaDict['satelliteToCentral']==-1)
			logMhalo = np.log10(megaDict['m_halo'][selection])
			blackHoleMasses[z_index] = np.concatenate((blackHoleMasses[z_index], megaDict['m_bh'][selection]))
			logStellarMasses[z_index] = np.concatenate((logStellarMasses[z_index], smhm.logMstar(logMhalo, closestRedshift)))
			expectedMasses[z_index] = np.concatenate((expectedMasses[z_index], 10**(alphaBeta[0] + alphaBeta[1]*np.log10(sf.velocityDispersion(10**logMhalo, closestRedshift)/200))))

	stellarMassBinEdges = smhm.logMstar(np.linspace(haloMassRange[0],haloMassRange[1],nbins+1), 0)

	occupationFractions = [np.zeros((nbins,2)) for z in redshifts]
	for z_index in range(len(redshifts)):
		for i in range(nbins):
			inBin = (stellarMassBinEdges[i] <= logStellarMasses[z_index]) & (logStellarMasses[z_index] < stellarMassBinEdges[i+1])
			numberInBin = np.sum(inBin)		
			if numberInBin == 0:
				occupationFractions[z_index][i,:] = np.nan
				continue
			relevantBlackHoleMasses = blackHoleMasses[z_index][inBin]
			if weightByMass:
				relevantExpectedMasses = expectedMasses[z_index][inBin]
				hasBH_inBin = relevantBlackHoleMasses / expectedMasses[z_index][inBin]
			else:
				hasBH_inBin = (relevantBlackHoleMasses > minimumMass).astype(float)

			#Bootstrapping architecture
			fractionEstimates = np.zeros(n_bootstrap)
			for boot in range(n_bootstrap):
				randomIntegers = np.random.randint(0, numberInBin, numberInBin)
				if takeLog:
					fractionEstimates[boot] = np.log10(np.sum(hasBH_inBin[randomIntegers]) / numberInBin)
				else:
					fractionEstimates[boot] = np.sum(hasBH_inBin[randomIntegers]) / numberInBin
			occupationFractions[z_index][i,:] = np.nanpercentile(fractionEstimates, (14,68))

	return stellarMassBinEdges, occupationFractions

def computeAGNFractions(ensemble, ensemble_path='./ensemble_output', logStellarMassBinEdges=np.array([9.0,9.5]), n_bootstrap=1000, redshifts=[0, 0.1, 0.2, 0.4, 0.5, 0.6], \
	Lx_range=(42.4,np.inf), takeLog=True):
	"""
	Calculate the occupation fractions of the ensemble
	"""

	ensemble_files = [file for file in os.listdir(ensemble_path+'/'+ensemble) if file[-5:]=='.pklz']
	print("Reading from {0}".format(ensemble))

	#Storage
	logStellarMasses = [[] for z in redshifts]
	xRayLuminosities_ergs = [[] for z in redshifts]
	unobscuredProbabilities = [[] for z in redshifts]

	for f_index in range(len(ensemble_files)):
		file = ensemble_files[f_index]

		#Unpack data
		with gzip.open(ensemble_path+"/"+ensemble+'/'+file, 'r') as myfile:
			megaDict = pickle.load(myfile)

		#NOTE:  These fractions include satellites that haven't merged yet by 0.  Be warned!
		redshiftsAvailable = np.unique(megaDict['redshift'])
		allRedshifts = megaDict['redshift']
		for z_index in range(len(redshifts)):
			redshift = redshifts[z_index]
			closestRedshift = redshiftsAvailable[np.argmin(np.abs(redshiftsAvailable-redshift))]
			logMhalo = np.log10(megaDict['m_halo'][allRedshifts==closestRedshift])
			xRayLuminosities_ergs[z_index].extend(af.LtoLx(megaDict['L_bol'][allRedshifts==closestRedshift]) * constants.L_sun / constants.erg * 10)
			logStellarMasses[z_index].extend(smhm.logMstar(logMhalo, redshift))
			unobscuredProbabilities[z_index].extend(af.typeIProbability(megaDict['L_bol'][allRedshifts==closestRedshift], closestRedshift))

	logStellarMasses = [np.array(subarray) for subarray in logStellarMasses]
	xRayLuminosities_ergs = [np.array(subarray) for subarray in xRayLuminosities_ergs]
	unobscuredProbabilties = [np.array(subarray) for subarray in unobscuredProbabilities]

	nbins = len(logStellarMassBinEdges)-1
	activeFractions = [np.zeros((nbins,2)) for z in redshifts]
	for z_index in range(len(redshifts)):
		for i in range(nbins):
			inBin = (logStellarMassBinEdges[i] <= logStellarMasses[z_index]) & (logStellarMasses[z_index] < logStellarMassBinEdges[i+1])
			numberInBin = sum(inBin)
			if numberInBin == 0:
				activeFractions[z_index][i,:] = np.nan
				continue
			logRelevantLuminosities = np.log10(xRayLuminosities_ergs[z_index][inBin])
			hasAGN_inBin = ((logRelevantLuminosities >= Lx_range[0]) & (logRelevantLuminosities < Lx_range[1])).astype(float)
			obscurationWeighting = unobscuredProbabilties[z_index][inBin]

			#Bootstrapping architecture
			fractionEstimates = np.zeros(n_bootstrap)
			for boot in range(n_bootstrap):
				randomIntegers = np.random.randint(0, numberInBin, numberInBin)
				if takeLog:
					fractionEstimates[boot] = np.log10(np.sum(hasAGN_inBin[randomIntegers] * obscurationWeighting[randomIntegers]) / numberInBin/10)
				else:
					fractionEstimates[boot] = np.sum(hasAGN_inBin[randomIntegers] * obscurationWeighting[randomIntegers]) / numberInBin
			activeFractions[z_index][i,:] = np.nanpercentile(fractionEstimates, (14,68))

			#Correct the upper limit based on the smallest actual fraction we could get.
			trueUpperLimit = 1.0/numberInBin
			if takeLog:
				trueUpperLimit = np.log10(trueUpperLimit)
			activeFractions[z_index][i,1] = np.maximum(activeFractions[z_index][i,1], trueUpperLimit)

	return logStellarMassBinEdges, activeFractions

def plotOccupationFractions(occupationFractionList, colors=['g'], labels=['Light'], figsize=(4,4.5), output=None, redshifts=[0], showMiller=True, \
	patchy=False, xlim=(7.5,12.0)):
	"""
	Plot outputs of computeOccupationFractions()
	"""

	fig, ax = plt.subplots(1, 1, figsize=figsize)

	if showMiller:
		#Weird code to add the Miller+2015 constraints.
		with open(currentPath + '../lookup_tables/bh_data/miller2015/miller2015_f2_bottom.dat', 'r') as myfile:
			data_bottom = np.loadtxt(myfile)
		with open(currentPath + '../lookup_tables/bh_data/miller2015/miller2015_f2_top.dat', 'r') as myfile:
			data_top = np.loadtxt(myfile)
		poly_x = np.concatenate((data_top[:,0], np.flipud(data_bottom[:,0])))
		poly_y = np.concatenate((data_top[:,1], np.flipud(data_bottom[:,1])))	
		polygon = [Polygon(np.transpose(np.vstack([poly_x,poly_y])))]
		ax.add_collection(PatchCollection(polygon, alpha=0.6, color='k'))
		ax.fill_between([], [], [], alpha=0.6, color='k', label='Miller+15')

	for e_index in range(len(occupationFractionList)):
		xaxis = occupationFractionList[e_index][0]
		for z_index in range(len(redshifts)):
			yaxis = occupationFractionList[e_index][1][z_index]
			if patchy:
				for m_index in range(len(xaxis)-1):
					ax.add_patch(patches.Rectangle((xaxis[m_index], yaxis[m_index,0]), (xaxis[m_index+1]-xaxis[m_index]), \
					(yaxis[m_index,1]-yaxis[m_index,0]), color=colors[z_index], alpha=1.0))
			else:
				ax.fill_between(0.5*(xaxis[1:]+xaxis[:-1]), yaxis[:,0], yaxis[:,1], color=colors[e_index], alpha=1.0)
			ax.fill_between([], [], [], color=colors[z_index], label=labels[z_index], alpha=1.0)

	ax.legend(loc='lower right', frameon=False)
	ax.set_xlabel(r'$\log (M_*/M_\odot)$', fontsize=13)
	ax.set_ylabel('Occupation Fraction', fontsize=13)
	ax.set_ylim(0,1.01)
	ax.set_xlim(xlim[0],xlim[1])
	fig.tight_layout()
	
	if output is not None:
		fig.savefig(output)
	else:
		fig.show()

def plotEffectiveOccupationFraction(occupationFractionList, colors=['g'], curveLabels=['Light'], figsize=(4,4), output=None, redshifts=[0], figshape=(1,1), \
	textLabel=[None], showMiller=False, patchy=False):
	"""
	Plot outputs of computeOccupationFractions()
	"""

	fig, axarr = plt.subplots(figshape[0], figshape[1], figsize=figsize, sharex=True, sharey=True)
	axarr = np.atleast_2d(axarr)

	for e_index in range(len(occupationFractionList)):
		i = int(e_index / figshape[0])
		j = e_index % figshape[0]
		ax = axarr[i,j]
		xaxis = occupationFractionList[e_index][0]

		#Add a dotted line at 0
		ax.plot([1,100], [0,0], ls='--', color='k', lw=1)

		for z_index in range(len(redshifts)):
			yaxis = occupationFractionList[e_index][1][z_index]
			if patchy:
				for m_index in range(len(xaxis)-1):
					ax.add_patch(patches.Rectangle((xaxis[m_index], yaxis[m_index,0]), (xaxis[m_index+1]-xaxis[m_index]), \
					(yaxis[m_index,1]-yaxis[m_index,0]), color=colors[z_index], alpha=0.8))
			else:
				ax.fill_between(0.5*(xaxis[1:]+xaxis[:-1]), yaxis[:,0], yaxis[:,1], color=colors[z_index], alpha=0.8)
			ax.fill_between([], [], [], color=colors[z_index], label=curveLabels[z_index], alpha=0.8)

		if showMiller:
			#Weird code to add the Miller+2015 constraints.
			with open(currentPath + '../lookup_tables/bh_data/miller2015/miller2015_f2_bottom.dat', 'r') as myfile:
				data_bottom = np.loadtxt(myfile)
			with open(currentPath + '../lookup_tables/bh_data/miller2015/miller2015_f2_top.dat', 'r') as myfile:
				data_top = np.loadtxt(myfile)
			poly_x = np.concatenate((data_top[:,0], np.flipud(data_bottom[:,0])))
			poly_y = np.concatenate((np.log10(data_top[:,1]), np.log10(np.flipud(data_bottom[:,1]))))
			polygon = [Polygon(np.transpose(np.vstack([poly_x,poly_y])))]
			ax.add_collection(PatchCollection(polygon, alpha=0.6, color='k'))
			ax.fill_between([], [], [], alpha=0.6, color='k', label='Miller+15')

		if (i == (figshape[0]-1)) & (j == (figshape[1]-1)):
			ax.legend(loc='lower left', frameon=False, ncol=2)
		if i == (figshape[0]-1):
			ax.set_xlabel(r'$\log (M_*/M_\odot)$', fontsize=13)
		if j == 0:
			ax.set_ylabel('log(Mass-Weighted Occupation)', fontsize=10)
		ax.text(10.2, 0.8, textLabel[e_index], fontsize=12)
		ax.set_xlim(7.8,11.5)
		ax.set_ylim(-1.0,1.0)
	fig.tight_layout()
	fig.subplots_adjust(hspace=0,wspace=0)

	if output is not None:
		fig.savefig(output)
	else:
		fig.show()

def plotOccupationFractions_minMass(occupationFractionList, colors=['g'], curveLabels=['Light'], figsize=(4,4), output=None, redshifts=[0], figshape=(1,1), \
	textLabel=[None], patchy=False, xlim=(7.5,11.5)):
	"""
	Plot outputs of computeOccupationFractions()
	"""

	fig, axarr = plt.subplots(figshape[0], figshape[1], figsize=figsize)
	axarr = np.atleast_2d(axarr)

	for e_index in range(len(occupationFractionList)):
		i = int(e_index / figshape[0])
		j = e_index % figshape[0]
		ax = axarr[i,j]
		xaxis = occupationFractionList[e_index][0]
		for z_index in range(len(redshifts)):
			yaxis = occupationFractionList[e_index][1][z_index]
			if patchy:
				for m_index in range(len(xaxis)-1):
					ax.add_patch(patches.Rectangle((xaxis[m_index], yaxis[m_index,0]), (xaxis[m_index+1]-xaxis[m_index]), \
					(yaxis[m_index,1]-yaxis[m_index,0]), color=colors[z_index], alpha=0.8))
			else:
				ax.fill_between(0.5*(xaxis[1:]+xaxis[:-1]), yaxis[:,0], yaxis[:,1], color=colors[z_index], alpha=0.8)
			ax.fill_between([], [], [], color=colors[z_index], label=curveLabels[z_index], alpha=0.8)

		with open(currentPath + '../lookup_tables/bh_data/Greene2012/Desroches.dat', 'r') as myfile:
			desroches_data = np.loadtxt(myfile)
			ax.scatter(np.log10(desroches_data[:,0]), desroches_data[:,2], s=50, marker='s', color='k')
		with open(currentPath + '../lookup_tables/bh_data/Greene2012/Gallo.dat', 'r') as myfile:
			gallo_data = np.loadtxt(myfile)
			ax.scatter(np.log10(gallo_data[:,0]), gallo_data[:,2], s=50, color='k', label='Greene2012')

		if (i == 0) & (j == (figshape[1]-1)):
			ax.legend(loc='lower right', frameon=False, ncol=1)
		if i == (figshape[0]-1):
			ax.set_xlabel(r'$\log (M_*/M_\odot)$', fontsize=13)
		if j == 0:
			ax.set_ylabel(r'Occupation Above $3 \times 10^5 \ M_\odot$', fontsize=11)
		if i > 0:
			ax.set_yticks(np.linspace(0,0.8,5))
		if j > 0:
			ax.set_yticks([])
		ax.text(7.7, 0.9, textLabel[e_index], fontsize=12)
		ax.set_xlim(xlim[0],xlim[1])
		ax.set_ylim(-0.01,1.01)
	fig.tight_layout()
	fig.subplots_adjust(hspace=0,wspace=0)

	if output is not None:
		fig.savefig(output)
	else:
		fig.show()

def plotActiveFraction(occupationFractionList, colors=['g'], curveLabels=['Light'], figsize=(4,4), output=None, redshifts=[0], figshape=(1,1), \
	textLabel=[None]):
	"""
	Plot outputs of computeOccupationFractions()
	"""

	fig, ax = plt.subplots(1, 1, figsize=figsize)

	offsets = np.linspace(-1.0,1.0,len(occupationFractionList))*0.02
	for e_index in range(len(occupationFractionList)):
		i = int(e_index / figshape[0])
		j = e_index % figshape[0]
		xaxis = np.array(redshifts) + offsets[e_index]
		yaxis_range = np.array(occupationFractionList[e_index][1])[:,0,:]
		isUpperLimit = yaxis_range[:,0] == -np.inf
		if np.any(isUpperLimit):
			ax.errorbar(xaxis[isUpperLimit], yaxis_range[isUpperLimit,1], yerr=0.2, uplims=[True]*np.sum(isUpperLimit), color=colors[e_index], ls='None', lw=2)
		if np.any(~isUpperLimit):
			ymean = np.mean(yaxis_range[~isUpperLimit,:], axis=1)
			yerr = 0.5 * np.squeeze(np.diff(yaxis_range[~isUpperLimit,:], axis=1))
			ax.errorbar(xaxis[~isUpperLimit], ymean, yerr=yerr, color=colors[e_index], ls='None', capsize=5, lw=2)
		ax.errorbar([], [], color=colors[e_index], label=textLabel[e_index])

		ax.legend(loc='upper right', frameon=False, ncol=2)
		ax.set_xlabel(r'$z$', fontsize=13)
		ax.set_ylabel('$\log_{10}(f_\mathrm{active})$', fontsize=12)
		ax.text(10.0, 1.2, textLabel[e_index], fontsize=12)
		ax.set_xlim(-0.05,0.65)
		ax.set_ylim(-4,-1)
	fig.tight_layout()

	if output is not None:
		fig.savefig(output)
	else:
		fig.show()

def saveOccupationFractions_text(occupations, outputName, labels):
	#outputName should be a folder
	if os.path.exists(outputName):
		print("Using existing folder, "+outputName)
	else:
		os.mkdir(outputName)

	#Need to make a 2D table for each model
	headerText = 'low, high'
	for e_index in range(len(occupations)):

		#Outputting
		textFileName = outputName+'/'+labels[e_index]+'.txt'
		np.savetxt(textFileName, occupations[e_index][1][0], header=headerText, delimiter=', ')

	#Then, add another files that just has the bin edges
	headerText = 'Stellar Mass [M_sun]'
	textFileName = outputName+'/'+'binEdgesStellarMass.txt'
	np.savetxt(textFileName, occupations[0][0], header=headerText, delimiter=', ')

if __name__ == '__main__':
	'''
	ensembles = ['powerLaw_dcbh_pmerge0.1_072018']#, 'agnms_popIII_021518', 'powerLaw_dcbh_112917', 'agnms_dcbh_120617']
	colors = ['royalblue', 'forestgreen', 'darkorange', 'firebrick']
	curveLabels = ['z=0','z=2','z=3','z=6']
	redshifts = [0,2,3,6]
	'''
	textLabel = ['Light-PL', 'Light-MS', 'Heavy-PL', 'Heavy-MS']
	'''
	ensembles = ['powerLaw_dcbh_pmerge0.1_061918', 'powerLaw_popIII_pmerge0.1_061918']
	colors = ['c', 'r']
	curveLabels = ['Heavy', 'Light']
	redshifts = [0]
	'''

	#Ordinary
	#of_calculations = [computeOccupationFractions(ensembles[e], redshifts=redshifts, nbins=20, weightByMass=False, takeLog=False) for e in range(len(ensembles))]
	#plotOccupationFractions(of_calculations, colors=colors, labels=curveLabels, redshifts=redshifts)
	
	#Effective
	#of_calculations = [computeOccupationFractions(e, redshifts=redshifts, nbins=20, weightByMass=True, takeLog=True) for e in ensembles]
	#plotEffectiveOccupationFraction(of_calculations, colors=colors, curveLabels=curveLabels, redshifts=redshifts, figshape=(2,2), figsize=(6,6), textLabel=textLabel)

	#Mass Threshold
	#ensembles = ['powerLaw_popIII_pmerge0.1_072018', 'agnms_popIII_pmerge0.1_072018', 'powerLaw_dcbh_pmerge0.1_072018', 'agnms_dcbh_pmerge0.1_072018']
	#redshifts = [0]
	#of_calculations = [computeOccupationFractions(e, redshifts=redshifts, nbins=20, weightByMass=False, takeLog=False, minimumMass=3e5) for e in ensembles]
	#plotOccupationFractions_minMass(of_calculations, colors=colors, curveLabels=curveLabels, redshifts=redshifts, figshape=(2,2), figsize=(6,6), textLabel=textLabel)
	saveOccupationFractions_text(of_calculations, './data_products/occupationFractions_massThreshold', textLabel)

	#AGN
	#redshifts = [0, 0.1, 0.2, 0.4, 0.5, 0.6]
	#of_calculations = [computeAGNFractions(e, redshifts=redshifts, takeLog=True) for e in ensembles]
	#plotActiveFraction(of_calculations, colors=colors, curveLabels=curveLabels, redshifts=redshifts, textLabel=textLabel)
