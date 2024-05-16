"""
ARR:  11.02.17

Compute contribution to the stochastic gravitational wave background.
"""

import numpy as np
from .. import constants
from .. import cosmology
from ..helpers import sam_functions as sf
from ..cosmology import cosmology_functions as cf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle
import os
import gzip
from scipy.interpolate import interp1d
from scipy.integrate import quad
from . import interpolateSignalToNoise as isn
from scipy.optimize import curve_fit
currentPath = os.path.abspath(os.path.dirname(__file__)) + '/'

def mergerContributionByHaloModel(haloMass, normalization):
	return normalization * haloMass * sf.calcHaloNumberDensity(haloMass,0)

class GravitationalWaveEventCalculator(object):

	def __init__(self, ensemble_folder, ensemble_path='./ensemble_output', n_mass=23, n_sample=20, logRange=(10.6,15), \
		hostRedshift=0, sensitivityCurve='lisa_sensitivity.dat'):
		""" 
		Read folder and assemble a gigantic list of gravitational wave events. Save numberDensities.

		ensemble_folder (arg):  string corresponding to the folder containing pickles in the ensemble_path

		n_mass (kwarg):  number of masses for which trees were made
		n_sample (kwarg):  number of times each mass was sampled
		logRange (kwarg):  log mass range of host halos
		"""

		#Read files
		self.n_sample = n_sample
		samplingFactor = float(n_mass) / (logRange[1] - logRange[0])
		ensemble_files = [file for file in os.listdir(ensemble_folder) if file[-5:]=='.pklz']
		
		#Storage
		self.masses = np.array([])
		self.redshifts = np.array([])
		self.massRatios = np.array([])
		self.spins = np.array([])
		self.remnantSpins = np.array([])
		self.chiEff = np.array([])
		self.numberDensities = np.array([])
		self.treeIndices = np.array([])
		self.hostMassesAtZero = np.array([])

		#Loop through, pull out merger information.
		for f_index in range(len(ensemble_files)):
			file = ensemble_files[f_index]
			hostHaloMass = 10**float(file.split('_')[-1].split('m')[1].split('n')[0])

			with gzip.open(ensemble_folder+'/'+file, 'rb') as myfile:
				megaDict = pickle.load(myfile)

			bh_mergers = megaDict['bh_mergers']
			for treeIndex in range(len(bh_mergers)):
				tree = bh_mergers[treeIndex]
				if len(tree) > 0:
					self.masses = np.concatenate((self.masses, tree[:,0]))
					self.massRatios = np.concatenate((self.massRatios, tree[:,1]))
					self.redshifts = np.concatenate((self.redshifts, tree[:,2]))
					self.remnantSpins = np.concatenate((self.remnantSpins, tree[:,3]))
					self.chiEff = np.concatenate((self.chiEff, tree[:,4]))
					self.spins = np.concatenate((self.spins, tree[:,5]))
					#Don't forget to get rid of little h.
					self.numberDensities = np.concatenate((self.numberDensities, np.full(tree.shape[0], sf.calcHaloNumberDensity(hostHaloMass, hostRedshift) / self.n_sample / samplingFactor * cosmology.h**3)))
					self.treeIndices = np.concatenate((self.treeIndices, np.full(tree.shape[0], treeIndex)))
					self.hostMassesAtZero = np.concatenate((self.hostMassesAtZero, np.full(tree.shape[0], hostHaloMass)))

		#Chirp mass definition
		self.chirpMasses = self.masses * self.massRatios**0.6 / (1.0 + self.massRatios)**0.2
		self.dzdt = cf.computedzdt(self.redshifts)
		self.dVdz = cf.computedVdz(self.redshifts)

		#Be default, everything is observable.  Will be modified later, if necessary
		self.observability = np.full(len(self.masses), 1.0)

		#Don't read this in until you have to.
		self.sensitivityFilename = sensitivityCurve
		self.sensitivityCurve = None
		self.sensitivityFunction = None

	def _signalToNoiseIntegrand(self, logf):
		"""
		DEPRECATED:  Used only by the slow computeSignalToNoise_slow function.  Now, tabluation is used.
		"""
		return (np.exp(logf)**(-1.0/6.0) / np.exp(self.sensitivityFunction(logf)))**2

	def computeSignalToNoise_slow(self, yearsOfObservation=4.0, sigmaThreshold=5.0):

		"""
		DEPRECATED:  It became worth my time to make a 3D table of the integral as a function of
		M1, M2, and z, then just interpolate.
		"""

		print("Evaluating signal to noise.  Get comfortable, because this involves the computation of {0} integrals.".format(len(self.masses)))

		#If you haven't done it already, open the sensitivity curve data and define an interpolation function.
		if self.sensitivityCurve is None:
			with open(self.sensitivityFilename, 'r') as myfile:
				self.sensitivityCurve = np.loadtxt(myfile)
				self.sensitivityFunction = interp1d(np.log(self.sensitivityCurve[:,0]), np.log(self.sensitivityCurve[:,1]), bounds_error=False, \
				fill_value='extrapolate')

		#Maximum frequency is that of the ISCO orbit
		f_ISCO = constants.c**3 / 6.0**1.5 / np.pi / constants.G / (self.masses * constants.M_sun * (1.0 + self.massRatios)) / (1.0 + self.redshifts)

		#This is a prefactor in front of the time derivative of frequency
		prefactors = (96.0 * np.pi**(8.0/3.0) * constants.G**(5.0/3.0) / (5.0 * constants.c**5) * (self.chirpMasses * constants.M_sun)**(5.0/3.0))**-1

		#A prefactor for the strain 
		comovingDistance = cf.computeLuminosityDistance(self.redshifts) / (1.0 + self.redshifts)
		strains = 1.0 / 3.0**0.5 / np.pi**(2.0/3.0) * constants.G**(5.0/6.0) / constants.c**1.5 * (self.chirpMasses * constants.M_sun)**(5.0/6.0) / \
		(comovingDistance * 1e6 * constants.pc)

		#Initialize array
		self.signalToNoiseArray = np.zeros(len(self.masses))
		tau = yearsOfObservation * constants.yr
		f_initial = np.maximum((8.0/3.0 / prefactors / (1.0 + self.redshifts) * tau + f_ISCO**(-8.0/3.0))**(-3.0/8.0), np.log(self.sensitivityCurve[0,0]))
		
		#If these object don't ever get above at 5 sigma in the sweet spot, then we can just ignore them and save time.
		maxStrainEver = strains * ((1.0 + self.redshifts) * f_initial)**(-1.0/6.0)
		minimumPossibleNoise = np.min(self.sensitivityCurve[:,1])
		ignorable = maxStrainEver < minimumPossibleNoise

		#There's a discontinuity in the integrand here
		f_break = ((5 * constants.c**5 * (1.0 + self.redshifts)**(-5.0/3.0)) / (96 * np.pi**(8.0/3.0) * constants.G**(5.0/3.0) * \
		(self.chirpMasses * constants.M_sun)**(5.0/3.0)) / (yearsOfObservation * constants.yr))**(3.0/8.0)
		
		self.signalToNoiseArray[ignorable,:] = 0.0
		for event in np.where(~ignorable)[0]:
			self.signalToNoiseArray[event] = strains[event] * (1.0 + self.redshifts[event])**(-1.0/6.0) * \
			np.sqrt(quad(self._signalToNoiseIntegrand, \
			np.maximum(np.log(f_break[event]),np.log(f_initial[event])), np.log(np.minimum(f_ISCO[event], self.sensitivityCurve[-1,0])), epsrel=0.01)[0])
			print("#{0}:  M_1 = {1:2.2e} solar masses, q = {2:2.2f}, z = {3:2.1f}, S/N = {4:3.3f}".format(event, \
				self.masses[event], self.massRatios[event], self.redshifts[event], self.signalToNoiseArray[event]))

		observingSuccesses = self.signalToNoiseArray > sigmaThreshold
		self.observability = observingSuccesses.astype(float)

	def evaluateSignalToNoise(self, sigmaThreshold=5.0):

		"""
		Uses a lookup table to compute the signal to noise of the events stored.
		"""

		#I'm interpolating in log space, so I need to set z=0 events to something near zero.  Rare anyway.
		usedRedshifts = self.redshifts
		usedRedshifts[self.redshifts==0] = 1e-20
		self.signalToNoiseArray = isn.interpSignalToNoise(self.masses, self.massRatios, usedRedshifts)
		self.observability = self.signalToNoiseArray > sigmaThreshold

	def computeEventRate(self, chirpMassLimits=(0,np.inf), ratioLimits=(0,1), redshiftLimits=(0,np.inf), spinLimits=(0,1), remnantSpinLimits=(0,1), chiEffLimits=(-1,1), n_bootstrap=10000, percentiles=(14,86), \
		weightByObservability=True, hostMassAtZero=None):

		"""
		Compute the event rate of events stored.
		"""

		#Perform various cuts on the data.
		massCut = (self.chirpMasses >= chirpMassLimits[0]) & (self.chirpMasses <= chirpMassLimits[1])
		ratioCut = (self.massRatios >= ratioLimits[0]) & (self.massRatios <= ratioLimits[1])
		redshiftCut = (self.redshifts >= redshiftLimits[0]) & (self.redshifts <= redshiftLimits[1])
		spinCut = (self.spins >= spinLimits[0]) & (self.spins <= spinLimits[1])
		remnantSpinCut = (self.remnantSpins >= remnantSpinLimits[0]) & (self.remnantSpins <= remnantSpinLimits[1])
		chiEffCut = (self.chiEff >= chiEffLimits[0]) & (self.chiEff <= chiEffLimits[1])
		combinedCut = massCut & ratioCut & redshiftCut & spinCut & remnantSpinCut & chiEffCut

		if hostMassAtZero is not None:
			uniqueHostMasses = np.unique(self.hostMassesAtZero)
			closestHostMass = uniqueHostMasses[np.argmin(np.abs(uniqueHostMasses-hostMassAtZero))]
			hostMassCut = self.hostMassesAtZero == closestHostMass
			combinedCut = combinedCut & hostMassCut

		valueOfRealization = np.zeros(self.n_sample)
		for h_index in range(self.n_sample):
			treeCut = self.treeIndices == h_index
			totalCut = combinedCut & treeCut
			if weightByObservability:
				valueOfRealization[h_index] = np.sum(self.observability[totalCut]*self.numberDensities[totalCut]*self.dzdt[totalCut]*self.dVdz[totalCut]/(1.0+self.redshifts[totalCut]))
			else:
				valueOfRealization[h_index] = np.sum(self.numberDensities[totalCut]*self.dzdt[totalCut]*self.dVdz[totalCut]/(1.0+self.redshifts[totalCut]))

		bootstraps = np.zeros(n_bootstrap)
		for boot in range(n_bootstrap):
			randomFileNumbers = np.random.randint(0, self.n_sample, self.n_sample)
			bootstraps[boot] = np.sum(valueOfRealization[randomFileNumbers])

		output = np.percentile(bootstraps, percentiles)

		return output

	def computeEventRateByRedshift(self, chirpMassLimits=(0,np.inf), ratioLimits=(0,1), redshiftBins=np.linspace(0,20,21), weightByObservability=True):

		output = np.zeros((len(redshiftBins)-1,2))	
		for z_index in range(len(redshiftBins)-1):
			output[z_index,:] = self.computeEventRate(chirpMassLimits=chirpMassLimits, ratioLimits=ratioLimits, \
			redshiftLimits=(redshiftBins[z_index],redshiftBins[z_index+1]), weightByObservability=weightByObservability)

		return redshiftBins, output

	def computeEventRateByMass(self, chirpMassBins=np.logspace(2,11,21), ratioLimits=(0,1), redshiftLimits=(0,20), weightByObservability=True):

		output = np.zeros((len(chirpMassBins)-1,2))
		for m_index in range(len(chirpMassBins)-1):
			output[m_index,:] = self.computeEventRate(chirpMassLimits=(chirpMassBins[m_index],chirpMassBins[m_index+1]), \
			ratioLimits=ratioLimits, redshiftLimits=redshiftLimits, weightByObservability=weightByObservability)

		return chirpMassBins, output

	def computeEventRateByHostMass(self, hostMasses=np.logspace(10.6,15,23), weightByObservability=True):

		output = np.zeros((len(hostMasses),2))
		for m_index in range(len(hostMasses)):
			output[m_index,:] = self.computeEventRate(hostMassAtZero=hostMasses[m_index], weightByObservability=weightByObservability)

		return hostMasses, output

	def computeEventRateByRemnantSpin(self, remnantSpinBins=np.linspace(0,1,21), weightByObservability=True):

		output = np.zeros((len(remnantSpinBins)-1,2))
		for a_index in range(len(remnantSpinBins)-1):
			output[a_index,:] = self.computeEventRate(remnantSpinLimits=(remnantSpinBins[a_index],remnantSpinBins[a_index+1]), weightByObservability=weightByObservability)

		return remnantSpinBins, output

	def computeEventRateBySpin(self, spinBins=np.linspace(0,1,21), weightByObservability=True):

		output = np.zeros((len(spinBins)-1,2))
		for a_index in range(len(spinBins)-1):
			output[a_index,:] = self.computeEventRate(spinLimits=(spinBins[a_index],spinBins[a_index+1]), weightByObservability=weightByObservability)

		return spinBins, output

	def computeEventRateByChiEff(self, chiBins=np.linspace(-1,1,21), weightByObservability=True):

		output = np.zeros((len(chiBins)-1,2))
		for a_index in range(len(chiBins)-1):
			output[a_index,:] = self.computeEventRate(chiEffLimits=(chiBins[a_index],chiBins[a_index+1]), weightByObservability=weightByObservability)

		return chiBins, output

def plotEventRateByRedshift(gw_packs, colors, labels, figsize=(4,4), yearsOfObservation=None, yscale='log', xlim=(0,20), \
	ylim=(1e-2,1e2), output=None, mode='fill'):

	fig, ax = plt.subplots(figsize=figsize)

	if yearsOfObservation == None:
		ylabel = '$d^2N/dz dt \ [\mathrm{yr}^{-1}$]'
	else:
		ylabel = "Events After {0} Years".format(yearsOfObservation)

	for e_index in range(len(gw_packs)):
		zbinList, histoList = gw_packs[e_index]
		zbinList = np.array(zbinList)
		histoList = np.array(histoList)
		if yearsOfObservation == None:
			zbinSizes = np.diff(zbinList)
			histoList /= zbinSizes[0]
		
		#This line fixes a bug in matplotlib for fill_between and log scaling.
		histoList[histoList == 0] = 1e-99

		if yearsOfObservation is not None:
			histoList = histoList * yearsOfObservation
		if mode == 'fill':
			ax.fill_between(0.5*(zbinList[1:]+zbinList[:-1]), histoList[:,0], histoList[:,1], color=colors[e_index], alpha=0.7)
		elif mode == 'boxes':
			for z_index in range(len(zbinList)-1):
							ax.add_patch(patches.Rectangle((zbinList[z_index], histoList[z_index,0]), (zbinList[z_index+1]-zbinList[z_index]), \
							(histoList[z_index,1]-histoList[z_index,0]), color=colors[e_index], alpha=0.7))
		ax.fill_between([], [], [], color=colors[e_index], label=labels[e_index], alpha=0.7)

	ax.plot(xlim, [1,1], ls='--', lw=1, color='k', zorder=0)
	ax.legend(frameon=False, loc='upper right')
	ax.set_xlim(xlim)
	ax.set_ylim(ylim)
	ax.set_xlabel("$z$", fontsize=14)
	ax.set_ylabel(ylabel, fontsize=13)
	ax.set_yscale(yscale)

	fig.tight_layout()
	if output is None:
		fig.show()
	else:
		fig.savefig(output)

def plotEventRateByMass(gw_packs, colors, labels, figsize=(4,4), yearsOfObservation=None, yscale='log', xlim=(1e2,1e10), \
	ylim=(1e-3,1e2), output=None, mode='fill'):

	fig, ax = plt.subplots(figsize=figsize)

	if yearsOfObservation == None:
		ylabel = '$d^2N/d\log M_\mathrm{chirp} dt \ [\mathrm{yr}^{-1}$]'
	else:
		ylabel = "Events After {0} Years".format(yearsOfObservation)

	for e_index in range(len(gw_packs)):
		mbinList, histoList = gw_packs[e_index]
		mbinList = np.array(mbinList)
		logMbinList = np.log10(mbinList)
		histoList = np.array(histoList)
		if yearsOfObservation == None:
			massBinSizes = np.diff(logMbinList)
			histoList /= massBinSizes[0]

		#This line fixes a bug in matplotlib for fill_between and log scaling.
		histoList[histoList == 0] = 1e-99

		if yearsOfObservation is not None:
			histoList = histoList * yearsOfObservation
		if mode == 'fill':
			ax.fill_between(10**(0.5*(logMbinList[1:]+logMbinList[:-1])), histoList[:,0], histoList[:,1], color=colors[e_index], alpha=0.7)
		elif mode == 'boxes':
			for m_index in range(len(mbinList)-1):
				ax.add_patch(patches.Rectangle((mbinList[m_index], histoList[m_index,0]), (mbinList[m_index+1]-mbinList[m_index]), \
				(histoList[m_index,1]-histoList[m_index,0]), color=colors[e_index], alpha=0.7))
		ax.fill_between([], [], [], color=colors[e_index], label=labels[e_index], alpha=0.7)

	ax.plot(xlim, [1,1], ls='--', lw=1, color='k', zorder=0)

	ax.legend(frameon=False, loc='upper right')
	ax.set_xlim(xlim)
	ax.set_ylim(ylim)
	ax.set_xlabel(r"$M_\mathrm{chirp} \ [\mathrm{M}_\odot]$", fontsize=13)
	ax.set_ylabel(ylabel, fontsize=13)
	ax.set_xscale('log')
	ax.set_yscale(yscale)
	fig.tight_layout()
	if output is None:
		fig.show()
	else:
		fig.savefig(output)

def plotEventRateBySpin(gw_packs, colors, labels, figsize=(4,4), yearsOfObservation=None, yscale='log', xlim=(0,1), \
	ylim=(1e-3,1e2), output=None, mode='fill'):

	fig, ax = plt.subplots(figsize=figsize)

	if yearsOfObservation == None:
		ylabel = '$d^2N/d\log M_\mathrm{chirp} dt \ [\mathrm{yr}^{-1}$]'
	else:
		ylabel = "Events After {0} Years".format(yearsOfObservation)

	for e_index in range(len(gw_packs)):
		abinList, histoList = gw_packs[e_index]
		abinList = np.array(abinList)
		histoList = np.array(histoList)
		if yearsOfObservation == None:
			spinBinSizes = np.diff(abinList)
			histoList /= spinBinSizes[0]

		#This line fixes a bug in matplotlib for fill_between and log scaling.
		histoList[histoList == 0] = 1e-99

		if yearsOfObservation is not None:
			histoList = histoList * yearsOfObservation
		if mode == 'fill':
			ax.fill_between(0.5*(abinList[1:]+abinList[:-1]), histoList[:,0], histoList[:,1], color=colors[e_index], alpha=0.7)
		elif mode == 'boxes':
			for a_index in range(len(abinList)-1):
				ax.add_patch(patches.Rectangle((abinList[a_index], histoList[a_index,0]), (abinList[a_index+1]-abinList[a_index]), \
				(histoList[a_index,1]-histoList[a_index,0]), color=colors[e_index], alpha=0.7))
		ax.fill_between([], [], [], color=colors[e_index], label=labels[e_index], alpha=0.7)

	ax.plot(xlim, [1,1], ls='--', lw=1, color='k', zorder=0)

	ax.legend(frameon=False, loc='upper right')
	ax.set_xlim(xlim)
	ax.set_ylim(ylim)
	ax.set_xlabel(r"$a_{\bullet}$", fontsize=13)
	ax.set_ylabel(ylabel, fontsize=13)
	ax.set_yscale(yscale)
	fig.tight_layout()
	if output is None:
		fig.show()
	else:
		fig.savefig(output)

def plotEventRateByChiEff(gw_packs, colors, labels, figsize=(4,4), yearsOfObservation=None, yscale='log', xlim=(-1,1), \
	ylim=(1e-3,1e2), output=None, mode='fill'):

	fig, ax = plt.subplots(figsize=figsize)

	if yearsOfObservation == None:
		ylabel = '$d^2N/d\log M_\mathrm{chirp} dt \ [\mathrm{yr}^{-1}$]'
	else:
		ylabel = "Events After {0} Years".format(yearsOfObservation)

	for e_index in range(len(gw_packs)):
		abinList, histoList = gw_packs[e_index]
		abinList = np.array(abinList)
		histoList = np.array(histoList)
		if yearsOfObservation == None:
			spinBinSizes = np.diff(abinList)
			histoList /= spinBinSizes[0]

		#This line fixes a bug in matplotlib for fill_between and log scaling.
		histoList[histoList == 0] = 1e-99

		if yearsOfObservation is not None:
			histoList = histoList * yearsOfObservation
		if mode == 'fill':
			ax.fill_between(0.5*(abinList[1:]+abinList[:-1]), histoList[:,0], histoList[:,1], color=colors[e_index], alpha=0.7)
		elif mode == 'boxes':
			for a_index in range(len(abinList)-1):
				ax.add_patch(patches.Rectangle((abinList[a_index], histoList[a_index,0]), (abinList[a_index+1]-abinList[a_index]), \
				(histoList[a_index,1]-histoList[a_index,0]), color=colors[e_index], alpha=0.7))
		ax.fill_between([], [], [], color=colors[e_index], label=labels[e_index], alpha=0.7)

	ax.plot(xlim, [1,1], ls='--', lw=1, color='k', zorder=0)

	ax.legend(frameon=False, loc='upper right')
	ax.set_xlim(xlim)
	ax.set_ylim(ylim)
	ax.set_xlabel(r"$\chi_\mathrm{Eff}$", fontsize=13)
	ax.set_ylabel(ylabel, fontsize=13)
	ax.set_yscale(yscale)
	fig.tight_layout()
	if output is None:
		fig.show()
	else:
		fig.savefig(output)

def plotEventRateByHostMass(gw_packs, colors, labels, figsize=(4,4), yearsOfObservation=None, yscale='log', xlim=(10**10.6,1e15), \
	ylim=(1e-3,1e2), output=None, modelNorm=None):

	fig, ax = plt.subplots(figsize=figsize)

	for e_index in range(len(gw_packs)):
		mList, histoList = gw_packs[e_index]
		if yearsOfObservation is not None:
			histoList = histoList * yearsOfObservation
		ax.errorbar(mList, np.mean(histoList, axis=1), yerr=np.diff(histoList, axis=1)/2, \
		color=colors[e_index], alpha=0.7, label=labels[e_index])
		if modelNorm is not None:
			simpleModel = mergerContributionByHaloModel(mList, modelNorm[e_index])
			if yearsOfObservation is not None:
				simpleModel *= yearsOfObservation
			ax.plot(mList, simpleModel, color=colors[e_index], lw=2)

	ax.legend(frameon=False, loc='upper right')
	ax.set_xlim(xlim)
	ax.set_xscale('log')
	ax.set_ylim(ylim)
	ax.set_xlabel(r"$M_h(z=0) \ [M_\odot]$", fontsize=13)
	if yearsOfObservation is None:
		ax.set_ylabel("Events At Host Halo Mass Per Year", fontsize=13)
	else:
		ax.set_ylabel("Events After {0} Years".format(yearsOfObservation), fontsize=13)
	ax.set_yscale(yscale)
	fig.tight_layout()
	if output is None:
		fig.show()
	else:
		fig.savefig(output)

def computeTotalEvents(gw_packs_m, yearsOfObservation=4):

	averages = np.array([np.average(np.sum(gw_packs_m[i][1], axis=0)) * yearsOfObservation for i in range(len(gw_packs_m))])
	errors = np.array([0.5*np.squeeze(np.diff(gw_packs_m[i][1], axis=1)) for i in range(len(gw_packs_m))])
	propagatedErrors = np.sqrt(np.sum(errors**2, axis=1))

	print("Total Numbers:", averages)
	print("Statistical Uncertainty:", propagatedErrors)

def fitModel(eventsByHostMass, fitRange=(1e11,1e13)):
	mList, histoList = eventsByHostMass
	inBounds = (mList>=fitRange[0]) & (mList<=fitRange[1])
	
	parameters = curve_fit(mergerContributionByHaloModel, mList[inBounds], np.mean(histoList[inBounds,:], axis=1), sigma=np.squeeze(np.diff(histoList[inBounds,:], axis=1))/2)
	return parameters

def packToText(output, gw_pack, variable='M_chirp', rescale=False, logx=False):

	newx = 0.5*(gw_pack[0][1:] + gw_pack[0][:-1])
	if logx:
		binSize = np.diff(np.log10(newx))[0]
	else:
		binSize = np.diff(newx)[0]
	if rescale:
		newy = np.array(gw_pack[1]) / binSize
	else:
		newy = gw_pack[1]
	bigArray = np.hstack((np.reshape(newx, (len(newx),1)), gw_pack[1]))
	np.savetxt(output, bigArray, header=variable+" lowRate highRate")
	
if __name__ == '__main__':
	sigmaThreshold = 8.0
	'''
	ensembles = ['powerLaw_popIII_pmerge0.1_072018', 'agnms_popIII_pmerge0.1_072018', 'powerLaw_dcbh_pmerge0.1_072018', 'agnms_dcbh_pmerge0.1_072018']
	colors = ['r', 'orange', 'purple', 'c']
	labels = ['Light-PL', 'Light-MS', 'Heavy-PL', 'Heavy-MS']
	'''

	ensembles = ['powerLaw_popIII_pmerge1.0_072018', 'agnms_popIII_pmerge1.0_072018', 'powerLaw_dcbh_pmerge1.0_072018', 'agnms_dcbh_pmerge1.0_072018']
	colors = ['r', 'orange', 'purple', 'c']
	labels = ['Light-PL*', 'Light-MS*', 'Heavy-PL*', 'Heavy-MS*']

	gw_packs_z = []
	gw_packs_m = []
	#gw_packs_0 = []
	#modelNorms = []
	for e_index in range(len(ensembles)):
		calculator = GravitationalWaveEventCalculator(ensembles[e_index])
		calculator.evaluateSignalToNoise(sigmaThreshold=sigmaThreshold)
		gw_packs_z.append(calculator.computeEventRateByRedshift(chirpMassLimits=(0,np.inf)))
		gw_packs_m.append(calculator.computeEventRateByMass(chirpMassBins=np.logspace(2,11,41)))
		#gw_packs_0.append(calculator.computeEventRateByHostMass(weightByObservability=True))
		#modelNorms.append(fitModel(gw_packs_0[-1], fitRange=(3e11,7e12))[0])
	#plotEventRateByRedshift(gw_packs_z, colors, labels, yearsOfObservation=4, ylim=(1e-1,4e2), xlim=(0,20))
	#plotEventRateByMass(gw_packs_m, colors, labels, yearsOfObservation=4, ylim=(1e-1,4e2), xlim=(2e2,1e7))
	plotEventRateByRedshift(gw_packs_z, colors, labels, yearsOfObservation=None, ylim=(1e-1,1e2), xlim=(0,20))
	plotEventRateByMass(gw_packs_m, colors, labels, yearsOfObservation=None, ylim=(1e-1,4e2), xlim=(2e2,1e7))

	savenames = [name+'-mergeAll' for name in ['Light-PL', 'Light-MS', 'Heavy-PL', 'Heavy-MS']]
	#savenames = [name+'-mergeATenth' for name in ['Light-PL', 'Light-MS', 'Heavy-PL', 'Heavy-MS']]
	for e_index in range(len(ensembles)):
		packToText('./data_products/lisa_white_paper_2020/Redshift/'+savenames[e_index], gw_packs_z[e_index], variable='Redshift', rescale=True, logx=False)
		packToText('./data_products/lisa_white_paper_2020/ChirpMass/'+savenames[e_index], gw_packs_m[e_index], variable='M_chirp', rescale=True, logx=True)

	#plotEventRateByHostMass(gw_packs_0, colors, labels, yearsOfObservation=4, ylim=(1e-1,3e1), xlim=(10**10.6,1e15), modelNorm=modelNorms)
	#computeTotalEvents(gw_packs_m, yearsOfObservation=4)
