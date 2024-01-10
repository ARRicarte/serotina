"""
ARR 11.05.15
This program reads in merger tree output and paints black holes onto them.
"""

import numpy as np
from . import cosmology
from . import constants
from .helpers import sam_functions as sf
from .helpers import bh_imfs, smhm, fabioMergerProbabilities
import pickle
import gzip
from . import accretion_recipes as acc
from . import black_hole_mergers as bhb
import time as tim
from . import config
from .util import crossmatch, findFirstDuplicate2, findFirstDuplicate2_2d, findDuplicates, primariesAndSecondaries 

class SAM(object):

	def __init__(self, treefile, parameters=None, silent=False):

		"""
		Initialize the merger tree and storage.  Globalize keywords.
		"""

		#Interpret the input parameters and assign them to the SAM.
		self.setParameters(parameters)

		#Open merger tree file.  These are zipped numpy arrays.  (Will there be version issues?)
		with gzip.open(treefile, 'rb') as myfile:
			if not silent:
				print("Opening dark matter merger tree: " + treefile)
			mergerTreeTable = np.load(myfile)

		#Tree files have a particular structure that I imposed.
		self.m_halo = mergerTreeTable[:,0]
		self.redshift = mergerTreeTable[:,1]
		self.parent = mergerTreeTable[:,2].astype(int) - 1
		self.progenitor = mergerTreeTable[:,3].astype(int) - 1
		self.nchild = mergerTreeTable[:,4].astype(int)
		#self.m_smooth = mergerTreeTable[:,5]
		self.n_node = mergerTreeTable.shape[0]
		self.nodeProperties = ['m_halo', 'redshift', 'parent', 'progenitor', 'nchild', 'time']

		#Get a time axis, representing age of the universe.
		if not silent:
			print("Calculating times and stellar masses.")
		self.uniqueRedshift = np.flipud(np.unique(self.redshift))
		self.uniqueTime = sf.z2t(self.uniqueRedshift)		#Gyr
		self.time = np.zeros(self.n_node)
		for lev in range(len(self.uniqueRedshift)):
			timeAtLev = self.uniqueTime[len(self.uniqueTime)-1-lev]
			self.time[self.redshift==self.uniqueRedshift[len(self.uniqueTime)-1-lev]] = timeAtLev		   

		#Stellar masses
		self.m_star = smhm.Mstar(self.m_halo, self.redshift)

		#Set up storage.  Black holes move among progenitors, which move along nodes.  If you ever want to start over, use self.initializeStorage()
		if not silent:
			print("Initializing storage arrays.")
		self.initializeStorage(keepSpins=False)
		
		if not silent:
			print("SAM initialized.")

	def setParameters(self, parameters):
		"""
		Interpret the input parameters of the SAM.
		"""

		#Parameters should be a dictionary.
		if isinstance(parameters, dict):
			parameterDictionary = parameters
		elif parameters is None:
			parameterDictionary = {}

		#Default values.
		parameterDefaults = {'seedingMode': 'dcbh', 'seedingEfficiency': 1.0, 'useDelayTimes': False, \
		'majorMergerMassRatio': 0.1, 'constant_f_min': None, 'constant_f_max': None, 'z_superEdd': 30, 'howToSave': 'progenitors', \
		'spinEvolution': False, 'MsigmaNormalization': 8.45, 'MsigmaSlope': 5.0, 'spinMax': 0.998, 'mergerKicks': False, 'fEdd_MyrPerDex': 10.0, \
		'useMsigmaCap': True, 'fixedSeedMass': None, 'silent': False, 'randomFractionOn': 0.0, 'alignQuasarSpins': True, 'seedingSigma': 3.5, \
		'useColdReservoir': False, 'f_superEdd': 5.0, 'makeEllipticals': False, 'includeDecline': False, 'f_EddCrit': None, \
		'minimumSeedingRedshift': 15, 'maximumSeedingRedshift': 20, 'useMetalConstraints': False, 'supernovaBarrier': False, 'blackHoleMergerProbability': 1.0, \
		'steady': None, 'isothermalSigma': False, 'defaultRadiativeEfficiency': 0.1, 'Q_dcbh': 3.0, 'nscMassThreshold': 1e8, 'mergerMode': 'flatProbability'}

		#For each parameter, set it to the dictionary value.
		for parameterKey in parameterDefaults.keys():
			if parameterKey in parameterDictionary.keys():
				setattr(self, parameterKey, parameterDictionary[parameterKey])
			else:
				setattr(self, parameterKey, parameterDefaults[parameterKey])

		#Check if there were any keys that did not get used.  This could be a mistake.
		for parameterKey in parameterDictionary.keys():
			if not (parameterKey in parameterDefaults.keys()):
				print("WARNING:  You provided a value for "+parameterKey+", but this is not a parameter.  Skipping.")

	def initializeStorage(self, keepSpins=True):
		"""
		Reset the SAM storage.
		"""

		self.createEmptyProgenitorArrays(keepSpins=keepSpins)
		self.createEmptyBlackHoleArrays()
		self.step = 0
		self.nSteps = len(self.uniqueRedshift)

	def createEmptyProgenitorArrays(self, keepSpins=False):
		"""
		Create (or reset) all arrays for progenitors (not nodes).
		"""

		#This points to the node in the merger tree at which the galaxy currently resides
		self.progToNode = np.where(self.nchild==0)[0]
		self.n_prog = len(self.progToNode)

		self.lastMajorMerger = np.zeros(self.n_prog)
		self.feedTime = np.full(self.n_prog, np.inf)
		self.merged = np.zeros(self.n_prog, dtype=bool)
		self.stripped = np.zeros(self.n_prog, dtype=bool)
		self.elliptical = np.zeros(self.n_prog, dtype=bool)
		self.metalPolluted = np.zeros(self.n_prog, dtype=bool)
		self.satelliteToCentral = np.full(self.n_prog, -1, dtype=int)
		self.haloMergerTime = np.full(self.n_prog, np.inf)
		if not keepSpins:
			self.spin_halo = sf.drawRandomHaloSpin(self.n_prog)
		self.progenitorProperties = ['progToNode', 'lastMajorMerger', 'feedTime', 'merged', \
		'stripped', 'elliptical', 'spin_halo', 'metalPolluted', 'satelliteToCentral', 'haloMergerTime']

	def createEmptyBlackHoleArrays(self):
		"""
		Create empty arrays for black holes that are expanded during seeding.
		"""

		self.n_bh = 0
		self.totalSeedMass = 0.0
		self.bh_id = np.empty(0, dtype=int)
		self.m_bh = np.empty(0)
		self.m_init = np.empty(0)
		self.m_merged = np.empty(0)
		self.m_burst = np.empty(0)
		self.m_steady = np.empty(0)
		self.spin_bh = np.empty(0)
		self.seedType = np.empty(0, dtype=(str,10))
		self.bhToProg = np.empty(0, dtype=int)
		self.seedTime = np.empty(0)
		self.t_decline = np.empty(0)
		self.t_fEdd = np.empty(0)
		self.mode = np.empty(0, dtype=(str,10))
		self.accretionBudget = np.empty(0)
		self.merged_bh = np.empty(0, dtype=bool)
		self.wandering = np.empty(0, dtype=bool)
		self.eddRatio = np.empty(0)
		self.L_bol = np.empty(0)
		self.lastIntegrationTime = np.empty(0)
		self.scheduledMergeTime = np.empty(0)

		self.blackHoleProperties = ['bh_id', 'm_bh', 'm_init', 'm_merged', 'm_burst', 'm_steady', 'spin_bh', 'seedType', 'bhToProg', \
		'seedTime', 't_decline', 't_fEdd', 'mode', 'accretionBudget', 'merged_bh', 'wandering', 'eddRatio', 'L_bol', 'lastIntegrationTime', \
		'scheduledMergeTime']

		#This special list is populated with M1, q, and z whenever a black hole merger occurs.
		self.bh_mergers = np.empty((0,3))

	def seed(self, relevantProgenitors, m_d=0.05, alpha_c=0.06, T_gas=5000, j_d=0.05, noProgenitor=False, \
		noMergers=False):
		"""
		Seeding prescription.
		"""

		if noProgenitor:
			#Don't allow seeding in halos that have been around for more than one time step.  This would be resolution-dependent.
			progenitorCut = self.progenitor[self.progToNode] == -1
			relevantProgenitors = relevantProgenitors & progenitorCut
		if noMergers:
			#Don't allow seeding if the halo has experienced a major merger, which may result in metal pollution.
			mergeCut = self.lastMajorMerger == 0
			relevantProgenitors = relevantProgenitors & mergeCut

		if not np.any(relevantProgenitors):
			#This means there is nothing to seed, and you're done.
			return

		seedTypes = []
		seedMasses = []
		seededProgenitors = []
		
		if (self.seedingMode == 'dcbh') | (self.seedingMode == 'mixed'):
			#NOTE:  Allowing this calculation to proceed even if there's already a seed present.  If there is a seed, it grows to the maximum mass.
			potentialSeedIndices, = np.where(relevantProgenitors)

			#First cut by redshift, then calculate virial temperatures and spins
			#MBW p375.  Remember that a SIS needs to be truncated, leading to a pressure term and a factor of 2.5
			v_vir = np.sqrt(constants.G * self.m_halo[self.progToNode[potentialSeedIndices]] * constants.M_sun / \
			sf.r_vir(self.m_halo[self.progToNode[potentialSeedIndices]], self.redshift[self.progToNode[potentialSeedIndices]])) / 1e3
			T_vir = 3.6e5 * (v_vir/100)**2

			#If the spin is above this, the disk is rotationally supported.
			spinMax = m_d*self.Q_dcbh/8*(m_d/j_d)*np.sqrt(T_vir/T_gas)
			spin = self.spin_halo[potentialSeedIndices]

			#Only take subset that fits spin criterion and has a hotter halo than the gas.
			firstPass = (spin < spinMax) & (T_vir > T_gas)
			firstCut = potentialSeedIndices[firstPass]
			spin = spin[firstPass]
			T_vir = T_vir[firstPass]

			#Will create some nans here that get removed in the subsequent cut.
			potentialSeedMasses = m_d*self.m_halo[self.progToNode[firstCut]]*(1.0 - np.sqrt(8*spin/m_d/self.Q_dcbh*(j_d/m_d)*np.sqrt(T_gas/T_vir)))

			#Above this temperature, internal disk torques cannot sustain the disk, and it fragments.
			T_max = T_gas * (4 * alpha_c / m_d / (1 + potentialSeedMasses/(m_d*self.m_halo[self.progToNode[firstCut]])))**(2.0/3.0)

			#Then cut by virial temperature
			secondCut = firstCut[T_vir<T_max]
			cutSeedMasses = potentialSeedMasses[T_vir<T_max]

			if len(secondCut) > 0:
				#Here, we check if there's already a seed in any of these progenitors.  If there is, take the maximum mass.
				isCentral = (~self.merged_bh) & (~self.wandering) & (self.scheduledMergeTime==0)
				hasASeedAlready = np.in1d(secondCut, self.bhToProg[isCentral])
				if any(hasASeedAlready):
					matcher = crossmatch(secondCut[hasASeedAlready], self.bhToProg[isCentral])
					self.m_bh[isCentral][matcher[1]] = np.maximum(self.m_bh[isCentral][matcher[1]], cutSeedMasses[hasASeedAlready][matcher[0]])

					#Remove these from the seeding list
					secondCut = secondCut[~hasASeedAlready]
					cutSeedMasses = cutSeedMasses[~hasASeedAlready]

				seededProgenitors.extend(secondCut)
				seedTypes.extend(['DCBH']*len(secondCut))
				if self.fixedSeedMass is not None:
					seedMasses.extend([self.fixedSeedMass]*len(secondCut))
				else:
					seedMasses.extend(cutSeedMasses.tolist())
		if (self.seedingMode == 'popIII') | (self.seedingMode == 'mixed'):
			#Do not add any seeds to anything that already has a seed.
			potentialSeedIndices = np.setdiff1d(np.where(relevantProgenitors)[0], self.bhToProg[(~self.merged_bh) & (~self.wandering) & (self.scheduledMergeTime==0)])
			if len(potentialSeedIndices) == 0:
				return

			#Simple, stupid model:  Every halo at some redshift above a nu-sigma cutoff gets a BH seed.
			sigmaPeaks = sf.m2nu(self.m_halo[self.progToNode[potentialSeedIndices]], self.uniqueRedshift[self.step])
			seedIndices = potentialSeedIndices[sigmaPeaks > self.seedingSigma].tolist()

			#Make sure you're not adding a PopIII seed to something that already has a DCBH seed.
			realSeedIndices = filter(lambda prog: prog not in seededProgenitors, seedIndices)
			seededProgenitors.extend(realSeedIndices)
			seedTypes.extend(['PopIII']*len(realSeedIndices))
			if self.fixedSeedMass is not None:
				seedMasses.extend([self.fixedSeedMass]*len(realSeedIndices))
			else:
				seedMasses.extend(bh_imfs.drawPopIIIMass(len(realSeedIndices)).tolist())
		if (self.seedingMode == 'nsc'):
			#Do not add any seeds to anything that already has a seed.
			potentialSeedIndices = np.setdiff1d(np.where(relevantProgenitors)[0], self.bhToProg[(~self.merged_bh) & (~self.wandering) & (self.scheduledMergeTime==0)])
			if len(potentialSeedIndices) == 0:
				return

			#Assume that any galaxy can get seeded with a SMBH if it has a stellar mass above some threshold.
			stellarMasses = self.m_star[self.progToNode[potentialSeedIndices]]
			realSeedIndices = potentialSeedIndices[stellarMasses >= self.nscMassThreshold]
			if len(realSeedIndices) == 0:
				return
			seededProgenitors.extend(realSeedIndices.tolist())
			seedTypes.extend(['nsc']*len(seededProgenitors))
			seedMasses.extend((1e-5*self.m_star[self.progToNode[realSeedIndices]]).tolist())

		#Concatenate the black hole arrays
		if len(seedMasses) > 0:
			n_new = len(seedMasses)
			self.n_bh += n_new
			self.bh_id = np.concatenate((self.bh_id, np.arange(len(self.bh_id),len(self.bh_id)+n_new)))
			self.m_bh = np.concatenate((self.m_bh, np.array(seedMasses)))
			self.totalSeedMass += np.sum(np.array(seedMasses))
			self.m_init = np.concatenate((self.m_init, np.array(seedMasses)))
			self.m_merged = np.concatenate((self.m_merged, np.zeros(n_new)))
			self.m_burst = np.concatenate((self.m_burst, np.zeros(n_new)))
			self.m_steady = np.concatenate((self.m_steady, np.zeros(n_new)))
			self.spin_bh = np.concatenate((self.spin_bh, np.zeros(n_new)))
			self.seedType = np.concatenate((self.seedType, np.array(seedTypes)))
			self.bhToProg = np.concatenate((self.bhToProg, np.array(seededProgenitors)))
			self.seedTime = np.concatenate((self.seedTime, np.full(n_new,self.uniqueTime[self.step])))
			self.t_decline = np.concatenate((self.t_decline, np.zeros(n_new)))
			self.t_fEdd = np.concatenate((self.t_fEdd, np.zeros(n_new)))
			newModes = np.full(n_new,'',dtype=(str,10))
			if self.uniqueRedshift[self.step] >= self.z_superEdd:
				dice = np.random.random(n_new)
				newModes[dice < self.randomFractionOn] = 'super-Edd'
			self.mode = np.concatenate((self.mode, newModes))
			self.accretionBudget = np.concatenate((self.accretionBudget, np.zeros(n_new)))
			if self.constant_f_min is not None:
				self.eddRatio = np.concatenate((self.eddRatio, np.full(n_new, self.constant_f_min)))
			else:
				self.eddRatio = np.concatenate((self.eddRatio, sf.draw_typeII(n_new, np.full(n_new, self.uniqueRedshift[self.step]))))
			self.merged_bh = np.concatenate((self.merged_bh, np.zeros(n_new, dtype=bool)))
			self.wandering = np.concatenate((self.wandering, np.zeros(n_new, dtype=bool)))
			self.L_bol = np.concatenate((self.L_bol, np.zeros(n_new)))
			self.lastIntegrationTime = np.concatenate((self.lastIntegrationTime, np.full(n_new,self.uniqueTime[self.step])))
			self.scheduledMergeTime = np.concatenate((self.scheduledMergeTime, np.zeros(n_new)))
		
	def mergeBHs(self, primaries, secondaries, progenitors, times):
		"""
		Merge two black holes.  Keeping the properties of the first index.
		"""

		npts = len(primaries)

		if (self.mergerKicks) | (self.spinEvolution):
			#Draw random spins here, to be self-consistent in both calculations.

			theta1 = 2*np.pi*np.random.random(npts)
			theta2 = 2*np.pi*np.random.random(npts)
			phi1 = 2*np.pi*np.random.random(npts)
			phi2 = 2*np.pi*np.random.random(npts)

			if self.alignQuasarSpins:
				onesToAlign = ((self.mode[primaries] == 'quasar') | (self.mode[secondaries] == 'quasar') | (self.accretionBudget[primaries] > 0))
				#If the BH is currently a quasar, I assume that spins align.
				theta1 = np.full(npts, 0.0)
				theta2 = np.full(npts, 0.0)
				phi1 = np.full(npts, 0.0)
				phi2 = np.full(npts, 0.0)

		if self.spinEvolution:
			self.spin_bh[primaries] = bhb.calcRemnantSpin(self.m_bh[primaries], self.m_bh[secondaries], \
			self.spin_bh[primaries], self.spin_bh[secondaries], \
			theta1=theta1, theta2=theta2, phi1=phi1, phi2=phi2, spinMax=self.spinMax)

		#Save to a list of all merger events.
		self.bh_mergers = np.vstack((self.bh_mergers, np.array([self.m_bh[primaries], self.m_bh[secondaries]/self.m_bh[primaries], sf.t2z(times)]).transpose()))

		#Simply adding the two masses together.
		self.m_bh[primaries] += self.m_bh[secondaries]
		self.m_merged[primaries] += self.m_bh[secondaries]
		self.seedType[primaries] = 'Merged'
		self.bhToProg[primaries] = progenitors

		#Now let's see if the kick is large enough to cause it to leave the halo.
		if self.mergerKicks:
			mergerKicks = bhb.calcRemnantKick(self.m_bh[primaries], self.m_bh[secondaries], self.spin_bh[primaries], self.spin_bh[secondaries], theta1=theta1, theta2=theta2, phi1=phi1, phi2=phi2)
			
			#Compare kick velocity to Choksi formula
			kicked = (mergerKicks > sf.calcRecoilEscapeVelocity_permanent(self.m_halo[self.progToNode[progenitors]], sf.t2z(times)))
		else:
			kicked = np.zeros(npts, dtype=bool)

		#Note:  The escape velocity here is kind of made up.  Depends on baryonic distribution, dark matter profile...

		#Set to wandering, halt accretion.
		if np.any(kicked):
			self.createWanderers(primaries[kicked], progenitors[kicked])

		#Get rid of the secondary, which no longer exists.
		self.eliminateBH(secondaries)

	def eliminateBH(self, bh_ids):
		"""
		Used when a secondary merges into another black hole and no longer exists
		"""

		self.bhToProg[bh_ids] = -1
		self.merged_bh[bh_ids] = True
		self.scheduledMergeTime[bh_ids] = 0
		self.accretionBudget[bh_ids] = 0
		self.t_decline[bh_ids] = 0
		self.mode[bh_ids] = ''

	def createWanderers(self, bh_ids, targets):
		"""
		Keep the primaries in their progenitors, set secondaries to wandering.  Used when BHs do not merge 
		"""

		self.bhToProg[bh_ids] = targets
		self.wandering[bh_ids] = True
		self.scheduledMergeTime[bh_ids] = 0
		self.accretionBudget[bh_ids] = 0
		self.t_decline[bh_ids] = 0		
		self.mode[bh_ids] = ''

	def scheduleBHTransfer(self, bh_ids, targets, times):
		#Wanderers are destined to remain wanderers
		isWandering = self.wandering[bh_ids]
		self.createWanderers(bh_ids[isWandering], targets[isWandering])

		#Draw from a distribution of delay times for the rest of the BHs
		if np.any(~isWandering):
			self.scheduledMergeTime[bh_ids[~isWandering]] = times[~isWandering] + sf.drawMergerDelayTimes(np.sum(~isWandering))
		self.bhToProg[bh_ids] = targets

	def transferBHs(self, bh_ids, targets, times):
		#Wanderers just need to get bhToProg updated
		isWandering = self.wandering[bh_ids]
		self.bhToProg[bh_ids[isWandering]] = targets[isWandering]

		if np.any(~isWandering):
			#Locate centrals in the target halos to set up potential mergers.
			targetsWithCentrals, centralBlackHolesInTargets = self.findMatchingBlackHolesToProgenitor(targets[~isWandering])
			actualMergers = np.in1d(targets[~isWandering], targetsWithCentrals)
			mergerMatches = crossmatch(targets[~isWandering], targetsWithCentrals)
			
			#Simply change bhToProg for the things not in duplicates
			self.bhToProg[bh_ids[~isWandering][~actualMergers]] = targets[~isWandering][~actualMergers]

			if np.any(actualMergers):
				bhid_array = np.transpose(np.vstack((bh_ids[~isWandering][mergerMatches[0]], centralBlackHolesInTargets[mergerMatches[1]])))
				mass_array = np.transpose(np.vstack((self.m_bh[bh_ids[~isWandering][mergerMatches[0]]], self.m_bh[centralBlackHolesInTargets[mergerMatches[1]]])))
				primaries, secondaries = primariesAndSecondaries(bhid_array, mass_array)

				#Trigger a black hole merger
				self.mergeBHs(primaries, secondaries, targets[~isWandering][mergerMatches[0]], times[~isWandering][mergerMatches[0]])

		#Reset the merge time.
		self.scheduledMergeTime[bh_ids] = 0

	def computeHaloMergerTime(self, satelliteProgenitors, centralProgenitors):
		#Factor of 10, since this program calculates it at the virial radius.
		t_dyn = 10 * sf.t_dyn(self.redshift[self.progToNode[centralProgenitors]])

		#Note:  I'm not just using self.progToNode[centralProgenitors] because they're already moved to a more massive parent node.
		haloMassRatios = self.m_halo[self.progToNode[satelliteProgenitors]] / self.m_halo[self.progenitor[self.parent[self.progToNode[satelliteProgenitors]]]]
		return self.time[self.progToNode[centralProgenitors]] + sf.computeMergerTime(t_dyn, haloMassRatios)

	def assignSatellites(self, newSatellites, newCentrals):
		self.satelliteToCentral[newSatellites] = newCentrals
		self.haloMergerTime[newSatellites] = self.computeHaloMergerTime(newSatellites, newCentrals)

	def transferSatellitesAndHoles(self, lostProgenitors, targets):
		for pair_index in range(len(lostProgenitors)):
			#Anything that had a satellite assigned to the progenitors goes to the target instead
			satellitesOfLost = self.satelliteToCentral==lostProgenitors[pair_index]
			self.satelliteToCentral[satellitesOfLost] = targets[pair_index]

			#Any black holes in the progenitors go to the target instead.
			bhsOfLost = self.bhToProg==lostProgenitors[pair_index]
			self.bhToProg[bhsOfLost] = targets[pair_index]

			#Turn off any quasars
			self.mode[bhsOfLost] = ''
			self.accretionBudget[bhsOfLost] = 0

	def mergeHalo(self, indices, targets):
		#Get rid of the merged halo.
		self.merged[indices] = True
		self.progToNode[indices] = -1
		self.feedTime[indices] = np.inf

		#Metal pollution might be an option for seeds.
		self.metalPolluted[targets] = self.metalPolluted[targets] | self.metalPolluted[indices]

		#Ellipticals might be an option for fueling.
		if self.makeEllipticals:
			#Koda (2007) criterion
			haloMassRatios = self.m_halo[self.progToNode[indices]] / self.m_halo[self.progenitor[self.parent[self.progToNode[indices]]]]
			newEllipticalTargets = (sf.v_peak(self.m_halo[self.progToNode[targets]], self.redshift[self.progToNode[targets]]) > 55) & \
			(haloMassRatio > 0.3)
			self.elliptical[targets[newEllipticalTargets]] = True		

		#If this had any satellites and black holes, give them to the new targets.
		self.transferSatellitesAndHoles(indices, targets)

	def M_bhSigma(self, progIndex):
		"""
		The default limit comes from momentum-driven winds and an isothermal sphere.
		"""

		return 10**(self.MsigmaNormalization + self.MsigmaSlope*np.log10(self.sigma[self.progToNode[progIndex]]/200))

	def integrateAll_superEdd(self, bhs, timeSteps, times):
		"""
		self.z_superEdd < self.uniqueRedshift[self.step]
		"""

		accreting = self.mode[bhs] == 'super-Edd'
		accretors = bhs[accreting]
		nonAccretors = bhs[~accreting]
		if np.any(accreting):
			initialMasses = self.m_bh[accretors]

			#No mass limits!
			massLimits = np.full(sum(accreting), np.inf)

			#Integrate with Cython package
			self.m_bh[accretors], self.spin_bh[accretors], self.L_bol[accretors] = \
			acc.integrateAccretion(self.m_bh[accretors], self.spin_bh[accretors], np.full(sum(accreting),self.f_superEdd), \
			timeSteps[accreting], mu=1, spinTracking=self.spinEvolution, f_EddCrit=self.f_EddCrit, spinMax=self.spinMax, spinlessDefault=self.defaultRadiativeEfficiency)

			#Define Eddington ratio based on Mdot as if efficiency is fixed at 0.1
			self.eddRatio[accretors] = np.log(self.m_bh[accretors]/initialMasses) / timeSteps[accreting] * \
			(constants.t_Sal / constants.yr / 1e9) / (1.0-self.defaultRadiativeEfficiency) * self.defaultRadiativeEfficiency
			self.accretionBudget[accretors] = np.max(np.vstack((self.accretionBudget[accretors] - (self.m_bh[accretors] - initialMasses), \
			np.full(accretors.shape[0],0,dtype=np.float))), axis=0)

		#Make sure that the non-accretors have 0 Eddington ratio and luminosity
		if np.any(~accreting):
			self.L_bol[nonAccretors] = 0
			self.eddRatio[nonAccretors] = 0
		self.lastIntegrationTime[bhs] = times

	def integrateAll_nobudget_cap(self, bh, timeStep, time):
		"""
		DEPRECATED:  Update to do multiple bhs at once.

		~self.useColdReservoir & self.useMsigmaCap
		"""

		if (self.f_min > 0) | (self.mode[bh] != '') | (self.t_decline[bh] > 0):
			oldMass = self.m_bh[bh]
			massLimit = self.M_bhSigma(self.bhToNode[bh])
			self.m_bh[bh], self.spin_bh[bh], self.L_bol[bh], self.t_decline[bh], self.t_fEdd[bh] = \
			acc.accretionComposite(self.m_bh[bh], self.spin_bh[bh], timeStep, time, f_EddCrit=self.f_EddCrit, \
			spinTracking=self.spinEvolution, t_fEdd=self.t_fEdd[bh], t_decline=self.t_decline[bh], maxQuasarMass=massLimit, \
			f_EddMax=self.f_max, f_EddMin=self.f_min, triggerDecline=self.includeDecline, spinMax=self.spinMax, spinlessDefault=self.defaultRadiativeEfficiency)
			self.eddRatio[bh] = np.log(self.m_bh[bh]/oldMass) / timeStep * (constants.t_Sal / constants.yr / 1e9) / (1.0-self.defaultRadiativeEfficiency) * self.defaultRadiativeEfficiency
			if self.m_bh[bh] == massLimit:
				self.mode[bh] == ''
		else:
			self.L_bol[bh] = 0
			self.eddRatio[bh] = 0
		self.lastIntegrationTime[bh] = time

	def integrateAll_nobudget_cap_nodecline(self, bhs, timeSteps, times):
		"""
		~self.useColdReservoir & self.useMsigmaCap & ~self.includeDecline
		"""

		#Determine which black holes are accreting
		accreting = (self.mode[bhs] == 'quasar') | np.full(len(bhs), self.constant_f_min>0, dtype=bool) | np.full(len(bhs), self.constant_f_min is None, dtype=bool)
		accretors = bhs[accreting]
		nonAccretors = bhs[~accreting]
		redshifts = sf.t2z(times)
		if np.any(accreting):
			initialMasses = self.m_bh[accretors]

			#Mass limits are determined by the accretionBudget.
			massLimitsQuasar = self.M_bhSigma(self.bhToProg[accretors])
			massLimitsSteady = np.full(len(accretors), np.inf)

			#Integrate with Cython package
			f_EddMax = np.zeros(sum(accreting))
			if self.constant_f_max is not None:
				f_EddMax[self.mode[accretors]=='quasar'] = self.constant_f_max
			else:
				f_EddMax[self.mode[accretors]=='quasar'] = sf.draw_typeI(np.sum(self.mode[accretors]=='quasar'), sf.t2z(times[accreting][self.mode[accretors]=='quasar']))
				#f_EddMax[self.mode[accretors]=='quasar'] = np.minimum(1.0, sf.draw_typeI(np.sum(self.mode[accretors]=='quasar'), sf.t2z(times[accreting][self.mode[accretors]=='quasar'])))
				"""
				f_EddMax[self.mode[accretors]=='quasar'] = sf.randomWalk_typeI(self.eddRatio[accretors], timeSteps[accreting], \
				self.fEdd_MyrPerDex, sf.t2z(times[accreting]))
				"""
			if self.steady == None:
				f_EddMin = np.zeros(sum(accreting))
			elif self.steady == 'PowerLaw':
				f_EddMin = sf.draw_typeII(sum(accreting), redshifts[accreting], logBounds=(-4.0,0.0), slope0=-0.9)
				#No steady mode for satellites due to RPS
				f_EddMin[self.satelliteToCentral[self.bhToProg[bhs[accreting]]]!=-1] = 0
			elif self.steady == 'AGNMS':
				#Estimate the SFR
				haloIDs = self.progToNode[self.bhToProg[accretors]]

				#Subtract away the merged component of the SFR
				DeltaM_star = np.array([self.m_star[haloIDs[i]] - \
				np.sum(self.m_star[self.progenitor[haloIDs[i]]:self.progenitor[haloIDs[i]]+self.nchild[haloIDs[i]]]) for i in range(len(accretors))])
				Deltat_star = self.time[haloIDs] - self.time[self.progenitor[haloIDs]]
				sfr = np.zeros(len(accretors))
				hasProgenitor = self.progenitor[haloIDs]!=-1
				sfr[hasProgenitor] = np.maximum(0, DeltaM_star[hasProgenitor]/Deltat_star[hasProgenitor])
				#If there is no progenitor, the sfr is left at 0.

				Delta_m_min = 1e-3 * sfr * timeSteps
				f_EddMin = np.minimum(self.constant_f_max,np.maximum(0,np.log(Delta_m_min/self.m_bh[bhs]+1) * (constants.t_Sal/constants.yr/1e9) * \
				self.defaultRadiativeEfficiency / (1.0-self.defaultRadiativeEfficiency) / timeSteps))
				#No steady mode for satellites due to RPS
				f_EddMin[self.satelliteToCentral[self.bhToProg[accretors]]!=-1] = 0
			"""
			if self.constant_f_min is not None:
				f_EddMin[:] = self.constant_f_min
			else:
				needsNewRatio = (self.eddRatio[accretors] == 0) & (initialMasses < massLimitsSteady)
				f_EddMin[needsNewRatio] = sf.draw_typeII(sum(needsNewRatio), redshifts[accreting][needsNewRatio])
				f_EddMin[~needsNewRatio] = sf.randomWalk_typeII(self.eddRatio[accretors[~needsNewRatio]], timeSteps[accreting][~needsNewRatio], \
				self.fEdd_MyrPerDex, redshifts[accreting][~needsNewRatio])
			"""

			#NOTE TO SELF:  Always aligned for now.

			self.m_bh[accretors], self.spin_bh[accretors], self.L_bol[accretors], self.eddRatio[accretors] = \
			test = acc.accretionDualMode(self.m_bh[accretors], self.spin_bh[accretors], np.full(len(accretors), 1), timeSteps[accreting], times[accreting], f_EddMax, f_EddMin, 
			f_EddCrit=self.f_EddCrit, includeSpinDependence=self.spinEvolution, maxBurstMass=massLimitsQuasar, maxSteadyMass=massLimitsSteady, spinMax=self.spinMax, \
			fiducialRadiativeEfficiency=self.defaultRadiativeEfficiency)

			#Change mode if necessary
			finishedQuasars = self.m_bh[accretors] >= massLimitsQuasar
			self.mode[accretors[finishedQuasars]] = ''

			#Adjust budget
			bhMassDifference = self.m_bh[accretors] - initialMasses
			possibleQuasarMassDifference = np.maximum(0, massLimitsQuasar - initialMasses)

			#Keep track of what was gained from the burst mode and the steady mode.
			self.m_burst[accretors] += np.minimum(bhMassDifference, possibleQuasarMassDifference)
			self.m_steady[accretors] += np.maximum(bhMassDifference - possibleQuasarMassDifference, 0)

		#Make sure that the non-accretors have 0 Eddington ratio and luminosity
		if np.any(~accreting):
			self.L_bol[nonAccretors] = 0
			self.eddRatio[nonAccretors] = 0
		self.lastIntegrationTime[bhs] = times

	def integrateAll_budget_nocap_nodecline(self, bhs, timeSteps, times):
		"""
		self.useColdReservoir & self.useMsigmaCap & ~self.includeDecline
		"""

		#Determine which black holes are accreting
		accreting = (self.accretionBudget[bhs] > 0) | np.full(len(bhs), self.constant_f_min>0, dtype=bool) | np.full(len(bhs), self.constant_f_min is None, dtype=bool)
		accretors = bhs[accreting]
		nonAccretors = bhs[~accreting]
		redshifts = sf.t2z(times)
		if np.any(accreting):
			initialMasses = self.m_bh[accretors]

			#Mass limits are determined by the accretionBudget.
			massLimitsBurst = initialMasses + self.accretionBudget[accretors]
			massLimitsSteady = np.full(len(accretors), np.inf)

			#Integrate with Cython package
			f_EddMax = np.ones(sum(accreting))
			typeI = self.accretionBudget[accretors] > 0
			typeII = ~typeI
			if np.any(typeI):
				if self.constant_f_max is not None:
					f_EddMax[typeI] = np.full(sum(typeI), self.constant_f_max)
				else:
					f_EddMax[typeI] = sf.randomWalk_typeI(self.eddRatio[accretors[typeI]], timeSteps[accreting][typeI], \
					self.fEdd_MyrPerDex, sf.t2z(times[accreting][typeI]))
			if self.steady == None:
				f_EddMin = np.zeros(sum(accreting))
			elif self.steady == 'PowerLaw':
				f_EddMin = sf.draw_typeII(sum(accreting), redshifts[accreting], logBounds=(-4.0,0.0), slope0=-0.9)
				#No steady mode for satellites due to RPS
				f_EddMin[self.satelliteToCentral[self.bhToProg[bhs[accreting]]]!=-1] = 0
			elif self.steady == 'AGNMS':
				#Estimate the SFR
				haloIDs = self.progToNode[self.bhToProg[accretors]]

				#Subtract away the merged component of the SFR
				DeltaM_star = np.array([self.m_star[haloIDs[i]] - \
				np.sum(self.m_star[self.progenitor[haloIDs[i]]:self.progenitor[haloIDs[i]]+self.nchild[haloIDs[i]]]) for i in range(len(accretors))])
				Deltat_star = self.time[haloIDs] - self.time[self.progenitor[haloIDs]]
				sfr = np.zeros(len(accretors))
				hasProgenitor = self.progenitor[haloIDs]!=-1
				sfr[hasProgenitor] = np.maximum(0, DeltaM_star[hasProgenitor]/Deltat_star[hasProgenitor])
				#If there is no progenitor, the sfr is left at 0.

				Delta_m_min = 1e-3 * sfr * timeSteps
				f_EddMin = np.minimum(self.constant_f_max,np.maximum(0,np.log(Delta_m_min/self.m_bh[bhs]+1) * (constants.t_Sal/constants.yr/1e9) * \
				self.defaultRadiativeEfficiency / (1.0-self.defaultRadiativeEfficiency) / timeSteps))
				#No steady mode for satellites due to RPS
				f_EddMin[self.satelliteToCentral[self.bhToProg[bhs[accreting]]]!=-1] = 0
			'''
			if self.constant_f_min is not None:
				f_EddMin[:] = np.full(sum(accreting), self.constant_f_min)
			else:
				needsNewRatio = (self.eddRatio[accretors] == 0) & (initialMasses < massLimitsSteady)
				f_EddMin[needsNewRatio] = sf.draw_typeII(sum(needsNewRatio), redshifts[accreting][needsNewRatio])
				f_EddMin[~needsNewRatio] = sf.randomWalk_typeII(self.eddRatio[accretors[~needsNewRatio]], timeSteps[accreting][~needsNewRatio], \
				self.fEdd_MyrPerDex, redshifts[accreting][~needsNewRatio])
			'''
			self.m_bh[accretors], self.spin_bh[accretors], self.L_bol[accretors], self.eddRatio[accretors] = \
			acc.accretionDualMode(self.m_bh[accretors], self.spin_bh[accretors], timeSteps[accreting], times[accreting], f_EddMax, f_EddMin, 
			f_EddCrit=self.f_EddCrit, spinTracking=self.spinEvolution, maxBurstMass=massLimitsBurst, maxSteadyMass=massLimitsSteady, spinMax=self.spinMax, \
			spinlessDefault=self.defaultRadiativeEfficiency)

			#Adjust budget
			bhMassDifference = self.m_bh[accretors] - initialMasses

			#Keep track of what was gained from the burst mode and the steady mode.
			self.m_burst[accretors] += np.minimum(bhMassDifference, self.accretionBudget[accretors])
			self.m_steady[accretors] += np.maximum(bhMassDifference - self.accretionBudget[accretors], 0)

			self.accretionBudget[accretors] = np.maximum(self.accretionBudget[accretors] - bhMassDifference, 0)

		#Make sure that the non-accretors have 0 Eddington ratio and luminosity
		if np.any(~accreting):
			self.L_bol[nonAccretors] = 0
			self.eddRatio[nonAccretors] = 0
		self.lastIntegrationTime[bhs] = times

	def integrateAll_budget_cap_nodecline(self, bhs, timeSteps, times):
		"""
		self.useColdReservoir & self.useMsigmaCap & ~self.includeDecline
		"""

		#Determine which black holes are accreting
		accreting = (self.eddRatio[bhs] > 0)
		if (self.constant_f_min is None) | (self.constant_f_min > 0):
			accreting[:] = True
		accretors = bhs[accreting]
		nonAccretors = bhs[~accreting]
		redshifts = sf.t2z(times)
		if np.any(accreting):
			initialMasses = self.m_bh[accretors]

			#Mass limits are determined by the accretionBudget.
			massLimitsBurst = initialMasses + self.accretionBudget[accretors]
			massLimitsSteady = self.M_bhSigma(self.bhToProg[accretors])

			#TODO:  Clean this up.  Setting things above M-s to 1e-5
			aboveMS = initialMasses >= massLimitsSteady
			massLimitsSteady[aboveMS] = np.inf

			#Integrate with Cython package
			f_EddMax = np.ones(sum(accreting))
			typeI = self.accretionBudget[accretors] > 0
			typeII = ~typeI
			if np.any(typeI):
				if self.constant_f_max is not None:
					f_EddMax[typeI] = np.full(sum(typeI), self.constant_f_max)
				else:
					f_EddMax[typeI] = sf.randomWalk_typeI(self.eddRatio[accretors[typeI]], timeSteps[accreting][typeI], \
					self.fEdd_MyrPerDex, sf.t2z(times[accreting][typeI]))
			f_EddMin = np.zeros(sum(accreting))
			if self.constant_f_min is not None:
				f_EddMin[:] = np.full(sum(accreting), self.constant_f_min)
			else:
				needsNewRatio = (self.eddRatio[accretors] == 0)
				f_EddMin[needsNewRatio] = sf.draw_typeII(sum(needsNewRatio), redshifts[accreting][needsNewRatio])
				f_EddMin[~needsNewRatio] = sf.randomWalk_typeII(self.eddRatio[accretors[~needsNewRatio]], timeSteps[accreting][~needsNewRatio], \
				self.fEdd_MyrPerDex, redshifts[accreting][~needsNewRatio])
			f_EddMin[aboveMS] = 1e-5
			self.m_bh[accretors], self.spin_bh[accretors], self.L_bol[accretors], self.eddRatio[accretors] = \
			acc.accretionDualMode(self.m_bh[accretors], self.spin_bh[accretors], timeSteps[accreting], times[accreting], f_EddMax, f_EddMin, 
			f_EddCrit=self.f_EddCrit, spinTracking=self.spinEvolution, maxBurstMass=massLimitsBurst, maxSteadyMass=massLimitsSteady, spinMax=self.spinMax, \
			spinlessDefault=self.defaultRadiativeEfficiency)

			#Adjust budget
			bhMassDifference = self.m_bh[accretors] - initialMasses
			self.accretionBudget[accretors] = np.maximum(self.accretionBudget[accretors] - bhMassDifference, 0)

		#Make sure that the non-accretors have 0 Eddington ratio and luminosity
		if np.any(~accreting):
			self.L_bol[nonAccretors] = 0
			self.eddRatio[nonAccretors] = 0
		self.lastIntegrationTime[bhs] = times

	def integrateAll_budget_cap_decline(self, bhs, timeSteps, times):
		"""
		self.useColdReservoir & self.useMsigmaCap
		"""

		#Determine which black holes are accreting
		accreting = (self.f_min > 0) | (self.accretionBudget[bhs] > 0) | (self.t_decline[bhs] > 0)
		accretors = bhs[accreting]
		nonAccretors = bhs[~accreting]
		if np.any(accreting):
			initialMasses = self.m_bh[accretors]

			#Mass limits are determined by the accretionBudget
			massLimits = np.minimum(initialMasses + self.accretionBudget[accretors], np.maximum(self.M_bhSigma(self.bhToProg[accretors]), self.m_bh[accretors]))

			#Integrate with Cython package
			if self.constant_f_max is not None:
				f_EddMax = np.full(sum(accreting), self.constant_f_max)
			else:
				f_EddMax = sf.drawEddingtonRatios(sum(accreting))
			self.m_bh[accretors], self.spin_bh[accretors], self.L_bol[accretors], self.t_decline[accretors], self.t_fEdd[accretors] = \
			acc.accretionComposite(self.m_bh[accretors], self.spin_bh[accretors], timeSteps[accreting], times[accreting], f_EddCrit=self.f_EddCrit, \
			spinTracking=self.spinEvolution, t_fEdd=self.t_fEdd[accretors], t_decline=self.t_decline[accretors], maxQuasarMass=massLimits, \
			f_EddMax=f_EddMax, f_EddMin=self.f_min, triggerDecline=self.includeDecline, spinMax=self.spinMax, spinlessDefault=self.defaultRadiativeEfficiency)

			#Define Eddington ratio based on Mdot as if efficiency is fixed at 0.1
			self.eddRatio[accretors] = np.log(self.m_bh[accretors]/initialMasses) / timeSteps[accreting] * \
			(constants.t_Sal / constants.yr / 1e9) / (1.0-self.defaultRadiativeEfficiency) * self.defaultRadiativeEfficiency
			self.accretionBudget[accretors] = np.max(np.vstack((self.accretionBudget[accretors] - (self.m_bh[accretors] - initialMasses), \
			np.full(accretors.shape[0],0,dtype=np.float))), axis=0)

		#Make sure that the non-accretors have 0 Eddington ratio and luminosity
		if np.any(~accreting):
			self.L_bol[nonAccretors] = 0
			self.eddRatio[nonAccretors] = 0
		self.lastIntegrationTime[bhs] = times

	def findMatchingBlackHolesToProgenitor(self, progenitorIndices, includeCentral=True, includeWandering=False, includeMerging=False):
		"""
		Given a list of progenitor indices, match them up with their associated black holes.
		"""

		#At no point would you ever want to include the ones which have merged already; these aren't real
		mask = ~self.merged_bh
		if not includeWandering:
			mask = mask & (~self.wandering)
		if not includeMerging:
			mask = mask & (self.scheduledMergeTime == 0)
		if not includeCentral:
			mask = mask & (self.wandering | (self.scheduledMergeTime > 0))
		relevantBHs, = np.where(mask)
		bh_matches = crossmatch(self.bhToProg[mask], progenitorIndices)
		return np.atleast_1d(progenitorIndices[bh_matches[1]]), np.atleast_1d(relevantBHs[bh_matches[0]])

	def printProperties(self, index):
		"""
		Print everything there is to know about this progenitor.
		"""
		properties = [p for p in self.__dict__]
		for p in properties:
			try:
				length = len(self.__dict__[p])
			except TypeError:
				continue
			if length == self.n_prog:
				print(p+':', self.__dict__[p][index])
			if length == self.n_node:
				print(p+':', self.__dict__[p][self.progToNode[index]])

	def printDiagnostics(self):
		"""
		Some final BH stats to be printed after evolveUniverse() is called.
		"""
		if self.n_bh == 0:
			print("No black holes ever existed.")
			return

		print("###############")
		print("##BLACK HOLES##")
		print("###############")
		print("Initial Number of BHs:", self.n_bh)
		print("Log Initial Mass in BHs:", np.log10(np.sum(self.m_init)))
		print("Final Number of BHs:", np.sum(~self.merged_bh))
		print("Log Final Mass in BHs:", np.log10(np.sum(self.m_bh[~self.merged_bh])))
		print("Final Number of Central BHs in the Central Halo:", np.sum((~self.wandering) & (self.scheduledMergeTime==0) & (~self.merged_bh) & (self.satelliteToCentral[self.bhToProg]==-1)))
		print("Final Number of Wandering BHs in the Central Halo:", np.sum((self.wandering) & (~self.merged_bh) & (self.satelliteToCentral[self.bhToProg]==-1)))
		print("Final Number of Central BHs in Satellite Halos:", np.sum((~self.wandering) & (self.scheduledMergeTime==0) & (~self.merged_bh) & (self.satelliteToCentral[self.bhToProg]!=-1)))
		print("Final Number of Wandering BHs in Satellite Halos:", np.sum((self.wandering) & (~self.merged_bh) & (self.satelliteToCentral[self.bhToProg]!=-1)))
		print("Final Number of BHs en-route to Merger:", np.sum((self.scheduledMergeTime>0) & (~self.merged_bh)))
		print("Final Number of Surviving Halos, Including Satellites:", np.sum((~self.merged) & (~self.stripped)))
		print("Final Number of Surviving Halos, Including Satellites, that aren't minor mergers:", np.sum((~self.merged) & (~self.stripped) & \
		((self.m_halo[self.progToNode]/self.m_halo[self.progenitor[self.parent[self.progToNode]]] >= self.majorMergerMassRatio) | (self.parent[self.progToNode]==-1))))
		if self.useDelayTimes:
			print("The mean and median merger probabilities were: {0} and {1}".format(np.mean(self.allMergerProbabilities),np.median(self.allMergerProbabilities)))
		if np.any(self.progToNode[self.bhToProg[(~self.wandering) & (~self.merged_bh) & (self.scheduledMergeTime==0)]] == 0):
			print("The host halo has a central black hole.")
		else:
			print("The host halo is missing a central black hole.")

		print("######################")
		print("##LARGEST BLACK HOLE##")
		print("######################")
		unmergedIndices = np.where(~self.merged_bh)[0]
		largestIndex = np.argmax(self.m_bh[~self.merged_bh])
		print("The most massive BH lives in progenitor {0}, at node {1}.".format(self.bhToProg[unmergedIndices[largestIndex]], self.progToNode[self.bhToProg[unmergedIndices[largestIndex]]]))
		print("Log Mass of Largest BH:", np.log10(self.m_bh[unmergedIndices[largestIndex]]))
		print("Fraction of Mass Gained in BH Mergers:", self.m_merged[unmergedIndices[largestIndex]] / self.m_bh[unmergedIndices[largestIndex]])
		print("Fraction of Mass Gained in Burst Mode:", self.m_burst[unmergedIndices[largestIndex]] / self.m_bh[unmergedIndices[largestIndex]])
		print("Fraction of Mass Gained in Steady Mode:", self.m_steady[unmergedIndices[largestIndex]] / self.m_bh[unmergedIndices[largestIndex]])
		print("From the M-sigma relation, Log Largest BH Mass should have been:", self.MsigmaNormalization + self.MsigmaSlope * np.log10(self.sigma[0]/200.0))

	def createSnapshotFolders(self, savedProperties):
		"""
		Create lists to save snapshots
		"""

		self.snapshotDict = {}
		for name in savedProperties:
			if name == 'redshift':
				self.snapshotDict['redshift'] = self.uniqueRedshift[self.savedSteps]
			elif name == 'time':
				self.snapshotDict['time'] = self.uniqueTime[self.savedSteps]
			else:
				self.snapshotDict[name+'_snap'] = []

	def saveSnapshot(self, savedProperties, saveMode):
		"""
		Save current state of SAM.
		"""

		#Take a snapshot.  Note that we are excluding stripped BHs.
		if saveMode == 'progenitors':
			#Cut within progenitor space.  Don't keep progenitors that are involved in low mass ratio mergers.
			isMinorMerger = (self.satelliteToCentral!=-1) & (self.m_halo[self.progToNode]/self.m_halo[self.progenitor[self.parent[self.progToNode]]] < self.majorMergerMassRatio)
			cut, = np.where((~self.merged) & (self.time[self.progToNode] <= self.uniqueTime[self.step]) & \
			(~self.stripped) & ~isMinorMerger)
			for name in savedProperties:
				if name == 'indices':
					self.snapshotDict['indices_snap'].append(cut.tolist())
				elif name == 'redshift':
					continue
				elif name == 'time':
					continue
				else:
					desiredArray = self.__dict__[name]
					if name in self.blackHoleProperties:
						#Need to find BHs that match up with each progenitor.  Not using findMatchingBlackHolesToProgenitor because I'm allowing zeros.
						constructedArray = np.zeros(cut.shape, dtype=desiredArray.dtype)
						validBHs = np.where((~self.merged_bh) & (~self.wandering) & (self.scheduledMergeTime==0))[0]
						matchesAmongValid = crossmatch(self.bhToProg[validBHs], cut)

						#If this is not true, something went wrong with matching.  Will need to change when binaries are enabled one day.
						assert matchesAmongValid[0].shape == matchesAmongValid[1].shape
						constructedArray[matchesAmongValid[1]] = desiredArray[validBHs[matchesAmongValid[0]]]
						self.snapshotDict[name+'_snap'].append(constructedArray.tolist())
					elif name in self.progenitorProperties:
						self.snapshotDict[name+'_snap'].append(desiredArray[cut].tolist())
					elif name in self.nodeProperties:
						self.snapshotDict[name+'_snap'].append(desiredArray[self.progToNode[cut]].tolist())
		elif saveMode == 'mainProgenitor':
			#Only save the single values for the main progenitor branch.
			cut, = np.array([self.mainProgenitorIndex])
			for name in savedProperties:
				if name == 'indices':
					self.snapshotDict['indices_snap'].append(cut.tolist())
				elif name == 'redshift':
					continue
				elif name == 'time':
					continue
				else:
					desiredArray = self.__dict__[name]
					if name in self.blackHoleProperties:
						#Need to find BHs that match up with each progenitor.  Not using findMatchingBlackHolesToProgenitor because I'm allowing zeros.
						constructedArray = np.zeros(cut.shape, dtype=desiredArray.dtype)
						validBHs = np.where((~self.merged_bh) & (~self.wandering) & (self.scheduledMergeTime==0))[0]
						matchesAmongValid = crossmatch(self.bhToProg[validBHs], cut)

						#If this is not true, something went wrong with matching.  Will need to change when binaries are enabled one day.
						assert matchesAmongValid[0].shape == matchesAmongValid[1].shape
						constructedArray[matchesAmongValid[1]] = desiredArray[validBHs[matchesAmongValid[0]]]
						self.snapshotDict[name+'_snap'].append(constructedArray.tolist())
					elif name in self.progenitorProperties:
						self.snapshotDict[name+'_snap'].append(desiredArray[cut].tolist())
					elif name in self.nodeProperties:
						self.snapshotDict[name+'_snap'].append(desiredArray[self.progToNode[cut]].tolist())

		else:
			#Cut within BH space
			isMinorMerger = (self.satelliteToCentral[self.bhToProg]!=-1) & (self.m_halo[self.progToNode[self.bhToProg]]/self.m_halo[self.progenitor[self.parent[self.progToNode[self.bhToProg]]]] < self.majorMergerMassRatio)
			cut, = np.where((~self.merged_bh) & (~self.wandering) & (~self.merged[self.bhToProg]) & \
			(self.time[self.progToNode[self.bhToProg]] <= self.uniqueTime[self.step]) & \
			(~self.stripped[self.bhToProg]) & ~isMinorMerger)
			for name in savedProperties:
				if name == 'indices':
					self.snapshotDict['indices_snap'].append(self.bhToProg[cut].tolist())
				elif name == 'redshift':
					continue
				elif name == 'time':
					continue
				else:
					desiredArray = self.__dict__[name]
					if name in self.blackHoleProperties:
						self.snapshotDict[name+'_snap'].append(desiredArray[cut].tolist())
					elif name in self.progenitorProperties:
						self.snapshotDict[name+'_snap'].append(desiredArray[self.bhToProg[cut]].tolist())
					elif name in self.nodeProperties:
						self.snapshotDict[name+'_snap'].append(desiredArray[self.progToNode[self.bhToProg[cut]]].tolist())

	def outputMergers(self):
		return self.bh_mergers

	def saveOutput(self, outputNameBase=None):
		"""
		Turn snapshots into output
		"""
		
		if outputNameBase is not None:
			#Pickle snapshots.
			if not self.silent:
				print("Saving to "+outputNameBase+'.pklz')
			with gzip.open(outputNameBase+'.pklz', 'wb') as myfile:
				pickle.dump(self.snapshotDict, myfile, protocol=2)
		else:
			#Return all of the lists
			return self.snapshotDict

	def evolveUniverse(self, outputNameBase=None, savedRedshifts=None, savedProperties=['m_bh', 'm_halo', 'redshift'], saveMode='progenitors'):
		"""
		Evolve BHs forward in time.
		"""

		#The slices, or those closest to it, where output is saved.
		if savedRedshifts is None:
			self.savedSteps = range(len(self.uniqueRedshift))
		else:
			#Put them in descending order so that weird things don't happen.
			savedRedshifts.sort()
			savedRedshifts.reverse()
			self.savedSteps = [np.argmin(np.abs(self.uniqueRedshift - z)) for z in savedRedshifts]

		#Make lists for output.
		self.createSnapshotFolders(savedProperties)

		#Randomly select some progenitors to be metal polluted, and thus ineligible for seeding.
		if self.useMetalConstraints:
			self.metalPolluted = sf.determinePollution(self.m_halo[self.progToNode], self.redshift[self.progToNode], seedingEfficiency=self.seedingEfficiency)
		else:
			self.metalPolluted = (np.random.random(self.n_prog) > self.seedingEfficiency)

		#Define velocity dispersion
		if self.isothermalSigma:
			self.sigma = sf.v2sigma_SIS(sf.M2v(self.m_halo, self.redshift))
		else:
			self.sigma = sf.velocityDispersion(self.m_halo, self.redshift)
		self.nodeProperties.append('sigma')

		#I'm just using this to get a handle on the merger rate.  Not important if you want to remove all traces of it.
		if self.useDelayTimes:
			self.allMergerProbabilities = []

		#Find out the index of the main progenitor, if desired.
		if saveMode == 'mainProgenitor':
			mpNode = 0
			while True:
				previousNode = self.progenitor[mpNode]
				if previousNode == -1:
					break
				else:
					mpNode = previousNode
			self.mainProgenitorIndex, = np.where(self.progToNode == mpNode)

		#Then, we begin stepping through time
		for self.step in range(self.nSteps):

			###################
			##UPDATE THE TIME##
			###################

			time = self.uniqueTime[self.step]
			previousTime = self.uniqueTime[max(self.step-1,0)]
			if not self.silent:
				print("t_ABB = {0:3.2f} Gyr, z={1:4.2f}.".format(time, sf.t2z([time])[0]))

			#Ignore any progenitors that are stripped, merged, or not formed yet.
			isRelevant = (~self.stripped) & (~self.merged) & (self.time[self.progToNode] <= time)

			#Check for new black hole seeds
			if (self.uniqueRedshift[self.step] > self.minimumSeedingRedshift) & (self.uniqueRedshift[self.step] <= self.maximumSeedingRedshift):
				self.seed(isRelevant & ~self.metalPolluted)

			#NOTE:  This is a place to make new accretion recipes! It may be as simple as just making a new bound method if you don't need extra infrastructure.

			#Determine the accretion recipe we're currently using.
			#TODO:  Make this cleaner
			if self.uniqueRedshift[self.step] > self.z_superEdd:
				integrateAll = self.integrateAll_superEdd
			elif self.useColdReservoir & ~self.useMsigmaCap:
				integrateAll = self.integrateAll_budget_nocap_nodecline
			elif ~self.useColdReservoir & self.useMsigmaCap:
				integrateAll = self.integrateAll_nobudget_cap_nodecline
			elif self.useColdReservoir & self.useMsigmaCap:
				if ~self.includeDecline:
					integrateAll = self.integrateAll_budget_cap_nodecline
				else:
					integrateAll = self.integrateAll_budget_cap_decline
			else:
				raise RuntimeError("This mode isn't implemented.  Write your own bound function for this.")

			if self.step > 0:

				#############################
				##TRANSFER MAIN PROGENITORS##
				#############################

				#Time must be checked because some progenitors pop into existence on the current time step.
				transferring = isRelevant & (self.progenitor[self.parent[self.progToNode]] == self.progToNode) & (self.time[self.progToNode]<time)
				self.progToNode[transferring] = self.parent[self.progToNode[transferring]]

				#########################
				##ASSIGN NEW SATELLITES##
				#########################

				#Time must be checked because some progenitors pop into existence on the current time step.
				newSatellites, = np.where(isRelevant & (~transferring) & (self.satelliteToCentral==-1) & (self.time[self.progToNode]<time) & (self.nchild[self.parent[self.progToNode]]>1))
				if len(newSatellites) > 0:
					newSatelliteParentNodes = self.parent[self.progToNode[newSatellites]]
					satelliteMatcher = crossmatch(newSatelliteParentNodes, self.progToNode[isRelevant])
					newCentrals = np.where(isRelevant)[0][satelliteMatcher[1]]
					newSatellites = newSatellites[satelliteMatcher[0]]
					self.assignSatellites(newSatellites, newCentrals)

				######################################
				##INTEGRATE SUB-TIME STEP IN A QUEUE##
				######################################

				#Here, we iterate to make sure things happen in the right order.  Sometimes events cause other events, so we need to keep doing this on the fly.

				while True:

					#######################
					##FIND FEEDING EVENTS##
					#######################

					feedingHalos, = np.where(isRelevant & (self.feedTime <= time))
					#feedingTargets is just a dummy.  This list isn't used for anything, but it's important that it doesn't match any other halo numbers.
					feedingTargets = -feedingHalos-1
					feedingTimes = self.feedTime[feedingHalos]

					#####################
					##FIND HALO MERGERS##
					#####################

					mergingHalos, = np.where(isRelevant & (self.haloMergerTime <= time))
					majorMergingTargets = self.satelliteToCentral[mergingHalos]
					majorMergingTimes = self.haloMergerTime[mergingHalos]

					###########################
					##FIND BLACK HOLE MERGERS##
					###########################
					
					if self.useDelayTimes:
						holesToMerge, = np.where((self.scheduledMergeTime <= time) & (self.scheduledMergeTime > 0))
						holeTargets = self.bhToProg[holesToMerge]
						holeMergingTimes = self.scheduledMergeTime[holesToMerge]
					else:
						holesToMerge = []
						holeTargets = []
						holeMergingTimes = []

					#Make sure there's something to do.  If there isn't, you're done.
					if len(mergingHalos) + len(holesToMerge) + len(feedingHalos) == 0:
						break

					#Create a combined queue and order it.
					queueTimes = np.concatenate((majorMergingTimes,holeMergingTimes,feedingTimes))
					queueIndices = np.concatenate((mergingHalos,holesToMerge,feedingHalos))
					queueTargets = np.concatenate((majorMergingTargets,holeTargets,feedingTargets))
					queueEventTypes = np.concatenate((np.full(len(majorMergingTimes), 'haloMerger'), \
					np.full(len(holeMergingTimes), 'blackHoleMerger'), \
					np.full(len(feedingHalos), 'feeding')))
					
					eventOrder = np.argsort(queueTimes)
					queueTimes = queueTimes[eventOrder]
					queueIndices = queueIndices[eventOrder].astype(int)
					queueTargets = queueTargets[eventOrder].astype(int)
					queueEventTypes = queueEventTypes[eventOrder]

					#This is an array that I must use to help sort the queue.  Whenever a black hole is present, have a 1 instead of 0. 
					isABlackHole = np.full(len(eventOrder), 0, dtype=int)
					isABlackHole[queueEventTypes=='blackHoleMerger'] = 1

					#######################
					##LOOK FOR DUPLICATES##
					#######################

					#If the same object appears twice, stop so that everything is taken care of in the right order.
					nextDuplicate = findFirstDuplicate2_2d(np.transpose(np.vstack((queueIndices, isABlackHole))), \
					np.transpose(np.vstack((queueTargets, np.zeros(len(queueTargets),dtype=int)))))
					if nextDuplicate == -1:
						chunkEnd = len(queueTimes)
					else:
						chunkEnd = nextDuplicate

					#Gather all events
					chunkTimes = queueTimes[:chunkEnd]
					chunkIndices = queueIndices[:chunkEnd]
					chunkTargets = queueTargets[:chunkEnd]
					chunkEventTypes = queueEventTypes[:chunkEnd]

					#First, let's give gas to any halos to which it is owed.
					isFeedingEvent = chunkEventTypes == 'feeding'
					if np.any(isFeedingEvent):
						feedingHalos = chunkIndices[isFeedingEvent]
						feedingTimes = chunkTimes[isFeedingEvent]
						if self.supernovaBarrier:
							#See if SN feedback prevents AGN from triggering.
							Mcrit = sf.Mcrit(self.uniqueRedshift[self.step])
							aboveSpecialHaloMass = self.m_halo[self.progToNode[feedingHalos]] > Mcrit
							self.feedTime[feedingHalos[~aboveSpecialHaloMass]] = np.inf
							feedingHalos = feedingHalos[aboveSpecialHaloMass]

						#Find corresponding central black holes, if they exist
						matchedProgenitors, feedingHoles = self.findMatchingBlackHolesToProgenitor(feedingHalos)
						if len(feedingHoles) > 0:
							#Integrate any BHs up to this point before we change their host conditions.
							timeMatcher = crossmatch(matchedProgenitors, chunkIndices)
							feedingHoles = feedingHoles[timeMatcher[0]]
							time_stop = chunkTimes[timeMatcher[1]]
							timeSteps = time_stop - self.lastIntegrationTime[feedingHoles]
							positiveTimes = timeSteps > 0
							nonwanderers = (~self.wandering[feedingHoles]) & (~self.merged_bh[feedingHoles])
							toIntegrate = positiveTimes & nonwanderers
							if np.any(toIntegrate):
								integrateAll(feedingHoles[toIntegrate], timeSteps[toIntegrate], time_stop[toIntegrate])

							#Do some feeding.
							if self.useColdReservoir:
								#Give the fuel that let's you get to M-sigma (of this instant)
								self.accretionBudget[feedingHoles] += np.maximum(self.capFudgeFactor * self.M_bhSigma(matchedProgenitors) - self.m_bh[feedingHoles], 0)
								if self.constant_f_max is not None:
									self.eddRatio[feedingHoles] = np.full(len(feedingHoles), self.constant_f_max)
								else:
									#Push Eddington ratio to something appropriate for a Type I AGN.
									self.eddRatio[feedingHoles] = sf.draw_typeI(len(feedingHoles), np.full(len(feedingHoles), self.uniqueRedshift[self.step]))
							else:
								self.mode[feedingHoles] = 'quasar'
						self.feedTime[feedingHalos] = np.inf

					#Next, let's do all the black hole mergers
					isBHMerger = chunkEventTypes == 'blackHoleMerger'
					if np.any(isBHMerger):
						#Integrate all relevant BHs up to this point.  Including any merging BHs in this list.
						progenitorsWithHoles, associatedBHs = self.findMatchingBlackHolesToProgenitor(chunkTargets[isBHMerger], includeMerging=True)
						timeMatcher = crossmatch(progenitorsWithHoles, chunkTargets[isBHMerger])
						associatedBHs = associatedBHs[timeMatcher[0]]
						time_stop = chunkTimes[timeMatcher[1]]
						timeSteps = time_stop - self.lastIntegrationTime[associatedBHs]
						positiveTimes = timeSteps > 0
						nonwanderers = (~self.wandering[associatedBHs]) & (~self.merged_bh[associatedBHs])
						toIntegrate = positiveTimes & nonwanderers
						if np.any(toIntegrate):
							integrateAll(associatedBHs[toIntegrate], timeSteps[toIntegrate], time_stop[toIntegrate])
						self.transferBHs(chunkIndices[isBHMerger], chunkTargets[isBHMerger], chunkTimes[isBHMerger])

					#Then, we do all of the halo mergers.
					isHaloMerger = chunkEventTypes == 'haloMerger'
					chunkAllProgenitors = np.concatenate((chunkIndices[isHaloMerger],chunkTargets[isHaloMerger]))
					chunkAllProgenitorMergeTimes = np.concatenate((chunkTimes[isHaloMerger],chunkTimes[isHaloMerger]))

					#Before doing so, locate any BHs that are affected and integrate those too.
					mergingHalosWithHoles, holesInMergingHalos = self.findMatchingBlackHolesToProgenitor(chunkAllProgenitors, includeMerging=True, includeWandering=True)
					if len(holesInMergingHalos) > 0:
						timeMatcher = crossmatch(mergingHalosWithHoles, chunkAllProgenitors)
						holesInMergingHalos = holesInMergingHalos[timeMatcher[0]]
						time_stop = chunkAllProgenitorMergeTimes[timeMatcher[1]]
						timeSteps = time_stop - self.lastIntegrationTime[holesInMergingHalos]
						positiveTimes = timeSteps > 0
						nonwanderers = (~self.wandering[holesInMergingHalos]) & (~self.merged_bh[holesInMergingHalos])
						toIntegrate = positiveTimes & nonwanderers
						if np.any(toIntegrate):
							integrateAll(holesInMergingHalos[toIntegrate], timeSteps[toIntegrate], time_stop[toIntegrate])

						#Transfer BHs to target halos
						bh_homes, bh_indices = self.findMatchingBlackHolesToProgenitor(chunkIndices[isHaloMerger], includeMerging=True, includeWandering=True)
						if len(bh_indices) > 0:
							homeMatcher = crossmatch(bh_homes, chunkIndices[isHaloMerger])
							bh_targets = chunkTargets[isHaloMerger][homeMatcher[1]]
							bh_merge_times = chunkTimes[isHaloMerger][homeMatcher[1]]
							bh_indices = bh_indices[homeMatcher[0]]
							if self.useDelayTimes:
								#Compute the probability that the BHs will eventually merge, then queue it up.
								smhm_redshift = sf.t2z(bh_merge_times)
								stellarMassMerging = self.m_star[self.progToNode[self.bhToProg[bh_indices]]]
								stellarMassTarget = self.m_star[self.progToNode[bh_targets]]
								mergerProbabilities = sf.mergerProbability(stellarMassTarget, stellarMassMerging/stellarMassTarget)
								self.allMergerProbabilities = np.concatenate([self.allMergerProbabilities,mergerProbabilities])
								#Set the merger probability to 1 if the target doesn't currently have a black hole.
								mergerProbabilities[~np.in1d(bh_targets, self.bhToProg[(~self.wandering) & (~self.merged_bh)])] = 1

								toEventuallyMerge = np.random.random(len(bh_indices)) < mergerProbabilities
								if np.any(toEventuallyMerge):
									self.scheduleBHTransfer(bh_indices[toEventuallyMerge], bh_targets[toEventuallyMerge], bh_merge_times[toEventuallyMerge])
								if np.any(~toEventuallyMerge):
									self.createWanderers(bh_indices[~toEventuallyMerge], bh_targets[~toEventuallyMerge])
							else:
								if self.mergerMode == 'flatProbability':
									toMerge = np.random.random(len(bh_indices)) < self.blackHoleMergerProbability
								elif self.mergerMode == 'fabioProbability':
									stellarMassMerging = self.m_star[self.progToNode[self.bhToProg[bh_indices]]]
									stellarMassTarget = self.m_star[self.progToNode[bh_targets]]
									computedMergerProbabilities = fabioMergerProbabilities(stellarMassTarget, stellarMassMerging/stellarMassTarget)
									toMerge = np.random.random(len(bh_indices)) < computedMergerProbabilities
								#Set the merger probability to 1 if the target doesn't currently have a black hole.  TODO:  Do we really want this?
								toMerge[~np.in1d(bh_targets, self.bhToProg[(~self.wandering) & (~self.merged_bh)])] = True
								if np.any(toMerge):
									self.transferBHs(bh_indices[toMerge], bh_targets[toMerge], bh_merge_times[toMerge])
								if np.any(~toMerge):
									self.createWanderers(bh_indices[~toMerge], bh_targets[~toMerge])

					#Finally merge those halos.
					self.mergeHalo(chunkIndices[isHaloMerger], chunkTargets[isHaloMerger])
					self.lastMajorMerger[chunkTargets[isHaloMerger]] = chunkTimes[isHaloMerger]
				
					#Reevaluate after halo mergers are complete
					isRelevant = (~self.stripped) & (~self.merged) & (self.time[self.progToNode] <= time)

				#Recalculate after halo mergers are complete
				isRelevant = (~self.stripped) & (~self.merged) & (self.time[self.progToNode] <= time)
				relevantBHs = (~self.merged_bh) & (self.seedTime <= time) & (~self.wandering)

				##########################################
				##INTEGRATE ALL TIME-DEPENDENT EQUATIONS##
				##########################################

				#Integrate all equations
				bh_indices = np.where(relevantBHs)[0]
				timeSteps = np.full(len(bh_indices), time) - self.lastIntegrationTime[bh_indices]
				positiveTimes = timeSteps > 0
				nonwanderers = (~self.wandering[bh_indices]) & (~self.merged_bh[bh_indices])
				toIntegrate = positiveTimes & nonwanderers
				if np.any(toIntegrate):
					integrateAll(bh_indices[toIntegrate], timeSteps[toIntegrate], np.full(np.sum(toIntegrate), time))

				#Reform disks in any halos that have not had any mergers in 5 Gyr and have v<300 km/s
				if self.makeEllipticals:
					newDisks = np.where(isRelevant & (self.elliptical) & (time-self.lastMajorMerger >= 5) & \
					(sf.v_peak(self.m_halo[self.progToNode], self.redshift[self.progToNode])<300))[0]
					self.elliptical[newDisks] = False

			####################
			##FILL FEEDINGLIST##
			####################

			if (self.step != self.nSteps-1) & (self.uniqueRedshift[self.step] < self.z_superEdd):
				#Find halos whose mass have changed by a large fraction and haven't just grown by smooth accretion.
				feedingListMask = isRelevant & (self.progToNode != self.progenitor[self.parent[self.progToNode]]) & (self.satelliteToCentral!=-1) & \
				(self.time[self.progToNode]<time) & (self.nchild[self.parent[self.progToNode]]>1) & (self.m_halo[self.progToNode]/self.m_halo[self.progenitor[self.parent[self.progToNode]]] >= self.majorMergerMassRatio)
				halosToFeed, = np.where(feedingListMask)

				#Find the merging partners and feed them too.
				partnersToFeed = self.satelliteToCentral[halosToFeed]
				if self.makeEllipticals:
					#To feed, at least one must be non-elliptical so that this isn't a dry merger.
					passesEllipticalCriterion = (~self.elliptical[halosToFeed]) | (~self.elliptical[partnersToFeed])
					halosToFeed = halosToFeed[passesEllipticalCriterion]
					partnersToFeed = partnersToFeed[passesEllipticalCriterion]

				#Note:  With isothermal spheres, the dynamical time depends only on redshift, not mass.
				feedingTime = time + sf.t_dyn(self.uniqueRedshift[self.step])
				#Looping through slowly, just in case there are duplicates in partnersToFeed.
				for i in range(len(halosToFeed)):
					self.feedTime[halosToFeed[i]] = np.minimum(feedingTime, self.feedTime[halosToFeed[i]])
					self.feedTime[partnersToFeed[i]] = np.minimum(feedingTime, self.feedTime[partnersToFeed[i]])

			########
			##SAVE##
			########

			if self.step in self.savedSteps:
				self.saveSnapshot(savedProperties, saveMode)

		#Print some diagnostics
		if not self.silent:
			self.printDiagnostics()

		#Neatly wrap up output into a dictionary.
		return self.saveOutput(outputNameBase=outputNameBase)
			
if __name__ == "__main__":
	#Obtain the appropriate merger tree
	m_halo = 1e13
	i_tree = 1
	path = None
	lorax = sf.treeNavigator(path=path)
	treefile = path + '/' + lorax.findClosestTree(m_halo, i_tree)
	parameters = None

	#Run the SAM
	outputNameBase = 'test'
	savedProperties = ['m_bh', 'm_halo', 'indices', 'L_bol', 'eddRatio', 'spin_bh', 'redshift', \
    'elliptical', 'time', 'lastMajorMerger', 'bhToProg', 'bh_id', 'satelliteToCentral', \
    'm_steady', 'm_burst', 'sigma']
	howToSave = 'mainProgenitor'
	savedRedshifts = None

	model = SAM(treefile, parameters=parameters)
	model.evolveUniverse(outputNameBase=outputNameBase, savedRedshifts=savedRedshifts, howToSave=howToSave, savedProperties=savedProperties)
