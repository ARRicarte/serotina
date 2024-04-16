"""
ARR 11.05.15
This program reads in merger tree output and paints black holes onto them.
"""

import numpy as np
from . import cosmology
from . import cosmology_functions
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
		#self.m_smooth = mergerTreeTable[:,5]  #This is in the file, but it's not needed.
		self.n_node = mergerTreeTable.shape[0]
		self.nodeProperties = ['m_halo', 'redshift', 'parent', 'progenitor', 'nchild', 'time']

		#Get a time axis, representing age of the universe.
		if not silent:
			print("Calculating times and stellar masses.")
		self.uniqueRedshift = np.flipud(np.unique(self.redshift))
		self.uniqueTime = cosmology_functions.z2t(self.uniqueRedshift)		#Gyr
		self.time = np.zeros(self.n_node)
		for lev in range(len(self.uniqueRedshift)):
			timeAtLev = self.uniqueTime[len(self.uniqueTime)-1-lev]
			self.time[self.redshift==self.uniqueRedshift[len(self.uniqueTime)-1-lev]] = timeAtLev		   

		#Stellar masses
		self.m_star = smhm.Mstar(self.m_halo, self.redshift)
		self.nodeProperties.append('m_star')

		#Set up storage.  Black holes move among progenitors, which move along nodes.  If you ever want to start over, use self.initializeStorage()
		if not silent:
			print("Initializing storage arrays.")
		self.initializeStorage(keepSpins=False)
		
		#if not silent:
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
		'useMsigmaCap': True, 'fixedSeedMass': None, 'silent': False, 'randomFractionOn': 0.0, 'alignMergerSpins': False, 'seedingSigma': 3.5, \
		'useColdReservoir': False, 'f_superEdd': 5.0, 'makeEllipticals': False, 'includeDecline': False, 'f_EddCrit': None, \
		'minimumSeedingRedshift': 15, 'maximumSeedingRedshift': 20, 'useMetalConstraints': False, 'supernovaBarrier': False, 'blackHoleMergerProbability': 1.0, \
		'steady': None, 'isothermalSigma': False, 'defaultRadiativeEfficiency': 0.1, 'Q_dcbh': 3.0, 'nscMassThreshold': 1e8, 'mergerMode': 'flatProbability', \
		'diskAlignmentParameters': [np.inf], 'MAD': False, 'violentMergers': False, 'alwaysMergeEmpty': True}

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
		self.scheduledFlipTime = np.empty(0)

		self.blackHoleProperties = ['bh_id', 'm_bh', 'm_init', 'm_merged', 'm_burst', 'm_steady', 'spin_bh', 'seedType', 'bhToProg', \
		'seedTime', 't_decline', 't_fEdd', 'mode', 'accretionBudget', 'merged_bh', 'wandering', 'eddRatio', 'L_bol', 'lastIntegrationTime', \
		'scheduledMergeTime', 'scheduledFlipTime']

		#This special list is populated with M1, q, and z whenever a black hole merger occurs.  Remnant spin and chi_eff are also included if spin evolution is on.
		self.bh_mergers = np.empty((0,5))

	def seed(self, relevantProgenitors, m_d=0.05, alpha_c=0.06, T_gas=5000, j_d=0.05, noProgenitor=False, noMergers=False):
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
			seedIndices = potentialSeedIndices[sigmaPeaks > self.seedingSigma]

			#Make sure you're not adding a PopIII seed to something that already has a DCBH seed.
			realSeedIndices = seedIndices[~np.in1d(seedIndices, seededProgenitors)]
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
			if self.spinEvolution:
				redshifts = self.redshift[self.progToNode[np.array(seededProgenitors)]]
				times = cosmology_functions.z2t(redshifts)
				newFlipTimes = times + sf.computeAccretionAlignmentFlipTime(np.array(seedMasses), redshifts, parameters=self.diskAlignmentParameters)
				self.scheduledFlipTime = np.concatenate((self.scheduledFlipTime, newFlipTimes))
			else:
				self.scheduledFlipTime = np.concatenate((self.scheduledFlipTime, np.zeros(n_new)))
		
	def mergeBHs(self, primaries, secondaries, progenitors, times):
		"""
		Merge two black holes.  Keeping the properties of the first index.
		"""

		npts = len(primaries)

		if (self.mergerKicks) | (self.spinEvolution):

			if self.alignMergerSpins:
				#One might expect this in gas rich environments, if the binaries shrunk in a gas disc.
				theta1 = np.full(npts, 0.0)
				theta2 = np.full(npts, 0.0)
				phi1 = np.full(npts, 0.0)
				phi2 = np.full(npts, 0.0)
			else:
				#One might expect this in gas poor environments.
				theta1 = np.pi*np.random.random(npts)
				theta2 = np.pi*np.random.random(npts)
				phi1 = 2*np.pi*np.random.random(npts)
				phi2 = 2*np.pi*np.random.random(npts)

		if self.spinEvolution:
			#Mass-weighted spin projected onto the binary axis, most easily accessed by GW detectors.
			chi_eff = (self.m_bh[primaries] * np.abs(self.spin_bh[primaries]) * np.cos(theta1) + self.m_bh[secondaries] * np.abs(self.spin_bh[secondaries]) * np.cos(theta2)) / (self.m_bh[primaries] + self.m_bh[secondaries])

			#Note that only positive spins go into this formula.
			signToPreserve = np.sign(self.spin_bh[primaries])
			self.spin_bh[primaries] = bhb.calcRemnantSpin(self.m_bh[primaries], self.m_bh[secondaries], \
			np.abs(self.spin_bh[primaries]), np.abs(self.spin_bh[secondaries]), \
			theta1=theta1, theta2=theta2, phi1=phi1, phi2=phi2, spinMax=self.spinMax)
		else:
			chi_eff = np.zeros(npts, dtype=float)
		
		#Save to a list of all merger events.
		self.bh_mergers = np.vstack((self.bh_mergers, np.array([self.m_bh[primaries], self.m_bh[secondaries]/self.m_bh[primaries], cosmology_functions.t2z(times), self.spin_bh[primaries], chi_eff]).transpose()))

		#Simply adding the two masses together.
		self.m_bh[primaries] += self.m_bh[secondaries]
		self.m_merged[primaries] += self.m_bh[secondaries]
		self.seedType[primaries] = 'Merged'
		self.bhToProg[primaries] = progenitors

		if self.spinEvolution:
			#By default, maintain prograde or retrograde nature of the primary.  That means re-flip anything that was negative.
			self.spin_bh[primaries][signToPreserve == -1] *= -1

			if self.violentMergers:
				#In principle, I think I should actually be able to get this from the GR formulae, but for now, option for a 50% chance of a flip.
				flipping = np.random.choice([True,False], size=len(primaries))
				self.spin_bh[primaries] = self.spin_bh[primaries] * np.random.choice([1,-1], size=len(primaries))

		#Now let's see if the kick is large enough to cause it to leave the halo.
		if self.mergerKicks:
			mergerKicks = bhb.calcRemnantKick(self.m_bh[primaries], self.m_bh[secondaries], self.spin_bh[primaries], self.spin_bh[secondaries], theta1=theta1, theta2=theta2, phi1=phi1, phi2=phi2)
			
			#Compare kick velocity to Choksi formula
			#Note:  Using the escape velocity here is pretty unfair.  In reality, it depends on baryonic distribution, dark matter profile, etc., all of which are time and orbit dependent.
			kicked = (mergerKicks > sf.calcRecoilEscapeVelocity_permanent(self.m_halo[self.progToNode[progenitors]], cosmology_functions.t2z(times)))
			#Set to wandering, halt accretion.
			if np.any(kicked):
				self.createWanderers(primaries[kicked], progenitors[kicked])
		else:
			kicked = np.zeros(npts, dtype=bool)

		#Get rid of the secondary, which no longer exists.
		self.eliminateBH(secondaries)

	def eliminateBH(self, bh_ids):
		"""
		Used when a secondary merges into another black hole and no longer exists
		"""

		self.bhToProg[bh_ids] = -1
		self.merged_bh[bh_ids] = True
		self.scheduledMergeTime[bh_ids] = 0
		self.scheduledFlipTime[bh_ids] = 0
		self.accretionBudget[bh_ids] = 0
		self.t_decline[bh_ids] = 0
		self.mode[bh_ids] = ''

	def createWanderers(self, bh_ids, targets):
		"""
		Keep the primaries in their progenitors, set secondaries to wandering.  Used when a BH does not merge, or if it is kicked via GW recoil.
		"""

		self.bhToProg[bh_ids] = targets
		self.wandering[bh_ids] = True
		self.scheduledMergeTime[bh_ids] = 0
		self.scheduledFlipTime[bh_ids] = 0
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

		lostProgenitors = np.atleast_1d(lostProgenitors)
		targets = np.atleast_1d(targets)

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

		if self.violentMergers:
			#50% chance that disks of centrals flip.
			matchedTargets, matchedBHs = self.findMatchingBlackHolesToProgenitor(targets)
			if len(matchedBHs) > 0:
				self.spin_bh[matchedBHs] = self.spin_bh[matchedBHs] * np.random.choice([1,-1], size=len(matchedBHs))

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

	def integrateAll_nobudget_cap_nodecline(self, bhs, timeSteps, times):
		"""
		~self.useColdReservoir & self.useMsigmaCap & ~self.includeDecline
		"""

		#Determine which black holes are accreting
		if self.constant_f_min is None:
			accreting = np.ones_like(self.mode[bhs], dtype=bool)
		elif self.constant_f_min > 0:
			accreting = np.ones_like(self.mode[bhs], dtype=bool)
		else:
			accreting = self.mode[bhs] == 'quasar'
		accretors = bhs[accreting]
		nonAccretors = bhs[~accreting]
		redshifts = cosmology_functions.t2z(times)
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
				f_EddMax[self.mode[accretors]=='quasar'] = sf.draw_typeI(np.sum(self.mode[accretors]=='quasar'), cosmology_functions.t2z(times[accreting][self.mode[accretors]=='quasar']))
			if self.steady == None:
				if self.constant_f_min is None:
					f_EddMin = np.zeros(sum(accreting))
				elif self.constant_f_min >= 0:
					f_EddMin = np.full(sum(accreting), self.constant_f_min)
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

			#The actual integration step.
			self.m_bh[accretors], self.spin_bh[accretors], self.L_bol[accretors], self.eddRatio[accretors], growthFromBurst, growthFromSteady = \
			acc.accretionDualMode(self.m_bh[accretors], self.spin_bh[accretors], timeSteps[accreting], times[accreting], f_EddMax, f_EddMin, 
			f_EddCrit=self.f_EddCrit, includeSpinDependence=self.spinEvolution, maxBurstMass=massLimitsQuasar, maxSteadyMass=massLimitsSteady, spinMax=self.spinMax, \
			fiducialRadiativeEfficiency=self.defaultRadiativeEfficiency, MAD=self.MAD)

			#Change mode if necessary
			finishedQuasars = self.m_bh[accretors] >= massLimitsQuasar
			self.mode[accretors[finishedQuasars]] = ''

			#Keep track of what was gained from the burst mode and the steady mode.
			self.m_burst[accretors] += growthFromBurst
			self.m_steady[accretors] += growthFromSteady

		#Make sure that the non-accretors have 0 Eddington ratio and luminosity
		if np.any(~accreting):
			self.L_bol[nonAccretors] = 0
			self.eddRatio[nonAccretors] = 0
		self.lastIntegrationTime[bhs] = times

	def integrateBHs(self, bhs, endTime, integratingMethod):
		"""
		Integrate a list of BHs up to some end time.
		"""

		bhs = np.atleast_1d(bhs)
		timeSteps = endTime - self.lastIntegrationTime[bhs]
		nonwanderer = (~self.wandering[bhs]) & (~self.merged_bh[bhs])
		toIntegrate = nonwanderer & (timeSteps>0)
		if np.any(toIntegrate):
			integratingMethod(bhs[toIntegrate], timeSteps[toIntegrate], np.full(np.sum(toIntegrate), endTime))

	def integrateBHsOfProgenitor(self, progenitors, endTime, integratingMethod, includeMerging=True, includeWandering=False):
		"""
		Integrate all of the BHs of a progenitor, or list of progenitors
		"""

		progenitors = np.atleast_1d(progenitors)
		matchedProgenitors, associatedBHs = self.findMatchingBlackHolesToProgenitor(progenitors, includeMerging=includeMerging, includeWandering=includeWandering)
		if len(associatedBHs) > 0:
			self.integrateBHs(associatedBHs, endTime, integratingMethod)

	def findMatchingBlackHolesToProgenitor(self, progenitorIndices, includeCentral=True, includeWandering=False, includeMerging=False):
		"""
		Given a list of progenitor indices, match them up with their associated black holes.
		"""

		progenitorIndices = np.atleast_1d(progenitorIndices)

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
		if len(bh_matches[0]) > 0:
			return np.atleast_1d(progenitorIndices[bh_matches[1]]), np.atleast_1d(relevantBHs[bh_matches[0]])
		else:
			return np.array([]), np.array([])

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
		print("Spin of Largest BH:", self.spin_bh[unmergedIndices[largestIndex]])
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

	def findUniverseInterruptions(self, time):
		"""
		Sometimes things happen in between timesteps.  Find them for evaluation.
		"""

		#Ignore any progenitors that are stripped, merged, or not formed yet.
		isRelevant = (~self.stripped) & (~self.merged) & (self.time[self.progToNode] <= time)

		relevantHalosOrBlackHoles = []
		times = []
		types = []

		#######################
		##FIND FEEDING EVENTS##
		#######################

		isFeeding = isRelevant & (self.feedTime <= time)
		if np.any(isFeeding):
			relevantHalosOrBlackHoles.extend(np.transpose(np.vstack([np.where(isFeeding)[0], np.full(np.sum(isFeeding),0)])))
			times.extend(self.feedTime[isFeeding])
			types.extend(['feeding']*np.sum(isFeeding))

		#####################
		##FIND HALO MERGERS##
		#####################

		isMerging = isRelevant & (self.haloMergerTime <= time)
		if np.any(isMerging):
			mergingHalos, = np.where(isMerging)
			majorMergingTargets = self.satelliteToCentral[mergingHalos]
			relevantHalosOrBlackHoles.extend([[mergingHalos[i],majorMergingTargets[i]] for i in range(len(majorMergingTargets))])
			times.extend(self.haloMergerTime[mergingHalos])
			types.extend(['haloMerger']*np.sum(isMerging))

		###################
		##FIND DISK FLIPS##
		###################

		if self.spinEvolution:
			#Unlike this other items, this one using the black hole index.
			isFlipping = (self.scheduledFlipTime <= time) & (self.scheduledFlipTime > 0) & (self.spin_bh != 0)
			if np.any(isFlipping):
				relevantHalosOrBlackHoles.extend(np.transpose(np.vstack([np.where(isFlipping)[0], np.full(np.sum(isFlipping),0)])))
				times.extend(self.scheduledFlipTime[isFlipping])
				types.extend(['diskFlip']*np.sum(isFlipping))
			
		###########################
		##FIND BLACK HOLE MERGERS##
		###########################

		if self.useDelayTimes:
			#Otherwise, this has already happened during a halo merger.
			isBHMerging = (self.scheduledMergeTime <= time) & (self.scheduledMergeTime > 0)
			if np.any(isBHMerging):
				holesToMerge, = np.where(isBHMerging)
				holeTargets = self.bhToProg[holesToMerge]
				#Note that the target is a progenitor, not a black hole.
				relevantHalosOrBlackHoles.extend([[holesToMerge[i],holeTargets[i]] for i in range(len(holesToMerge))])
				times.extend(self.scheduledMergeTime[holesToMerge])
				types.extend(['blackHoleMerger']*np.sum(isBHMerging))

		#Wrap up, then order by time.
		times = np.array(times)
		relevantHalosOrBlackHoles = np.array(relevantHalosOrBlackHoles)
		types = np.array(types)
		if len(times) > 0:
			#If multiple events occur at the same time, what order to we do them in?  It probably doesn't matter, but I've made a consistent choice here.
			tieBreakers = np.zeros_like(times)
			tieBreakers[types=='diskFlip'] = 0
			tieBreakers[types=='blackHoleMerger'] = 1
			tieBreakers[types=='haloMerger'] = 2
			tieBreakers[types=='feeding'] = 3			

			#Sort
			arrayToSort = np.array(list(zip(times, tieBreakers)), dtype=[('times', float), ('tieBreakers', int)])
			eventOrder = np.argsort(arrayToSort, order=['times', 'tieBreakers'])
			times = times[eventOrder]
			relevantHalosOrBlackHoles = relevantHalosOrBlackHoles[eventOrder]
			types = types[eventOrder]

		return times, relevantHalosOrBlackHoles, types

	def resolveUniverseInterruption(self, time, relevantHaloOrBlackHole, eventType, integrateAll):
		"""
		Resolve an individual interruption.
		"""

		#Turn on BHs
		if eventType == 'feeding':
			#Note: for this event, several indices are allowed to pass in parallel to save time.
			relevantHaloOrBlackHole = np.atleast_2d(relevantHaloOrBlackHole)

			if self.supernovaBarrier:
				#See if SN feedback prevents AGN from triggering.
				Mcrit = sf.Mcrit(self.uniqueRedshift[self.step])
				aboveSpecialHaloMass = self.m_halo[self.progToNode[relevantHaloOrBlackHole[:,0]]] > Mcrit
				if np.any(aboveSpecialHaloMass):
					relevantHaloOrBlackHole = relevantHaloOrBlackHole[aboveSpecialHaloMass,:]
				else:
					self.feedTime[relevantHaloOrBlackHole[:,0]] = np.inf
					#Never mind; we're done.
					return

			#Find corresponding central black holes, if they exist
			matchedProgenitor, feedingHole = self.findMatchingBlackHolesToProgenitor(relevantHaloOrBlackHole[:,0])
			if len(feedingHole) > 0:
				self.integrateBHs(feedingHole, time, integrateAll)

				#Activate feeding.
				if self.useColdReservoir:
					#Give the fuel that let's you get to M-sigma (of this instant)
					self.accretionBudget[feedingHole] += np.maximum(self.capFudgeFactor * self.M_bhSigma(matchedProgenitor) - self.m_bh[feedingHole], 0)
					if self.constant_f_max is not None:
						self.eddRatio[feedingHole] = np.full(len(feedingHole), self.constant_f_max)
					else:
						#Push Eddington ratio to something appropriate for a Type I AGN.
						self.eddRatio[feedingHole] = sf.draw_typeI(len(feedingHole), np.full(len(feedingHole), self.uniqueRedshift[self.step]))
				else:
					self.mode[feedingHole] = 'quasar'
			self.feedTime[relevantHaloOrBlackHole[:,0]] = np.inf

		#Merge BHs
		elif eventType == 'blackHoleMerger':
			#Integrate all relevant BHs up to this point.  Including any merging BHs in this list.
			self.integrateBHs(relevantHaloOrBlackHole[0], time, integrateAll)
			self.integrateBHsOfProgenitor(relevantHaloOrBlackHole[1], time, integrateAll)
			self.transferBHs(relevantHaloOrBlackHole[0], relevantHaloOrBlackHole[1], time)

		#Halo Mergers
		elif eventType == 'haloMerger':
			self.integrateBHsOfProgenitor(relevantHaloOrBlackHole, time, integrateAll, includeMerging=True, includeWandering=True)

			#Transfer BHs to target halos
			bh_homes, bh_indices = self.findMatchingBlackHolesToProgenitor(relevantHaloOrBlackHole[0], includeMerging=True, includeWandering=True)

			#Either queue up eventual BH mergers or merge the BHs immediately.
			if len(bh_indices) > 0:
				bh_targets = np.full(len(bh_indices), relevantHaloOrBlackHole[1])
				if self.useDelayTimes:
					#Compute the probability that the BHs will eventually merge, then queue it up.
					stellarMassMerging = self.m_star[self.progToNode[self.bhToProg[bh_indices]]]
					stellarMassTarget = self.m_star[self.progToNode[bh_targets]]
					mergerProbabilities = sf.mergerProbability(stellarMassTarget, stellarMassMerging/stellarMassTarget)
					self.allMergerProbabilities = np.concatenate([self.allMergerProbabilities,mergerProbabilities])
					#Set the merger probability to 1 if the target doesn't currently have a black hole.
					mergerProbabilities[~np.in1d(bh_targets, self.bhToProg[(~self.wandering) & (~self.merged_bh)])] = 1

					toEventuallyMerge = np.random.random(len(bh_indices)) < mergerProbabilities
					if np.any(toEventuallyMerge):
						self.scheduleBHTransfer(bh_indices[toEventuallyMerge], bh_targets[toEventuallyMerge], np.full(np.sum(toEventuallyMerge), time))
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
					if self.alwaysMergeEmpty:
						#Set the merger probability to 1 if the target doesn't currently have a black hole.  This especially matters for DCBHs.  Is it a good assumption?
						toMerge[~np.in1d(bh_targets, self.bhToProg[(~self.wandering) & (~self.merged_bh)])] = True
					if np.any(toMerge):
						self.transferBHs(bh_indices[toMerge], bh_targets[toMerge], np.full(np.sum(toMerge), time))
					if np.any(~toMerge):
						self.createWanderers(bh_indices[~toMerge], bh_targets[~toMerge])

			#Finally merge those halos.
			self.mergeHalo(relevantHaloOrBlackHole[0], relevantHaloOrBlackHole[1])
			self.lastMajorMerger[relevantHaloOrBlackHole[0]] = time

		#Disk Flips
		elif eventType == 'diskFlip':
			self.integrateBHs(relevantHaloOrBlackHole[0], time, integrateAll)
			#Flip the spin, then set the next flip.
			self.spin_bh[relevantHaloOrBlackHole[0]] *= -1
			self.scheduledFlipTime[relevantHaloOrBlackHole[0]] = time + sf.computeAccretionAlignmentFlipTime(self.m_bh[relevantHaloOrBlackHole[0]], cosmology_functions.t2z(time), parameters=self.diskAlignmentParameters)

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
				print("t_ABB = {0:3.2f} Gyr, z={1:4.2f}.".format(time, cosmology_functions.t2z([time])[0]))

			#Ignore any progenitors that are stripped, merged, or not formed yet.
			isRelevant = (~self.stripped) & (~self.merged) & (self.time[self.progToNode] <= time)

			#Check for new black hole seeds
			if (self.uniqueRedshift[self.step] > self.minimumSeedingRedshift) & (self.uniqueRedshift[self.step] <= self.maximumSeedingRedshift):
				self.seed(isRelevant & ~self.metalPolluted)

			#NOTE:  This is a place to make new accretion recipes! It may be as simple as just making a new bound method if you don't need extra infrastructure.

			#Determine the accretion recipe we're currently using.
			#TODO:  Make this cleaner.  Most of these are gone actually.
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
				queueTimes, queueIndices, queueTypes = self.findUniverseInterruptions(time)
				queueCount = 0
				while True:
					if len(queueTimes) == queueCount:
						break
					#Since interruptions can cause other interruptions, we only do the first, then possibly regenerate the queue.
					eventTime, eventIndices, eventType = queueTimes[queueCount], queueIndices[queueCount], queueTypes[queueCount]
					if eventType == 'feeding':
						#Simultaneous feeding events can be done in parallel to save time, since they are quite simple.
						inParallel = (queueTimes[queueCount:] == eventTime) & (queueTypes[queueCount:] == 'feeding')
						self.resolveUniverseInterruption(eventTime, queueIndices[queueCount:][inParallel], eventType, integrateAll)
						chunkLength = np.sum(inParallel)
					else:
						self.resolveUniverseInterruption(eventTime, eventIndices, eventType, integrateAll)
						chunkLength = 1
					if (self.spinEvolution & (eventType == 'feeding')) | (self.useDelayTimes & (eventType == 'haloMerger')) | (self.spinEvolution & (eventType == 'diskFlip')):
						#Currently, these are the ones that put things in the queue.  In this case, it will need to be regenerated.
						#Note: rapid disk flips can cause arbitrary slowness.
						queueTimes, queueIndices, queueTypes = self.findUniverseInterruptions(time)
						queueCount = 0
					else:
						queueCount += chunkLength

				##########################################
				##INTEGRATE ALL TIME-DEPENDENT EQUATIONS##
				##########################################

				#Recalculate after halo mergers are complete
				isRelevant = (~self.stripped) & (~self.merged) & (self.time[self.progToNode] <= time)
				relevantBHs = (~self.merged_bh) & (self.seedTime <= time) & (~self.wandering)
				self.integrateBHs(np.where(relevantBHs)[0], time, integrateAll)

				#Reform disks in any halos that have not had any mergers in 5 Gyr and have v<300 km/s
				#TODO: This is really outdated. Look up any new information, and potentially do this during the merger step.
				if self.makeEllipticals:
					newDisks = np.where(isRelevant & (self.elliptical) & (time-self.lastMajorMerger >= 5) & \
					(sf.v_peak(self.m_halo[self.progToNode], self.redshift[self.progToNode])<300))[0]
					self.elliptical[newDisks] = False

			####################
			##FILL FEEDINGLIST##
			####################

			#This is probably fine... not touching this.  2/12/24
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
