import numpy as np
from . import cosmology_functions
from .helpers import sam_functions as sf
from .helpers import fabioMergerProbabilities
from .util import crossmatch


class EvolutionMixin:

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

			#Check the criterion that quasar targets are below the maximum mass.  Otherwise, don't trigger.
			#NOTE: immediateDeclineHole was a test.  These runs were nuts.

			#immediateDeclineHole = np.array([]) #Default to possibly be overwritten
			if len(feedingHole) > 0:
				massLimitsQuasar = self.M_bhSigma(self.bhToProg[feedingHole])
				toQuasar = self.m_bh[feedingHole] < massLimitsQuasar
				#feedingHole, immediateDeclineHole = feedingHole[toQuasar], feedingHole[~toQuasar]
				feedingHole = feedingHole[toQuasar]

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
					notQuasars = self.mode[feedingHole] != 'quasar'
					self.mode[feedingHole[notQuasars]] = 'quasar'
					self.t_decline[feedingHole[notQuasars]] = np.inf

			'''
			if len(immediateDeclineHole) > 0:
				#These were already overmassive. Put into decline immediately.
				self.integrateBHs(immediateDeclineHole, time, integrateAll)
				self.t_decline[immediateDeclineHole] = time
			'''

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
			#TODO:  Make this cleaner.  Most of these are gone actually, especially useColdReservoir stuff.
			if self.uniqueRedshift[self.step] > self.z_superEdd:
				integrateAll = self.integrateAll_superEdd
			elif self.useColdReservoir and not self.useMsigmaCap:
				integrateAll = self.integrateAll_budget_nocap_nodecline
			elif not self.useColdReservoir and self.useMsigmaCap:
				#Only these are currently being used.
				if self.powerLawDecline is None:
					integrateAll = self.integrateAll_nobudget_cap_nodecline
				else:
					integrateAll = self.integrateAll_nobudget_cap_decline
			elif self.useColdReservoir & self.useMsigmaCap:
				integrateAll = self.integrateAll_budget_cap_nodecline
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
