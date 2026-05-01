import numpy as np
from . import cosmology_functions
from .helpers import sam_functions as sf
from . import black_hole_mergers as bhb
from .util import crossmatch, primariesAndSecondaries


class MergersMixin:

	def mergeBHs(self, primaries, secondaries, progenitors, times):
		"""
		Merge two black holes.  Keeping the properties of the first index.
		"""

		npts = len(primaries)

		# Default to None so calcRemnantProperties uses its internal defaults when
		# neither spin evolution nor merger kicks are active.
		theta1 = theta2 = phi1 = phi2 = None

		if (self.mergerKicks) | (self.spinEvolution):

			if self.alignMergerSpins:
				#One might expect this in gas rich environments, if the binaries shrunk in a gas disc.
				theta1 = np.full(npts, 0.0)
				theta2 = np.full(npts, 0.0)
				phi1 = np.full(npts, 0.0)
				phi2 = np.full(npts, 0.0)
			else:
				#One might expect this in gas poor environments.  Note even sampling with respect to solid angle, and allowing for negative signs (retrograde orbits).
				theta1 = np.arccos(1.0-2*np.random.random(npts)) * np.random.choice([1.0,-1.0], npts)
				theta2 = np.arccos(1.0-2*np.random.random(npts)) * np.random.choice([1.0,-1.0], npts)
				phi1 = 2*np.pi*np.random.random(npts)
				phi2 = 2*np.pi*np.random.random(npts)

		initialSpins = np.abs(self.spin_bh[primaries])

		#Compute mass, spin, and recoil based on GR fits.
		mass_gr, spin_gr, kick_gr = bhb.calcRemnantProperties(self.m_bh[primaries], self.m_bh[secondaries], np.abs(self.spin_bh[primaries]), np.abs(self.spin_bh[secondaries]), \
		theta1=theta1, theta2=theta2, phi1=phi1, phi2=phi2, spinMax=self.spinMax)
		if self.spinEvolution:
			#Mass-weighted spin projected onto the binary axis, most easily accessed by GW detectors.
			chi_eff = (self.m_bh[primaries] * np.abs(self.spin_bh[primaries]) * np.cos(theta1) + self.m_bh[secondaries] * np.abs(self.spin_bh[secondaries]) * np.cos(theta2)) / (self.m_bh[primaries] + self.m_bh[secondaries])

			#Note that only positive spins go into this formula.
			initiallyRetrograde = self.spin_bh[primaries] < 0
			self.spin_bh[primaries] = spin_gr
		else:
			chi_eff = np.zeros(npts, dtype=float)

		#Save to a list of all merger events.
		massRatio = self.m_bh[secondaries]/self.m_bh[primaries]
		self.bh_mergers = np.vstack((self.bh_mergers, np.array([self.m_bh[primaries], massRatio, cosmology_functions.t2z(times), self.spin_bh[primaries], chi_eff, initialSpins]).transpose()))

		#Simply adding the two masses together.
		#Account for radiated mass loss.  Usually a few percent, but might as well.
		self.m_merged[primaries] += mass_gr-self.m_bh[primaries]
		self.m_bh[primaries] = mass_gr
		self.seedType[primaries] = 'Merged'
		self.bhToProg[primaries] = progenitors

		if self.spinEvolution:
			#By default, maintain prograde or retrograde nature of the primary.  That means re-flip anything that was negative.
			if np.any(initiallyRetrograde):
				self.spin_bh[primaries] = (-1)**(int(initiallyRetrograde)) * self.spin_bh[primaries]

			if self.violentMergers:
				#If the merger was retrograde, assume that the new gas reservoir is also retrograde.
				flippers = (np.abs(theta1) > np.pi/2) & (massRatio >= self.violentMassRatio)
				if np.any(flippers):
					self.spin_bh[primaries] = (-1)**(int(flippers)) * self.spin_bh[primaries]

		#Now let's see if the kick is large enough to cause it to leave the halo.
		if self.mergerKicks:
			#In reality, one may want to do an orbital integration.  This is definitely an area for improvement.  Here, there's only one local speed, and that's sigma.  Maybe just get rid of anything with v>sigma.
			kicked = kick_gr > self.sigma[self.progToNode[self.bhToProg[primaries]]]

			'''
			#Compare kick velocity to Choksi formula
			#Note:  Using the escape velocity here is pretty unfair.  In reality, it depends on baryonic distribution, dark matter profile, etc., all of which are time and orbit dependent.
			kicked = (kick_gr > sf.calcRecoilEscapeVelocity_permanent(self.m_halo[self.progToNode[progenitors]], cosmology_functions.t2z(times)))
			'''
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
		self.t_decline[bh_ids] = np.inf
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
		self.t_decline[bh_ids] = np.inf
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
			self.t_decline[bhsOfLost] = np.inf
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
