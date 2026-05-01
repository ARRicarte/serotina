import numpy as np
from . import constants
from . import cosmology_functions
from . import accretion_recipes as acc
from .helpers import sam_functions as sf
from .util import crossmatch


class AccretionMixin:

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
			np.full(accretors.shape[0],0,dtype=float))), axis=0)

		#Make sure that the non-accretors have 0 Eddington ratio and luminosity
		if np.any(~accreting):
			self.L_bol[nonAccretors] = 0
			self.eddRatio[nonAccretors] = 0
		self.lastIntegrationTime[bhs] = times

	def integrateAll_nobudget_cap_nodecline(self, bhs, timeSteps, times):
		"""
		~self.useColdReservoir & self.useMsigmaCap & (self.powerLawDecline is None)

		Note that times is the END of the timestep.
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
			fiducialRadiativeEfficiency=self.defaultRadiativeEfficiency, MAD_super=self.MAD_super, MAD_sub=self.MAD_sub, coherent=self.coherentAccretion, madness=self.madness)

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

	def integrateAll_nobudget_cap_decline(self, bhs, timeSteps, times):
		"""
		~self.useColdReservoir & self.useMsigmaCap & (self.powerLawDecline is not None)

		Note that times is the END of the timestep.
		"""

		#Determine which black holes are accreting
		accreting = (self.mode[bhs] == 'quasar') | (self.t_decline[bhs] < np.inf)
		accretors = bhs[accreting]
		nonAccretors = bhs[~accreting]
		redshifts = cosmology_functions.t2z(times)
		if np.any(accreting):
			initialMasses = self.m_bh[accretors]

			#Mass limits are determined by the accretionBudget.
			massLimitsQuasar = self.M_bhSigma(self.bhToProg[accretors])
			massLimitsSteady = np.full(len(accretors), np.inf)

			#You may find that a quasar has overgrown its accretion budget here.  If this happens, put it into decline.
			overgrown = (self.mode[accretors] == 'quasar') & (self.m_bh[accretors] > massLimitsQuasar)
			if np.any(overgrown):
				self.t_decline[accretors[overgrown]] = times[accreting][overgrown] - timeSteps[accreting][overgrown]
				self.mode[accretors[overgrown]] = ''

			#Integrate with Cython package
			f_EddMax = np.zeros(sum(accreting))
			if self.constant_f_max is not None:
				f_EddMax[self.mode[accretors]=='quasar'] = self.constant_f_max
			else:
				f_EddMax[self.mode[accretors]=='quasar'] = sf.draw_typeI(np.sum(self.mode[accretors]=='quasar'), redshifts[accreting][self.mode[accretors]=='quasar'])

			#The actual integration step.
			self.m_bh[accretors], self.spin_bh[accretors], self.L_bol[accretors], self.eddRatio[accretors], self.t_decline[accretors], growthFromBurst, growthFromSteady = \
			acc.accretionDualModeDecline(self.m_bh[accretors], self.spin_bh[accretors], timeSteps[accreting], times[accreting], self.t_decline[accretors], f_EddMax,
			f_EddCrit=self.f_EddCrit, includeSpinDependence=self.spinEvolution, maxBurstMass=massLimitsQuasar, maxSteadyMass=massLimitsSteady, spinMax=self.spinMax, \
			fiducialRadiativeEfficiency=self.defaultRadiativeEfficiency, MAD_super=self.MAD_super, MAD_sub=self.MAD_sub, powerLawDecline=self.powerLawDecline, sigma_steady=sf._sigma_typeI(redshifts[accreting]), \
			coherent=self.coherentAccretion, madness=self.madness)

			#Change mode if necessary.
			finishedQuasars = (self.mode[accretors] == 'quasar') & (self.m_bh[accretors] >= massLimitsQuasar)
			self.mode[accretors[finishedQuasars]] = ''
			needsDeclineTime = self.t_decline[accretors[finishedQuasars]] == np.inf
			if np.any(needsDeclineTime):
				#accretionDualModeDecline isn't supposed to let this happen, but it may occur due to approximations made when estimating time spent as quasar.
				self.t_decline[accretors[finishedQuasars][needsDeclineTime]] = np.full(np.sum(needsDeclineTime), times[accreting][finishedQuasars][needsDeclineTime])

			#Keep track of what was gained from the burst mode and the steady mode.
			self.m_burst[accretors] += growthFromBurst
			self.m_steady[accretors] += growthFromSteady

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
