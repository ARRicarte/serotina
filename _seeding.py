import numpy as np
from . import constants
from . import cosmology_functions
from .helpers import sam_functions as sf
from .helpers import bh_imfs
from .util import crossmatch


class SeedingMixin:

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
			self.spin_bh = np.concatenate((self.spin_bh, np.full(n_new,self.spin_seed)))
			self.seedType = np.concatenate((self.seedType, np.array(seedTypes)))
			self.bhToProg = np.concatenate((self.bhToProg, np.array(seededProgenitors)))
			self.seedTime = np.concatenate((self.seedTime, np.full(n_new,self.uniqueTime[self.step])))
			self.t_decline = np.concatenate((self.t_decline, np.full(n_new,np.inf)))
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
