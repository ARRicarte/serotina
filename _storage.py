import numpy as np
from .helpers import sam_functions as sf


class StorageMixin:

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
		self.bh_mergers = np.empty((0,6))
