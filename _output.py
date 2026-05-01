import numpy as np
import pickle
import gzip
from .util import crossmatch


class OutputMixin:

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
