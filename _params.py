import numpy as np


class ParametersMixin:

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
		'useColdReservoir': False, 'f_superEdd': 5.0, 'makeEllipticals': False, 'f_EddCrit': 3e-2, \
		'minimumSeedingRedshift': 15, 'maximumSeedingRedshift': 20, 'useMetalConstraints': False, 'supernovaBarrier': False, 'blackHoleMergerProbability': 1.0, \
		'steady': None, 'isothermalSigma': False, 'defaultRadiativeEfficiency': 0.1, 'Q_dcbh': 3.0, 'nscMassThreshold': 1e8, 'mergerMode': 'flatProbability', \
		'diskAlignmentParameters': [np.inf], 'MAD_sub': False, 'MAD_super': False, 'violentMergers': False, 'alwaysMergeEmpty': True, 'powerLawDecline': None, \
		'coherentAccretion': True, 'spin_seed': 0.0, 'madness': 1.0, 'violentMassRatio': 0.05}

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
