import numpy as np
from .calcRadiativeEfficiency import *
from .integrateAccretion import *
from .. import constants

t_Edd = constants.t_Sal / constants.yr / 1e9

def accretionDualMode(mass, spin, timeStep, time, f_EddBurst, f_EddSteady, f_EddCrit=3e-2, includeSpinDependence=False, \
	maxBurstMass=[np.inf], maxSteadyMass=[np.inf], spinMax=0.998, fiducialRadiativeEfficiency=0.1, includeHotTransition=False, MAD=False):

	#Input sanitization
	mass = np.atleast_1d(mass)
	spin = np.atleast_1d(spin)
	timeStep = np.atleast_1d(timeStep)
	time = np.atleast_1d(time)
	f_EddBurst = np.atleast_1d(f_EddBurst)
	f_EddSteady = np.atleast_1d(f_EddSteady)
	maxBurstMass = np.atleast_1d(maxBurstMass)
	maxSteadyMass = np.atleast_1d(maxSteadyMass)

	#Prepare output
	newMass = np.copy(mass)
	newSpin = np.copy(spin)
	finalLuminosity = np.zeros_like(newMass)
	finalfEdd = np.zeros_like(newMass)
	growthFromBurst = np.zeros_like(newMass)
	growthFromSteady = np.zeros_like(newMass)

	#############
	#QUASAR MODE#
	#############

	#First, see how much time each BH spends in the burst mode.
	timeAsQuasar = np.zeros_like(mass)
	hasQuasarTime = (f_EddBurst > 0) & (mass < maxBurstMass)

	if np.any(hasQuasarTime):
		#Keep track of mass changes.  Doing it this way so as not to save another variable.
		growthFromBurst[hasQuasarTime] -= newMass[hasQuasarTime]

		#Radiative efficiency determines mass growth.
		quasarRadiativeEfficiencies = calcRadiativeEfficiency(spin[hasQuasarTime], f_EddBurst[hasQuasarTime], \
		f_EddCrit=f_EddCrit, includeSpinDependence=includeSpinDependence, fiducialDefault=fiducialRadiativeEfficiency, includeHotTransition=includeHotTransition)

		#Infer time from the mass limit...
		timeAsQuasar[hasQuasarTime] = np.minimum(np.log(maxBurstMass[hasQuasarTime]/mass[hasQuasarTime]) / f_EddBurst[hasQuasarTime] * t_Edd / (1.0-quasarRadiativeEfficiencies) * quasarRadiativeEfficiencies, timeStep[hasQuasarTime])

		if MAD:
			newMass[hasQuasarTime], newSpin[hasQuasarTime], finalLuminosity[hasQuasarTime] = integrateAccretion_MAD(mass[hasQuasarTime], spin[hasQuasarTime], f_EddBurst[hasQuasarTime], timeAsQuasar[hasQuasarTime], \
			includeHotTransition=includeHotTransition, f_EddCrit=f_EddCrit, spinMax=spinMax)
		else:
			newMass[hasQuasarTime], newSpin[hasQuasarTime], finalLuminosity[hasQuasarTime] = integrateAccretion(mass[hasQuasarTime], spin[hasQuasarTime], f_EddBurst[hasQuasarTime], timeAsQuasar[hasQuasarTime], \
			includeSpinDependence=includeSpinDependence, includeHotTransition=includeHotTransition, f_EddCrit=f_EddCrit, spinMax=spinMax, fiducialRadiativeEfficiency=fiducialRadiativeEfficiency)
		finalfEdd[hasQuasarTime] = f_EddBurst[hasQuasarTime]

		growthFromBurst[hasQuasarTime] += newMass[hasQuasarTime]

	#############
	#STEADY MODE#
	#############

	#Next, see how much time each BH spends in the steady mode.
	timeAsSteady = np.zeros_like(mass)
	hasSteadyTime = (f_EddSteady > 0) & (newMass < maxSteadyMass) & (timeAsQuasar < timeStep)
	if np.any(hasSteadyTime):
		#Keep track of mass changes.  Doing it this way so as not to save another variable.
		growthFromSteady[hasSteadyTime] -= newMass[hasSteadyTime]

		#Radiative efficiency determines mass growth.  Note that we are accounting for any changes during the quasar mode.
		steadyRadiativeEfficiencies = calcRadiativeEfficiency(newSpin[hasSteadyTime], f_EddSteady[hasSteadyTime], \
		f_EddCrit=f_EddCrit, includeSpinDependence=includeSpinDependence, fiducialDefault=fiducialRadiativeEfficiency, includeHotTransition=includeHotTransition)

		#Infer time from the mass limit
		timeAsSteady[hasSteadyTime] = np.minimum(np.log(maxSteadyMass[hasSteadyTime]/mass[hasSteadyTime]) / f_EddSteady[hasSteadyTime] * t_Edd / (1.0-steadyRadiativeEfficiencies) * steadyRadiativeEfficiencies, timeStep[hasSteadyTime]-timeAsQuasar[hasSteadyTime])

		if MAD:
			newMass[hasSteadyTime], newSpin[hasSteadyTime], finalLuminosity[hasSteadyTime] = integrateAccretion(newMass[hasSteadyTime], newSpin[hasSteadyTime], f_EddSteady[hasSteadyTime], timeAsSteady[hasSteadyTime], \
			includeHotTransition=includeHotTransition, f_EddCrit=f_EddCrit, spinMax=spinMax)
		else:
			newMass[hasSteadyTime], newSpin[hasSteadyTime], finalLuminosity[hasSteadyTime] = integrateAccretion(newMass[hasSteadyTime], newSpin[hasSteadyTime], f_EddSteady[hasSteadyTime], timeAsSteady[hasSteadyTime], \
			includeSpinDependence=includeSpinDependence, includeHotTransition=includeHotTransition, f_EddCrit=f_EddCrit, spinMax=spinMax, fiducialRadiativeEfficiency=fiducialRadiativeEfficiency)
		finalfEdd[hasSteadyTime] = f_EddSteady[hasSteadyTime]

		growthFromSteady[hasSteadyTime] += newMass[hasSteadyTime]

	#0 luminosity if there was quasar time, but it shut off.
	dormant = hasQuasarTime & ~hasSteadyTime & (timeAsQuasar < timeStep)
	finalfEdd[dormant] = 0
	finalLuminosity[dormant] = 0

	return newMass, newSpin, finalLuminosity, finalfEdd, growthFromBurst, growthFromSteady
