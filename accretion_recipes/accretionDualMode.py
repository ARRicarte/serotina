import numpy as np
from .calcRadiativeEfficiency import *
from .integrateAccretion import *
from .. import constants

t_Edd = constants.t_Sal / constants.yr / 1e9

def accretionDualMode(mass, spinMagnitude, alignment, timeStep, time, f_EddBurst, f_EddSteady, f_EddCrit=3e-2, includeSpinDependence=False, \
	maxBurstMass=[np.inf], maxSteadyMass=[np.inf], spinMax=0.998, fiducialRadiativeEfficiency=0.1, includeHotTransition=False):

	#Input sanitization
	mass = np.atleast_1d(mass)
	spinMagnitude = np.atleast_1d(spinMagnitude)
	alignment = np.atleast_1d(alignment)
	timeStep = np.atleast_1d(timeStep)
	time = np.atleast_1d(time)
	f_EddBurst = np.atleast_1d(f_EddBurst)
	f_EddSteady = np.atleast_1d(f_EddSteady)
	maxBurstMass = np.atleast_1d(maxBurstMass)
	maxSteadyMass = np.atleast_1d(maxSteadyMass)

	#Prepare output
	newMass = np.copy(mass)
	newSpinMagnitude = np.copy(spinMagnitude)
	finalLuminosity = np.zeros_like(newMass)
	finalfEdd = np.zeros_like(newMass)

	#############
	#QUASAR MODE#
	#############

	#First, see how much time each BH spends in the burst mode.
	timeAsQuasar = np.zeros_like(mass)
	hasQuasarTime = (f_EddBurst > 0) & (mass < maxBurstMass)

	if np.any(hasQuasarTime):
		#Radiative efficiency determines mass growth.
		quasarRadiativeEfficiencies = calcRadiativeEfficiency(spinMagnitude[hasQuasarTime], f_EddBurst[hasQuasarTime], alignment[hasQuasarTime], \
		f_EddCrit=f_EddCrit, includeSpinDependence=includeSpinDependence, fiducialDefault=fiducialRadiativeEfficiency, includeHotTransition=includeHotTransition)

		#Infer time from the mass limit...
		timeAsQuasar[hasQuasarTime] = np.minimum(np.log(maxBurstMass[hasQuasarTime]/mass[hasQuasarTime]) / f_EddBurst[hasQuasarTime] * t_Edd / (1.0-quasarRadiativeEfficiencies) * quasarRadiativeEfficiencies, timeStep[hasQuasarTime])

		newMass[hasQuasarTime], newSpinMagnitude[hasQuasarTime], finalLuminosity[hasQuasarTime] = integrateAccretion(mass[hasQuasarTime], spinMagnitude[hasQuasarTime], f_EddBurst[hasQuasarTime], timeAsQuasar[hasQuasarTime], alignment[hasQuasarTime], \
		includeSpinDependence=includeSpinDependence, includeHotTransition=includeHotTransition, f_EddCrit=f_EddCrit, spinMax=spinMax, fiducialRadiativeEfficiency=fiducialRadiativeEfficiency)
		finalfEdd[hasQuasarTime] = f_EddBurst[hasQuasarTime]

		#NOTE: alignment flips are still an issue to think about.

	#############
	#STEADY MODE#
	#############

	#Next, see how much time each BH spends in the steady mode.
	timeAsSteady = np.zeros_like(mass)
	hasSteadyTime = (f_EddSteady > 0) & (newMass < maxSteadyMass) & (timeAsQuasar < timeStep)
	if np.any(hasSteadyTime):
		#Radiative efficiency determines mass growth.  Note that we are accounting for any changes during the quasar mode.
		steadyRadiativeEfficiencies = calcRadiativeEfficiency(newSpinMagnitude[hasSteadyTime], f_EddSteady[hasSteadyTime], alignment[hasSteadyTime], \
		f_EddCrit=f_EddCrit, includeSpinDependence=includeSpinDependence, fiducialDefault=fiducialRadiativeEfficiency, includeHotTransition=includeHotTransition)

		#Infer time from the mass limit
		timeAsSteady[hasSteadyTime] = np.minimum(np.log(maxSteadyMass[hasSteadyTime]/mass[hasSteadyTime]) / f_EddSteady[hasSteadyTime] * t_Edd / (1.0-steadyRadiativeEfficiencies) * steadyRadiativeEfficiencies, timeStep[hasSteadyTime]-timeAsQuasar[hasSteadyTime])

		newMass[hasSteadyTime], newSpinMagnitude[hasSteadyTime], finalLuminosity[hasSteadyTime] = integrateAccretion(mass[hasSteadyTime], spinMagnitude[hasSteadyTime], f_EddSteady[hasSteadyTime], timeStep[hasSteadyTime], alignment[hasSteadyTime], \
		includeSpinDependence=includeSpinDependence, includeHotTransition=includeHotTransition, f_EddCrit=f_EddCrit, spinMax=spinMax, fiducialRadiativeEfficiency=fiducialRadiativeEfficiency)
		finalfEdd[hasSteadyTime] = f_EddSteady[hasSteadyTime]

	return newMass, newSpinMagnitude, finalLuminosity, finalfEdd
