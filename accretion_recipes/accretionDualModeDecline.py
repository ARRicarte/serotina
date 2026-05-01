import numpy as np
from .calcRadiativeEfficiency import *
from .integrateAccretion import *
from .. import constants

t_Edd = constants.t_Sal / constants.yr / 1e9

def accretionDualModeDecline(mass, spin, timeStep, time, t_decline, f_EddBurst, f_EddCrit=3e-2, includeSpinDependence=False, coherent=True, \
	maxBurstMass=[np.inf], maxSteadyMass=[np.inf], spinMax=0.998, fiducialRadiativeEfficiency=0.1, includeHotTransition=False, MAD_super=False, MAD_sub=False, t_quasar=0.01, powerLawDecline=2.0, sigma_steady=[0.6], madness=1.0):

	#Input sanitization
	mass = np.atleast_1d(mass)
	spin = np.atleast_1d(spin)
	timeStep = np.atleast_1d(timeStep)
	time = np.atleast_1d(time)
	t_decline = np.atleast_1d(t_decline)
	f_EddBurst = np.atleast_1d(f_EddBurst)
	maxBurstMass = np.atleast_1d(maxBurstMass)
	maxSteadyMass = np.atleast_1d(maxSteadyMass)
	sigma_steady = np.atleast_1d(sigma_steady)

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
	hasQuasarTime = (f_EddBurst > 0) & (mass < maxBurstMass) & (t_decline > time-timeStep)

	if np.any(hasQuasarTime):
		#Keep track of mass changes.  Doing it this way so as not to save another variable.
		growthFromBurst[hasQuasarTime] -= newMass[hasQuasarTime]

		#Radiative efficiency determines mass growth.
		quasarRadiativeEfficiencies = calcRadiativeEfficiency(spin[hasQuasarTime], f_EddBurst[hasQuasarTime], \
		f_EddCrit=f_EddCrit, includeSpinDependence=includeSpinDependence, fiducialDefault=fiducialRadiativeEfficiency, includeHotTransition=includeHotTransition)

		#Infer time from the mass limit.  Note that this approximates that the radiative efficiency stays the same, which may not end up being the case when integrated self-consistently below.
		timeAsQuasar[hasQuasarTime] = np.minimum(np.log(maxBurstMass[hasQuasarTime]/mass[hasQuasarTime]) / f_EddBurst[hasQuasarTime] * t_Edd / (1.0-quasarRadiativeEfficiencies) * quasarRadiativeEfficiencies, timeStep[hasQuasarTime])

		if (MAD_super or MAD_sub or not coherent):
			#Trick: integrator terminates at maximum mass, so the time as quasar may not be the same.  Given timestep instead of timeAsQuasar.
			newMass[hasQuasarTime], newSpin[hasQuasarTime], finalLuminosity[hasQuasarTime], madTimeUsed = integrateAccretion_MAD(mass[hasQuasarTime], spin[hasQuasarTime], f_EddBurst[hasQuasarTime], timeStep[hasQuasarTime], \
			includeHotTransition=includeHotTransition, f_EddCrit=f_EddCrit, spinMax=spinMax, MAD_super=MAD_super, MAD_sub=MAD_sub, maximumMass=maxBurstMass[hasQuasarTime], coherent=coherent, madness=madness)
			
			#Replace with actual used time.
			timeAsQuasar[hasQuasarTime] = madTimeUsed
		else:
			newMass[hasQuasarTime], newSpin[hasQuasarTime], finalLuminosity[hasQuasarTime] = integrateAccretion(mass[hasQuasarTime], spin[hasQuasarTime], f_EddBurst[hasQuasarTime], timeAsQuasar[hasQuasarTime], \
			includeSpinDependence=includeSpinDependence, includeHotTransition=includeHotTransition, f_EddCrit=f_EddCrit, spinMax=spinMax, fiducialRadiativeEfficiency=fiducialRadiativeEfficiency)
		finalfEdd[hasQuasarTime] = f_EddBurst[hasQuasarTime]

		growthFromBurst[hasQuasarTime] += newMass[hasQuasarTime]

	#############
	#STEADY MODE#
	#############

	#BHs may start declining during the time step.
	switchedModes = hasQuasarTime & (timeAsQuasar < timeStep)
	t_decline[switchedModes] = time[switchedModes] + timeAsQuasar[switchedModes] - timeStep[switchedModes]

	#Next, see how much time each BH spends in the steady mode.
	timeAsSteady = np.zeros_like(mass)
	hasSteadyTime = (newMass < maxSteadyMass) & (timeAsQuasar < timeStep) & (t_decline < time)

	#Approximate power-law decline as a constant Eddington ratio in this time-step.
	f_EddSteady = (1.0 + (time-timeStep + (timeStep+timeAsQuasar)/2 - t_decline)/t_quasar)**(-powerLawDecline)
	f_EddSteady *= np.random.lognormal(sigma=sigma_steady, size=len(f_EddSteady))

	if np.any(hasSteadyTime):
		#Keep track of mass changes.  Doing it this way so as not to save another variable.
		growthFromSteady[hasSteadyTime] -= newMass[hasSteadyTime]

		#Radiative efficiency determines mass growth.  Note that we are accounting for any changes during the quasar mode.
		steadyRadiativeEfficiencies = calcRadiativeEfficiency(newSpin[hasSteadyTime], f_EddSteady[hasSteadyTime], \
		f_EddCrit=f_EddCrit, includeSpinDependence=includeSpinDependence, fiducialDefault=fiducialRadiativeEfficiency, includeHotTransition=includeHotTransition)

		#Infer time from the mass limit.  Note that this approximates that the radiative efficiency stays the same, which may not end up being the case when integrated self-consistently below.
		timeAsSteady[hasSteadyTime] = np.minimum(np.log(maxSteadyMass[hasSteadyTime]/mass[hasSteadyTime]) / f_EddSteady[hasSteadyTime] * t_Edd / (1.0-steadyRadiativeEfficiencies) * steadyRadiativeEfficiencies, timeStep[hasSteadyTime]-timeAsQuasar[hasSteadyTime])

		if (MAD_super or MAD_sub or not coherent):
			newMass[hasSteadyTime], newSpin[hasSteadyTime], finalLuminosity[hasSteadyTime], madTimeUsed = integrateAccretion_MAD(newMass[hasSteadyTime], newSpin[hasSteadyTime], f_EddSteady[hasSteadyTime], timeStep[hasSteadyTime]-timeAsQuasar[hasSteadyTime], \
			includeHotTransition=includeHotTransition, f_EddCrit=f_EddCrit, spinMax=spinMax, MAD_super=MAD_super, MAD_sub=MAD_sub, coherent=coherent, madness=madness)

			#Replace with actual used time.  I'm not expecting much of a difference.
			timeAsSteady[hasSteadyTime] = madTimeUsed
		else:
			newMass[hasSteadyTime], newSpin[hasSteadyTime], finalLuminosity[hasSteadyTime] = integrateAccretion(newMass[hasSteadyTime], newSpin[hasSteadyTime], f_EddSteady[hasSteadyTime], timeAsSteady[hasSteadyTime], \
			includeSpinDependence=includeSpinDependence, includeHotTransition=includeHotTransition, f_EddCrit=f_EddCrit, spinMax=spinMax, fiducialRadiativeEfficiency=fiducialRadiativeEfficiency)
		finalfEdd[hasSteadyTime] = f_EddSteady[hasSteadyTime]

		growthFromSteady[hasSteadyTime] += newMass[hasSteadyTime]

	return newMass, newSpin, finalLuminosity, finalfEdd, t_decline, growthFromBurst, growthFromSteady
