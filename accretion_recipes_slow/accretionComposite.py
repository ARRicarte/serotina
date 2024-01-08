import numpy as np
from .integrateAccretion import *
from .integrateAccretion_decline import *
import constants

#e-folding time for Eddington accretion
t_Edd = constants.t_Sal / constants.yr / 1e9  #Was in seconds; converting to Gyr

def accretionComposite(mass, spin, timeStep, time, f_EddCrit=None, spinTracking=False, t_fEdd=0.0, t_decline=0.0, \
	maxQuasarMass=np.inf, f_EddMax=1.0, f_EddMin=0.0, triggerDecline=True, spinMax=0.998):
	"""
	Used for custom accretion

	Caution:  time is the time at the END of the timeStep; the starting time is time-timeStep.
	"""
	
	#Set up output
	newMass = mass
	newSpin = spin
	currentLuminosity = 0

	if mass >= maxQuasarMass:
		timeAsQuasar = 0
	else:
		efficiencyIfAligned = calcRadiativeEfficiency(spin, f_EddMax, mu=1, f_EddCrit=f_EddCrit, spinDependence=spinTracking)
		timeAsQuasar = np.min(np.log(maxQuasarMass/mass) / f_EddMax * t_Edd / (1.0-efficiencyIfAligned) * \
		efficiencyIfAligned, timeStep)

		#############
		#QUASAR MODE#
		#############

		newMass, newSpin, currentLuminosity = integrateAccretion(newMass, newSpin, f_EddMax, timeAsQuasar, mu=1, \
		spinTracking=spinTracking, f_EddCrit=f_EddCrit, spinMax=spinMax)
		if triggerDecline & (timeAsQuasar < timeStep):
			#Trigger the decline mode
			t_decline = time - timeStep + timeAsQuasar
			t_fEdd = 4.08e-3 * np.sqrt((mass/1e8))

	if t_decline > 0:
		#Continue accreting at a declining Eddington rate following Hopkins & Hernquist (2006)
		lastTime = 1.0/np.sqrt(f_EddMin) * t_fEdd - t_fEdd + t_decline
		timeAsDecline = np.min(lastTime-time-timeStep, timeStep-timeAsQuasar)
		if timeAsDecline > 0:
			#Note:  This method uses the starting time instead of the ending one, hence the complicated expression.
			newMass, newSpin, currentLuminosity = integrateAccretion_decline(newMass, newSpin, timeAsDecline, \
			np.max(time-timeStep,t_decline), t_decline, t_fEdd, mu=1, spinTracking=spinTracking, f_EddCrit=f_EddCrit, \
			spinMax=spinMax)
		timeAsSteady = timeStep - timeAsQuasar - timeAsDecline
		if timeAsSteady > 0:
			#You're done.  Remove the variables associated with declining accretion.
			t_decline = 0
			t_fEdd = 0
	else:
		timeAsSteady = timeStep - timeAsQuasar

	#For the remaining time, accrete at the minimum rate
	if (f_EddMin > 0) & (timeAsSteady > 0):
		#############
		#STEADY MODE#
		#############

		#Prograde or retrograde is chosen randomly for this mode
		newMass, newSpin, currentLuminosity = integrateAccretion(newMass, newSpin, f_EddMin, timeAsSteady, mu=np.random.choice([1,-1]), \
		spinTracking=spinTracking, f_EddCrit=f_EddCrit, spinMax=spinMax)

	return newMass, newSpin, currentLuminosity, t_decline, t_fEdd
