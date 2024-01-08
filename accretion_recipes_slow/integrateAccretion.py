import constants
import numpy as np
from .calcSpinEvolutionFromAccretion import *
from .calcRadiativeEfficiency import *

t_Edd = constants.t_Sal / constants.yr / 1e9
def integrateAccretion(mass, spin, f_Edd, timeStep, mu=1, spinTracking=False, f_EddCrit=None, spinMax=0.998):
	"""
	Accrete at f_Edd for a fixed amount of time.
	"""
	efficiency = calcRadiativeEfficiency(spin, f_Edd, mu=mu, spinDependence=spinTracking, f_EddCrit=f_EddCrit)
	newMass = mass * np.exp(f_Edd * timeStep / t_Edd * (1.0-efficiency)/efficiency)
	if spinTracking:
		#Evolve spin via Bardeen 1970, assuming everything accretes coherently.
		newSpin = calcSpinEvolutionFromAccretion(spin, newMass/mass, mu=mu, spinMax=spinMax)
	else:
		newSpin = spin
	currentLuminosity = efficiency * f_Edd * (newMass-mass)/timeStep * \
	constants.M_sun / (1e9 * constants.yr) * constants.c**2 / constants.L_sun
	return newMass, newSpin, currentLuminosity


