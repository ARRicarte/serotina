from .. import constants
import numpy as np
from .calcSpinEvolutionFromAccretion import *
from .calcRadiativeEfficiency import *
from .calcEddingtonLuminosity import *

t_Edd = constants.t_Sal / constants.yr / 1e9
def integrateAccretion(mass, spinMagnitude, f_Edd, timeStep, alignment, includeSpinDependence=False, includeHotTransition=False, f_EddCrit=3e-2, spinMax=0.998, fiducialRadiativeEfficiency=0.1):
	"""
	Accrete at f_Edd for a fixed amount of time.
	"""

	mass = np.atleast_1d(mass)
	timeStep = np.atleast_1d(timeStep)
	spinMagnitude = np.atleast_1d(spinMagnitude)
	alignment = np.atleast_1d(alignment)

	#Compute radiative efficiency from the INITIAL spin.  If evolving rapidly, consider having more time steps.
	efficiency = calcRadiativeEfficiency(spinMagnitude, f_Edd, alignment, includeSpinDependence=includeSpinDependence, f_EddCrit=f_EddCrit, \
	fiducialDefault=fiducialRadiativeEfficiency, includeHotTransition=includeHotTransition)

	#Next, evolve mass.  Note that there are radiative efficiency corrections here, meaning that the Eddington ratio is what flows into the accretion disk, not the same as the growth rate.
	newMass = mass * np.exp(f_Edd * timeStep / t_Edd * (1.0-efficiency)/efficiency)

	#Assuming the radiative efficiency hasn't changed significantly during this time step, return the luminosity at the final time.
	luminosity = f_Edd * calcEddingtonLuminosity(mass)

	if includeSpinDependence:
		#Evolve spin.  Currently, only a thin disk is implemented.
		m_ratios = newMass / mass
		newSpinMagnitude = calcSpinEvolutionFromAccretion(spinMagnitude, alignment, m_ratios, spinMax=spinMax)
	else:
		newSpinMagnitude = spinMagnitude
	return newMass, newSpinMagnitude, luminosity
