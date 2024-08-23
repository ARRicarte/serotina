from .. import constants
import numpy as np
from .calcSpinEvolutionFromAccretion import *
from .calcRadiativeEfficiency import *
from .calcEddingtonLuminosity import *
from .spinEvolverMAD import *

t_Edd = constants.t_Sal / constants.yr / 1e9
def integrateAccretion(mass, spin, f_Edd, timeStep, includeSpinDependence=False, includeHotTransition=False, f_EddCrit=3e-2, spinMax=0.998, fiducialRadiativeEfficiency=0.1):
	"""
	Accrete at f_Edd for a fixed amount of time.
	"""

	mass = np.atleast_1d(mass)
	timeStep = np.atleast_1d(timeStep)
	spin = np.atleast_1d(spin)

	#Compute radiative efficiency from the INITIAL spin.  If evolving rapidly, consider having more time steps.
	efficiency = calcRadiativeEfficiency(spin, f_Edd, includeSpinDependence=includeSpinDependence, f_EddCrit=f_EddCrit, \
	fiducialDefault=fiducialRadiativeEfficiency, includeHotTransition=includeHotTransition)

	#Next, evolve mass.  Note that there are radiative efficiency corrections here, meaning that the Eddington ratio is what flows into the accretion disk, not the same as the growth rate.
	newMass = mass * np.exp(f_Edd * timeStep / t_Edd * (1.0-efficiency)/efficiency)

	#Assuming the radiative efficiency hasn't changed significantly during this time step, return the luminosity at the final time.
	luminosity = f_Edd * calcEddingtonLuminosity(newMass)

	if includeSpinDependence:
		#Evolve spin.  Currently, only a thin disk is implemented.
		m_ratios = newMass / mass
		newSpin = calcSpinEvolutionFromAccretion(spin, m_ratios, spinMax=spinMax)
	else:
		newSpin = spin
	return newMass, newSpin, luminosity

def integrateAccretion_MAD(mass, spin, f_Edd, timeStep, includeHotTransition=False, f_EddCrit=3e-2, spinMax=0.998, MAD_super=False, MAD_sub=False, maximumMass=None, maxSteps=100, coherent=True, madness=1.0):
	"""
	Account for MAD disks and use an integrator to evolve mass and spin.  Not optimized.

	includeHotTransition not currently used.
	"""

	newMass = np.zeros_like(mass)
	newSpin = np.zeros_like(mass)
	luminosity = np.zeros_like(mass)
	if maximumMass is None:
		maximumMassUsed = np.full(len(mass), np.inf)
	else:
		maximumMassUsed = maximumMass
	timeSpent = np.zeros_like(mass)

	#A for loop with Runge-Kutta, surely very slow.  Relaxed some precision requirements.
	#TODO: parallelize
	for index in range(len(mass)):
		integrator = SpinEvolverRK45(mass[index], spin[index], f_Edd[index], nsteps=maxSteps, maximumTime_yr=timeStep[index]*1e9, minimumSpin=-spinMax, maximumSpin=spinMax, \
		allowedFractionalMassError=1e-2, allowedSpinError=1e-3, initialTimeStep_yr=timeStep[index]*1e9/10, minimumFractionalTimeResolution=1.0, \
		mad_superEddington=MAD_super, mad_subEddington=MAD_sub, fEdd_ADAF_threshold=f_EddCrit, maximumMass=maximumMassUsed[index], coherent=coherent, madness=madness)
		constantEddingtonRatioFunction = lambda t, m, a: f_Edd[index]
		integrator.integrateAll(constantEddingtonRatioFunction)
		if integrator.currentIndex == maxSteps-1:
			raise ValueError("RK45 integrator reached maximum step.")
		newMass[index] = integrator.mass[integrator.currentIndex]
		newSpin[index] = integrator.spin[integrator.currentIndex]
		luminosity[index] = f_Edd[index] * calcEddingtonLuminosity(newMass[index])
		timeSpent[index] = integrator.time[integrator.currentIndex]/1e9

	return newMass, newSpin, luminosity, timeSpent
