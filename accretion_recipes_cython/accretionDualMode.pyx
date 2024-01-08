import numpy as np
cimport numpy as cnp
from calcRadiativeEfficiency import *
from integrateAccretion import *
from .. import constants
from libc.math cimport log, fmax, fmin, sqrt

#e-folding time for Eddington accretion
cdef cnp.float64_t t_Edd = constants.t_Sal / constants.yr / 1e9  #Was in seconds; converting to Gyr

def accretionDualMode(mass, spin, timeStep, time, f_EddBurst, f_EddSteady, f_EddCrit=None, spinTracking=False, \
	maxBurstMass=[np.inf], maxSteadyMass=[np.inf], spinMax=0.998, spinlessDefault=0.1):
	"""
	Used for custom accretion

	Caution:  time is the time at the END of the timeStep; the starting time is time-timeStep.
	"""
	
	#Check once that all of the inputs have the right size.
	try:
		assert len(mass) == len(spin) == len(timeStep) == len(time) == len(f_EddBurst) == len(f_EddSteady) == \
		len(maxBurstMass) == len(maxSteadyMass)
	except:
		AssertionError, "One of your inputs has the wrong size."

	#Declare types of args & kwargs
	cdef cnp.float64_t[:] m_view = np.ascontiguousarray(mass, dtype=np.float64)
	cdef int npts = len(m_view)
	cdef cnp.float64_t[:] a_view = np.ascontiguousarray(spin, dtype=np.float64)
	cdef cnp.float64_t[:] timeStep_view = np.ascontiguousarray(timeStep, dtype=np.float64)
	cdef cnp.float64_t[:] time_view = np.ascontiguousarray(time, dtype=np.float64)
	cdef cnp.float64_t[:] f_EddBurst_view = np.ascontiguousarray(f_EddBurst, dtype=np.float64)
	cdef cnp.float64_t[:] f_EddSteady_view = np.ascontiguousarray(f_EddSteady, dtype=np.float64)
	cdef cnp.float64_t[:] maxBurstMass_view = np.ascontiguousarray(maxBurstMass, dtype=np.float64)
	cdef cnp.float64_t[:] maxSteadyMass_view = np.ascontiguousarray(maxSteadyMass, dtype=np.float64)

	#Some temporary variables
	cdef cnp.float64_t[:] timeAsQuasar = np.zeros(npts, dtype=np.float64)
	cdef cnp.float64_t[:] timeAsSteady = np.zeros(npts, dtype=np.float64)
	cdef cnp.float64_t[:] efficiencyIfAligned = np.zeros(npts, dtype=np.float64)
	cdef int i

	#Set up output.  Note that they are initialized with values.
	cdef cnp.float64_t[:] newMass = m_view
	cdef cnp.float64_t[:] newSpin = a_view
	cdef cnp.float64_t[:] currentLuminosity = np.zeros(npts, dtype=np.float64)
	cdef cnp.float64_t[:] fEdd = np.ascontiguousarray(f_EddBurst, dtype=np.float64)

	#############
	#QUASAR MODE#
	#############

	#Determine the amount of time spent in the quasar mode.
	efficiencyIfAligned  = calcRadiativeEfficiency(a_view, fEdd, mu=1, f_EddCrit=f_EddCrit, spinDependence=spinTracking, spinlessDefault=spinlessDefault)
	cdef int[:] isAQuasar = np.zeros(npts, dtype=np.intc)
	cdef int totalQuasars = 0
	for i in range(0,npts):
		if (m_view[i] < maxBurstMass_view[i]) & (f_EddBurst_view[i] > 0):
			timeAsQuasar[i] = fmin(log(maxBurstMass_view[i]/m_view[i]) / f_EddBurst_view[i] * t_Edd / (1.0-efficiencyIfAligned[i]) * \
			efficiencyIfAligned[i], timeStep_view[i])
			isAQuasar[i] = 1
			totalQuasars += 1
		else:
			timeAsQuasar[i] = 0

	#Temporary quasar arrays
	cdef cnp.float64_t[:] newMass_q = np.zeros(totalQuasars, dtype=np.float64)
	cdef cnp.float64_t[:] newSpin_q = np.zeros(totalQuasars, dtype=np.float64)
	cdef cnp.float64_t[:] currentLuminosity_q = np.zeros(totalQuasars, dtype=np.float64)
	cdef cnp.float64_t[:] fEdd_q = np.zeros(totalQuasars, dtype=np.float64)
	cdef cnp.float64_t[:] timeAsQuasar_q = np.zeros(totalQuasars, dtype=np.float64)
	cdef int j

	if totalQuasars > 0:

		#Fill subset arrays with appropriate values
		j = 0
		for i in range(0,npts):
			if isAQuasar[i] == 1:
				newMass_q[j] = newMass[i]
				newSpin_q[j] = newSpin[i]
				currentLuminosity_q[j] = currentLuminosity[i]
				fEdd_q[j] = fEdd[i]
				timeAsQuasar_q[j] = timeAsQuasar[i]
				j += 1

		#Perform calculation on subset
		newMass_q, newSpin_q, currentLuminosity_q = \
		integrateAccretion(newMass_q, newSpin_q, fEdd_q, timeAsQuasar_q, \
		mu=1, spinTracking=spinTracking, f_EddCrit=f_EddCrit, spinMax=spinMax, spinlessDefault=spinlessDefault)

		#Replace the values in the output arrays with those of the subset
		j = 0
		for i in range(0,npts):
			if isAQuasar[i] == 1:
				newMass[i] = newMass_q[j]
				newSpin[i] = newSpin_q[j]
				currentLuminosity[i] = currentLuminosity_q[j]
				fEdd[i] = fEdd_q[j]
				timeAsQuasar[i] = timeAsQuasar_q[j]
				j += 1

	#############
	#STEADY MODE#
	#############

	#Determine the amount of time spent in the steady mode.
	cdef int[:] isSteady = np.zeros(npts, dtype=np.intc)
	cdef int totalSteady = 0

	for i in range(0,npts):
		#This definition may be subject to floating point errors, but they shouldn't matter.
		if (f_EddSteady_view[i] > 0) & (newMass[i] < maxSteadyMass_view[i]):
			timeAsSteady[i] = fmin(log(maxSteadyMass_view[i]/newMass[i]) / f_EddSteady_view[i] * t_Edd / (1.0-efficiencyIfAligned[i]) * \
			efficiencyIfAligned[i], timeStep_view[i] - timeAsQuasar[i])
		else:
			timeAsSteady[i] = 0

		if timeAsSteady[i] > 0:
			fEdd[i] = f_EddSteady_view[i]
			isSteady[i] = 1
			totalSteady += 1

	#Temporary steady arrays
	cdef cnp.float64_t[:] newMass_s = np.zeros(totalSteady, dtype=np.float64)
	cdef cnp.float64_t[:] newSpin_s = np.zeros(totalSteady, dtype=np.float64)
	cdef cnp.float64_t[:] currentLuminosity_s = np.zeros(totalSteady, dtype=np.float64)
	cdef cnp.float64_t[:] fEdd_s = np.zeros(totalSteady, dtype=np.float64)
	cdef cnp.float64_t[:] timeAsSteady_s = np.zeros(totalSteady, dtype=np.float64)

	if totalSteady > 0:

		#Fill subset arrays with appropriate values
		j = 0
		for i in range(0,npts):
			if isSteady[i] == 1:
				newMass_s[j] = newMass[i]
				newSpin_s[j] = newSpin[i]
				currentLuminosity_s[j] = currentLuminosity[i]
				fEdd_s[j] = fEdd[i]
				timeAsSteady_s[j] = timeAsSteady[i]
				j += 1

		#Perform calculation on subset
		newMass_s, newSpin_s, currentLuminosity_s = \
		integrateAccretion(newMass_s, newSpin_s, fEdd_s, timeAsSteady_s, \
		mu=np.random.choice([1,-1], size=totalSteady), spinTracking=spinTracking, f_EddCrit=f_EddCrit, spinMax=spinMax, spinlessDefault=spinlessDefault)

		#Replace the values in the output arrays with those of the subset
		j = 0
		for i in range(0,npts):
			if isSteady[i] == 1:
				newMass[i] = newMass_s[j]
				newSpin[i] = newSpin_s[j]
				currentLuminosity[i] = currentLuminosity_s[j]
				fEdd[i] = fEdd_s[j]
				timeAsSteady[i] = timeAsSteady_s[j]
				j += 1

	#Arrange final f_Edd
	for i in range(0,npts):
		if timeAsSteady[i] + timeAsQuasar[i] < timeStep[i]:
			fEdd[i] = 0

	return np.array(newMass), np.array(newSpin), np.array(currentLuminosity), np.array(fEdd)
