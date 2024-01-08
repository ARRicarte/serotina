import numpy as np
cimport numpy as cnp
from calcRadiativeEfficiency import *
from integrateAccretion import *
from integrateAccretion_decline import *
from .. import constants
from libc.math cimport log, fmax, fmin, sqrt

#e-folding time for Eddington accretion
cdef cnp.float64_t t_Edd = constants.t_Sal / constants.yr / 1e9  #Was in seconds; converting to Gyr

def accretionComposite(mass, spin, timeStep, time, f_EddMax, f_EddMin, f_EddCrit=None, spinTracking=False, t_fEdd=0.0, t_decline=0.0, \
	maxQuasarMass=np.inf, triggerDecline=True, spinMax=0.998, spinlessDefault=0.1):
	"""
	Used for custom accretion

	Caution:  time is the time at the END of the timeStep; the starting time is time-timeStep.
	"""
	
	#Declare types of args & kwargs
	cdef cnp.float64_t[:] m_view = np.ascontiguousarray(mass, dtype=np.float64)
	cdef int npts = len(m_view)
	cdef cnp.float64_t[:] a_view = np.ascontiguousarray(spin, dtype=np.float64)
	cdef cnp.float64_t[:] timeStep_view = np.ascontiguousarray(timeStep, dtype=np.float64)
	cdef cnp.float64_t[:] time_view = np.ascontiguousarray(time, dtype=np.float64)
	cdef cnp.float64_t[:] f_EddMax_view = np.ascontiguousarray(f_EddMax, dtype=np.float64)
	cdef cnp.float64_t[:] f_EddMin_view = np.ascontiguousarray(f_EddMin, dtype=np.float64)
	cdef cnp.float64_t[:] massLimit_view = np.ascontiguousarray(maxQuasarMass, dtype=np.float64)
	cdef cnp.float64_t[:] tfEdd_view = np.ascontiguousarray(t_fEdd, dtype=np.float64)
	cdef cnp.float64_t[:] tdec_view = np.ascontiguousarray(t_decline, dtype=np.float64)

	#Some temporary variables
	cdef cnp.float64_t[:] timeAsQuasar = np.zeros(npts, dtype=np.float64)
	cdef cnp.float64_t[:] timeAsDecline = np.zeros(npts, dtype=np.float64)
	cdef cnp.float64_t[:] timeAsSteady = np.zeros(npts, dtype=np.float64)
	cdef cnp.float64_t[:] efficiencyIfAligned = np.zeros(npts, dtype=np.float64)
	cdef int i

	#Set up output.  Note that they are initialized with values.
	cdef cnp.float64_t[:] newMass = m_view
	cdef cnp.float64_t[:] newSpin = a_view
	cdef cnp.float64_t[:] currentLuminosity = np.zeros(npts, dtype=np.float64)

	#############
	#QUASAR MODE#
	#############

	#Determine the amount of time spent in the quasar mode.
	#cdef cnp.float64_t[:] fEdd = np.full(npts, f_EddMax, dtype=np.float64)
	cdef cnp.float64_t[:] fEdd = np.ascontiguousarray(f_EddMax, dtype=np.float64)
	efficiencyIfAligned  = calcRadiativeEfficiency(a_view, fEdd, mu=1, f_EddCrit=f_EddCrit, spinDependence=spinTracking, spinlessDefault=spinlessDefault)
	cdef int[:] isAQuasar = np.zeros(npts, dtype=np.intc)
	cdef int totalQuasars = 0
	for i in range(0,npts):
		if m_view[i] >= massLimit_view[i]:
			timeAsQuasar[i] = 0
		else:
			timeAsQuasar[i] = fmin(log(massLimit_view[i]/m_view[i]) / f_EddMax_view[i] * t_Edd / (1.0-efficiencyIfAligned[i]) * \
			efficiencyIfAligned[i], timeStep_view[i])
			isAQuasar[i] = 1
			totalQuasars += 1

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

		if triggerDecline:
			for i in range(0,npts):
				if (isAQuasar[i]) & (timeAsQuasar[i] < timeStep_view[i]):
					#FREE PARAMETERIZATION!
					tfEdd_view[i] = 4.08e-3 * sqrt(m_view[i]/1e8)
					tdec_view[i] = time_view[i] - timeStep_view[i] + timeAsQuasar[i]

	##############
	#DECLINE MODE#
	##############

	#Count decliners
	cdef int totalDecline = 0
	cdef int[:] isDeclining = np.zeros(npts, dtype=np.intc)
	if triggerDecline:
		for i in range(npts):
			if tdec_view[i] > 0:
				isDeclining[i] = 1
				totalDecline += 1

	#Temporary decline arrays
	cdef cnp.float64_t[:] newMass_d = np.zeros(totalDecline, dtype=np.float64)
	cdef cnp.float64_t[:] newSpin_d = np.zeros(totalDecline, dtype=np.float64)
	cdef cnp.float64_t[:] currentLuminosity_d = np.zeros(totalDecline, dtype=np.float64)
	cdef cnp.float64_t[:] timeAsDecline_d = np.zeros(totalDecline, dtype=np.float64)
	cdef cnp.float64_t[:] timeAsQuasar_d = np.zeros(totalDecline, dtype=np.float64)
	cdef cnp.float64_t[:] time_view_d = np.zeros(totalDecline, dtype=np.float64)
	cdef cnp.float64_t[:] timeStep_view_d = np.zeros(totalDecline, dtype=np.float64)
	cdef cnp.float64_t[:] tdec_view_d = np.zeros(totalDecline, dtype=np.float64)
	cdef cnp.float64_t[:] tfEdd_view_d = np.zeros(totalDecline, dtype=np.float64)
	cdef cnp.float64_t[:] declineStartTimes_d = np.zeros(totalDecline, dtype=np.float64)

	#Determine the amount of time spent in the decline mode during this time step.
	if triggerDecline:
		for i in range(0,npts):
			if tdec_view[i] > 0:
				timeAsDecline[i] = fmin((tdec_view[i] + (f_EddMin_view[i]/f_EddMax_view[i])**-0.5 * tfEdd_view[i] - tfEdd_view[i]) - (time_view[i]-timeStep_view[i]+timeAsQuasar[i]), \
				timeStep_view[i]-timeAsQuasar[i])

	if totalDecline > 0:

		#Fill subset arrays with appropriate values
		j = 0
		for i in range(0,npts):
			if isDeclining[i] == 1:
				newMass_d[j] = newMass[i]
				newSpin_d[j] = newSpin[i]
				currentLuminosity_d[j] = currentLuminosity[i]
				timeAsDecline_d[j] = timeAsDecline[i]
				timeAsQuasar_d[j] = timeAsQuasar[i]
				time_view_d[j] = time_view[i]
				timeStep_view_d[j] = timeStep_view[i]
				tdec_view_d[j] = tdec_view[i]
				tfEdd_view_d[j] = tfEdd_view[i]
				j += 1

		#NOTE:  Time here is the time at the START of the integration, hence why I have to make a new array.
		for j in range(totalDecline):
			declineStartTimes_d[j] = fmax(time_view_d[j]-timeStep_view_d[j]+timeAsQuasar_d[j],tdec_view_d[j])

		#Perform calculation on subset
		newMass_d, newSpin_d, currentLuminosity_d = \
		integrateAccretion_decline(newMass_d, newSpin_d, timeAsDecline_d, \
		declineStartTimes_d, tdec_view_d, tfEdd_view_d, mu=1, \
		spinTracking=spinTracking, f_EddCrit=f_EddCrit, spinMax=spinMax, spinlessDefault=spinlessDefault)
			
		#Replace the values in the output arrays with those of the subset
		j = 0	
		for i in range(0,npts):
			if isDeclining[i] == 1:
				newMass[i] = newMass_d[j]
				newSpin[i] = newSpin_d[j]
				currentLuminosity[i] = currentLuminosity_d[j]
				timeAsDecline[i] = timeAsDecline_d[j]
				timeAsQuasar[i] = timeAsQuasar_d[j]
				time_view[i] = time_view_d[j]
				timeStep_view[i] = timeStep_view_d[j]
				tdec_view[i] = tdec_view_d[j]
				tfEdd_view[i] = tfEdd_view_d[j]

				#While we're at it, let's shut off any finished decliners
				if timeAsDecline[i] < timeStep_view[i]-timeAsQuasar[i]:
					tdec_view[i] = 0
					tfEdd_view[i] = 0
				j += 1

	#############
	#STEADY MODE#
	#############

	#Determine the amount of time spent in the steady mode.
	cdef int[:] isSteady = np.zeros(npts, dtype=np.intc)
	cdef int totalSteady = 0

	for i in range(0,npts):
		#This definition may be subject to floating point errors, but they shouldn't matter.
		timeAsSteady[i] = timeStep_view[i] - timeAsQuasar[i] - timeAsDecline[i]
		if timeAsSteady[i] > 0:
			fEdd[i] = f_EddMin_view[i]
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
		mu=np.random.choice([1,-1], size=totalSteady), spinTracking=spinTracking, f_EddCrit=f_EddCrit, spinMax=spinMax, \
		spinlessDefault=spinlessDefault)

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

	return np.array(newMass), np.array(newSpin), np.array(currentLuminosity), np.array(tdec_view), np.array(tfEdd_view)
