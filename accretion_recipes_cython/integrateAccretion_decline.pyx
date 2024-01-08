from .. import constants
import numpy as np
cimport numpy as cnp
from calcSpinEvolutionFromAccretion import *
from calcRadiativeEfficiency import *
from libc.math cimport exp

cdef cnp.float64_t t_Edd = constants.t_Sal / constants.yr / 1e9
cdef cnp.float64_t _codeLumToWatts = constants.M_sun / (1e9 * constants.yr) * constants.c**2 / constants.L_sun
def integrateAccretion_decline(mass, spin, timeStep, time, t_decline, t_fEdd, \
	mu=1, spinTracking=False, f_EddCrit=None, spinMax=0.998, spinlessDefault=0.1):
	"""
	Accrete at a declining Eddington ratio
	"""

	cdef cnp.float64_t[:] m_view = np.ascontiguousarray(mass, dtype=np.float64)
	cdef int npts = len(m_view)
	cdef cnp.float64_t[:] a_view = np.ascontiguousarray(spin, dtype=np.float64)
	cdef cnp.float64_t[:] fEdd = np.zeros(npts, dtype=np.float64)
	cdef cnp.float64_t[:] timeStep_view = np.ascontiguousarray(timeStep, dtype=np.float64)
	cdef cnp.float64_t[:] time_view = np.ascontiguousarray(time, dtype=np.float64)
	cdef cnp.float64_t[:] tdec_view = np.ascontiguousarray(t_decline, dtype=np.float64)
	cdef cnp.float64_t[:] tfEdd_view = np.ascontiguousarray(t_fEdd, dtype=np.float64)
	cdef int i
	cdef cnp.float64_t[:] efficiency = np.zeros(npts, dtype=np.float64)
	cdef cnp.float64_t[:] newMass = np.zeros(npts, dtype=np.float64)
	cdef cnp.float64_t[:] currentLuminosity = np.zeros(npts, dtype=np.float64)
	cdef cnp.float64_t[:] newSpin = np.zeros(npts, dtype=np.float64)
	cdef cnp.float64_t[:] m_ratios = np.zeros(npts, dtype=np.float64)

	for i in range(0,npts):
		fEdd[i] = ((time_view[i] - tdec_view[i] + tfEdd_view[i])/tfEdd_view[i])**-2
	efficiency = calcRadiativeEfficiency(a_view, fEdd, mu=mu, spinDependence=spinTracking, f_EddCrit=f_EddCrit, spinlessDefault=spinlessDefault)
	for i in range(0,npts):
		if timeStep_view[i] > 0:
			newMass[i] = m_view[i] * exp((1.0-efficiency[i])/efficiency[i] * tfEdd_view[i]**2 / t_Edd * \
			((time_view[i] - tdec_view[i] + tfEdd_view[i])**-1 - (time_view[i] - tdec_view[i] + timeStep_view[i] + tfEdd_view[i])**-1))
			currentLuminosity[i] = efficiency[i] * (newMass[i]-m_view[i])/timeStep_view[i] * _codeLumToWatts
		else:
			newMass[i] = m_view[i]
	if spinTracking:
		#Evolve spin via Bardeen 1970, assuming everything accretes coherently.
		for i in range(0,npts):
			m_ratios[i] = newMass[i]/m_view[i]
		newSpin = calcSpinEvolutionFromAccretion(a_view, m_ratios, mu=mu, spinMax=spinMax)
		return np.array(newMass), np.array(newSpin), np.array(currentLuminosity)
	else:
		return np.array(newMass), np.array(a_view), np.array(currentLuminosity)
