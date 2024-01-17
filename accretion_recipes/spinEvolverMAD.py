"""
Adapted from spinEvolverRK45.py from Ricarte, Narayan, and Curd (2023).

Some functions and constants are currently redundant with other parts of serotina.
"""

import numpy as np

#Constants in SI
G = 6.67430e-11
c = 2.99792458e8
m_p = 1.67262192369e-27
thomsonCrossSection = 6.6524e-29
solarMass = 1.98847e30
yr = 3.154e7

def r_isco(spin):
	"""
	Innermost stable circular orbit, bearing in mind prograde vs. retrograde.
	"""

	Z1 = 1.0 + (1.0 - spin**2)**(1.0/3.0) * ((1.0 + spin)**(1.0/3.0) + (1.0 - spin)**(1.0/3.0))
	Z2 = np.sqrt(3*spin**2 + Z1**2)
	r = 3 + Z2 - np.sign(spin)*np.sqrt((3.0 - Z1) * (3.0 + Z1 + 2*Z2))  #Note the addition of sign(spin) that's not in Shapiro's paper.
	return r

def radiativeEfficiencyThin(spin):
	"""
	Used to define the Eddington ratio.
	"""
	
	return 1.0 - np.sqrt(1.0 - 2.0/3.0/r_isco(spin))

def calcEddingtonAccretionRate(mass, spin):
	"""
	Returns the Eddington limit of accretion in units of solar masses per year.
	"""

	efficiency = radiativeEfficiencyThin(spin)
	return 4 * np.pi * G * mass * m_p / (efficiency * thomsonCrossSection * c) * yr

def calcMaximumMagnetization(spin):
	"""
	Fitting function we made up in Narayan et al. 2022
	"""

	return -20.2*spin**3 - 14.9*spin**2 + 34*spin + 52.6

def spinToHorizon(a):
	"""
	Computes the horizon radius for a given spin value.
	"""

	return 1.0 + np.sqrt(1.0 - a**2)

def Omega_H(spin):
	"""
	Rotation velocity of hole.
	"""
	return np.abs(spin) / (2*spinToHorizon(spin))

def sHD_min(spin):
	"""
	Result borrowed from Lowell et al. 2023.  They report that the HD parts of e and l are remarkably flat as a function of spin.
	"""

	return 0.86 - 2*spin*0.97

def sHD(spin, eddingtonRatio, parameters=[0.01971853]):
	"""
	Direct fit to s with respect to Eddington ratio.
	"""

	xi = parameters[0]*eddingtonRatio
	return (calcSpinupNT(spin) + sHD_min(spin)*xi) / (1+xi)

def calcMagnetization(spin, eddingtonRatio, parameters=[1.29, 1.88], fEdd_ADAF_threshold=3e-2):
	"""
	Newly introduced model function.
	"""

	if eddingtonRatio > fEdd_ADAF_threshold:
		evolutionFactor = (eddingtonRatio/parameters[1])**parameters[0]
		return calcMaximumMagnetization(spin) * evolutionFactor / (1.0 + evolutionFactor)
	else:
		return calcMaximumMagnetization(spin)

def jetEfficiencyFunction(phi, spin, kappa=0.05):
	"""
	Formula from Tchekhovskoy et al. (2010)
	"""

	horizonAngularVelocity = Omega_H(spin)
	return kappa / (4*np.pi) * phi**2 * horizonAngularVelocity**2 * (1. + 1.38*horizonAngularVelocity**2 - 9.2*horizonAngularVelocity**4)

def etaEM(spin, eddingtonRatio, fEdd_ADAF_threshold=3e-2):
	"""
	Assume models for magnetization and jet efficiency.
	"""

	return jetEfficiencyFunction(calcMagnetization(spin, eddingtonRatio, fEdd_ADAF_threshold=fEdd_ADAF_threshold), spin)

def k_ratio(spin):
	"""
	Fitting function from Lowell et al. (2023) for Omega_F / Omega_H
	"""

	if spin <= 0:
		return 0.23
	else:
		return np.minimum(0.35, 0.1 + 0.5*spin)

def calcJetPower(mass, spin, eddingtonRatio, fEdd_ADAF_threshold=3e-2):
	"""
	Use a fitting function for phi and connect to BZ.
	"""

	#Simply eta * Mdot * c^2
	return etaEM(spin, eddingtonRatio, fEdd_ADAF_threshold=fEdd_ADAF_threshold) * eddingtonRatio * calcEddingtonAccretionRate(mass, spin) * c**2 * solarMass / yr

def calcMassEvolutionRate(mass, spin, eddingtonRatio):
	"""
	Definition of the Eddington ratio.
	"""

	return calcEddingtonAccretionRate(mass, spin) * eddingtonRatio

def calcSpinupNT(spin):
	"""
	Following equations of Shapiro (2005)
	"""

	#Note:  Not allowing spins outside physical range.
	a = np.maximum(-1,np.minimum(1,spin))

	r = r_isco(spin)
	E = (r**2 - 2*r + a*np.sqrt(r)) / (r*np.sqrt(r**2 - 3*r + 2*a*np.sqrt(r)))
	l = np.sqrt(r) * (r**2 - 2*a*np.sqrt(r) + a**2) / (r*np.sqrt(r**2 - 3*r + 2*a*np.sqrt(r)))

	return l - 2*a*E

def calcSpinupMAD(spin):
	"""
	Fitting function from Narayan et al. (2022)
	"""

	#Assuming coefficients from np.polyfit.  Rearranging so that the exponent increases.
	fitCoefficients = np.flipud(np.array([-4.02630245, 5.70845519, 9.43649497, -7.79639428, -12.53175763, 0.44763054]))
	order = np.arange(len(fitCoefficients))
	return np.sum(fitCoefficients * spin**order)

def calcSpinup(spin, eddingtonRatio, fEdd_ADAF_threshold=3e-2):
	"""
	Spinup parameter of Gammie et al. (2004), Shapiro et al. (2005)
	"""

	if eddingtonRatio < fEdd_ADAF_threshold:
		#You're a MAD ADAF
		return calcSpinupMAD(spin)
	else:
		#Start with hydrodynamic spinup.
		s = sHD(spin, eddingtonRatio)
		if spin != 0:
			#Then, add electromagnetic spinup.
			s += -etaEM(spin, eddingtonRatio) * (1.0 / k_ratio(spin) / Omega_H(spin) - 2.0*spin) * np.sign(spin)
		return s

def calcSpinEvolutionRate(mass, spin, eddingtonRatio, fEdd_ADAF_threshold=3e-2):
	"""
	Combine equations of Narayan et al. (2022) with new Mdot dependence.
	"""
	
	return calcSpinup(spin, eddingtonRatio, fEdd_ADAF_threshold=fEdd_ADAF_threshold) * eddingtonRatio * calcEddingtonAccretionRate(mass, spin) / mass

class SpinEvolverRK45(object):
	"""
	Use Runge-Kutta-Fehlberg algorithm to integrate spin, mass, and jet power at once.
	"""

	def __init__(self, M0, a0, fEdd0, nsteps=1e6, maximumTime_yr=13.8e9, maximumMass=2e10, minimumSpin=-0.99999, maximumSpin=0.998, allowedFractionalMassError=1e-10, allowedSpinError=1e-10, initialTimeStep_yr=1e6, minimumFractionalTimeResolution=0.001):

		#Some cleanup
		M0 = float(M0)
		a0 = float(a0)
		fEdd0 = float(fEdd0)
		nsteps = int(nsteps)
		maximumTime_yr = float(maximumTime_yr)
		maximumMass = float(maximumMass)
		allowedFractionalMassError = float(allowedFractionalMassError)
		allowedSpinError = float(allowedSpinError)
		maximumSpin = float(maximumSpin)
		minimumSpin = float(minimumSpin)
		initialTimeStep_yr = float(initialTimeStep_yr)

		#Initial conditions
		self.mass = np.zeros(nsteps)
		self.mass[0] = M0   #Solar masses
		self.spin = np.zeros(nsteps)
		self.spin[0] = a0
		self.time = np.zeros(nsteps)
		self.time[0] = 0.0  #yr
		self.eddingtonRatio = np.zeros(nsteps)
		self.eddingtonRatio[0] = np.nan
		self.jetPower = np.zeros(nsteps)
		self.jetPower[0] = calcJetPower(M0, a0, fEdd0)

		#Integration parameters
		self.timeStep = initialTimeStep_yr  #Will naturally get bigger.
		self.minimumFractionalTimeResolution = minimumFractionalTimeResolution 
		self.currentIndex = 0
		self.maximumSpin = maximumSpin
		self.minimumSpin = minimumSpin
		self.allowedFractionalMassError = allowedFractionalMassError
		self.allowedSpinError = allowedSpinError

		#The RKF algorithm needs coefficients which are hard-coded here...
		self._A = np.array([0.0, 2.0/9.0, 1.0/3.0, 3.0/4.0, 1.0, 5.0/6.0])
		self._B = np.zeros((6,5), dtype=float)
		self._B[1,0] = 2.0/9.0
		self._B[2,0] = 1.0/12.0
		self._B[3,0] = 69.0/128.0
		self._B[4,0] = -17.0/12.0
		self._B[5,0] = 65.0/432.0
		self._B[2,1] = 1.0/4.0
		self._B[3,1] = -243.0/128.0
		self._B[4,1] = 27.0/4.0
		self._B[5,1] = -5.0/16.0
		self._B[3,2] = 135.0/64.0
		self._B[4,2] = -27.0/5.0
		self._B[5,2] = 13.0/16.0
		self._B[4,3] = 16.0/15.0
		self._B[5,3] = 4.0/27.0
		self._B[5,4] = 5.0/144.0
		self._C = np.array([1.0/9.0, 0.0, 9.0/20.0, 16.0/45.0, 1.0/12.0])
		self._CH = np.array([47.0/450.0, 0.0, 12.0/25.0, 32.0/225.0, 1.0/30.0, 6.0/25.0])
		self._CT = np.array([-1.0/150.0, 0.0, 3.0/100.0, -16.0/75.0, -1.0/20.0, 6.0/25.0])

		#Flips when calculations are done.
		self.maximumMass = maximumMass
		self.maximumTime_yr = maximumTime_yr
		self.maximumSteps = nsteps
		self.finished = False

	def evolve(self, eddingtonRatioFunction):
		"""
		Simultaneous integration of mass, spin, and jet power.

		For generality, eddingtonRatioFunction needs to return eddingtonRatio with arguments (time, mass, spin)
		"""	

		stepTakenWithinError = False
		while not stepTakenWithinError:
			#Mass and spin evolve together
			k_list_mass = np.zeros(6, dtype=float)
			k_list_spin = np.zeros(6, dtype=float)
			for k in range(6):
				massPrediction = self.mass[self.currentIndex]
				spinPrediction = self.spin[self.currentIndex]
				for l in range(k):
					massPrediction += k_list_mass[l] * self._B[k,l]
					spinPrediction += k_list_spin[l] * self._B[k,l]

				#Many formulae break if you don't enforce this here.
				spinPrediction = np.maximum(self.minimumSpin,np.minimum(self.maximumSpin, spinPrediction))

				fEdd = eddingtonRatioFunction(self.time[self.currentIndex] + self.timeStep*self._A[k], massPrediction, spinPrediction)
				k_list_mass[k] = self.timeStep * calcMassEvolutionRate(massPrediction, spinPrediction, fEdd)
				k_list_spin[k] = self.timeStep * calcSpinEvolutionRate(massPrediction, spinPrediction, fEdd)
			proposedMass = self.mass[self.currentIndex] + np.sum(self._CH * k_list_mass)
			truncationError_mass = np.abs(np.sum(self._CT * k_list_mass))
			stepsizeFactor_mass = 0.9 * (self.allowedFractionalMassError / (truncationError_mass/self.mass[self.currentIndex]))**0.2
			proposedSpin = np.maximum(self.minimumSpin, np.minimum(self.maximumSpin, self.spin[self.currentIndex] + np.sum(self._CH * k_list_spin)))
			truncationError_spin = np.abs(np.sum(self._CT * k_list_spin))
			stepsizeFactor_spin = 0.9 * (self.allowedSpinError / truncationError_spin)**0.2

			#Check if the errors are within tolerances
			if (stepsizeFactor_mass >= 1) & (stepsizeFactor_spin >= 1):
				stepTakenWithinError = True
			self.timeStep *= np.min([stepsizeFactor_mass,stepsizeFactor_spin,100,(self.minimumFractionalTimeResolution*self.time[self.currentIndex]+self.timeStep)/self.timeStep])
			print(f"t={self.time[self.currentIndex]:1.3e}, f_Edd={self.eddingtonRatio[self.currentIndex]:1.3e}, M={self.mass[self.currentIndex]:1.3e}, a={self.spin[self.currentIndex]:1.3e}, P_jet={self.jetPower[self.currentIndex]:1.3e}")

		#If you succeeded, let's make some global changes.
		self.time[self.currentIndex+1] = self.time[self.currentIndex] + self.timeStep
		self.mass[self.currentIndex+1] = proposedMass
		self.spin[self.currentIndex+1] = proposedSpin
		self.eddingtonRatio[self.currentIndex+1] = eddingtonRatioFunction(self.time[self.currentIndex+1], self.mass[self.currentIndex+1], self.spin[self.currentIndex])
		self.jetPower[self.currentIndex+1] = calcJetPower(self.mass[self.currentIndex+1], self.spin[self.currentIndex+1], self.eddingtonRatio[self.currentIndex+1])
		self.currentIndex += 1
		print(f"t={self.time[self.currentIndex]:1.3e}, f_Edd={self.eddingtonRatio[self.currentIndex]:1.3e}, M={self.mass[self.currentIndex]:1.3e}, a={self.spin[self.currentIndex]:1.3e}, P_jet={self.jetPower[self.currentIndex]:1.3e}")

		#Finally, check termination conditions.
		if (self.time[self.currentIndex] > self.maximumTime_yr) | (self.mass[self.currentIndex] > self.maximumMass) | (self.currentIndex > self.maximumSteps-2): 
			self.finished = True

	def integrateAll(self, eddingtonRatioFunction):
		"""
		Calls evolve until it finishes.  eddingtonRatioFunction must return eddingtonRatio as a function of time and mass.
		"""

		self.eddingtonRatio[0] = eddingtonRatioFunction(0, self.mass[0], self.spin[0])
		self.jetPower[0] = calcJetPower(self.mass[0], self.spin[0], self.eddingtonRatio[0])
		while not self.finished:
			self.evolve(eddingtonRatioFunction)

if __name__ == '__main__':
	#This is an example.

	simpleEddingtonRatioFunction = lambda t, m, a: 1000.00
	M0 = 10.0
	a0 = 0.1
	fEdd0 = simpleEddingtonRatioFunction(0,M0,a0)
	integrator = SpinEvolverRK45(M0, a0, fEdd0, maximumTime_yr=1e9, initialTimeStep_yr=1e7, minimumFractionalTimeResolution=1.0)
	integrator.integrateAll(simpleEddingtonRatioFunction)
