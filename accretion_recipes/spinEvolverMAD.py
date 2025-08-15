"""
Built upon spinEvolverRK45.py from Ricarte, Narayan, and Curd (2023).

Some functions and constants are currently redundant with other parts of serotina.

Unfortunately, there are three different similar definitions for Eddington ratio.  Be careful.

mdot_Edd = (mass inflow rate) / (Eddington accretion rate with a fiducial radiative efficiency of 0.1):  used for defining a spin-independent inflow rate.  It is assumed that this is input.
f_Edd = L_bol / L_edd:  used for predicting bolometric luminosity.  Due to a discontinuity at mdot_Edd_hot_thin, it is not possible to use this as input.
r_Edd = (mass inflow rate) / (Eddington accretion rate assuming the radiative efficiency of a thin disk):  used in the fitting formulae of Ricarte, Narayan, and Curd (2023)
"""

import numpy as np

#Constants in SI
G = 6.67430e-11
c = 2.99792458e8
m_p = 1.67262192369e-27
thomsonCrossSection = 6.6524e-29
solarMass = 1.98847e30
solarLuminosity = 3.827e26
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

def radiativeEfficiency(spin, mdot_Edd, mdot_Edd_hot_thin=3e-2, hotAccretionSuppressionFactor=100):
	"""
	Include radiative inefficiency during hot or super-Eddington accretion.
	"""

	#Default to radiative efficiency of a thin disk, determined by the ISCO.
	epsilon = radiativeEfficiencyThin(spin)
	if mdot_Edd < mdot_Edd_hot_thin:
		#Hot accretion: suppress epsilon by a factor of 100.  Somewhat arbitrary, but motivated by Sgr A* (EHTC et al. 2022e).
		epsilon /= hotAccretionSuppressionFactor
	elif mdot_Edd > 5:
		#This functional form comes from the slim disk model fits of Watarai et al. (2000), Mineshige et al. (2000).  
		#Pivot point is somewhat arbitrary, and I've tweaked it to link continuously with the thin disk.
		epsilon *= 5 / mdot_Edd * (1 + np.log(mdot_Edd/5))
	return epsilon

def calcEddingtonLuminosity(mass):
	"""
	Standard equation, depends only on mass, bolometric.

	Input solar masses, output solar luminosities.  Only used for reporting values, not evolution.
	"""

	return 4 * np.pi * G * mass * solarMass * m_p * c / thomsonCrossSection / solarLuminosity

def calcEddingtonAccretionRateFiducial(mass, fiducialRadiativeEfficiency=0.1):
	"""
	Returns the Eddington mass accretion rate assuming a fiducial radiative efficiency.
	"""

	return 4 * np.pi * G * mass * m_p / (fiducialRadiativeEfficiency * thomsonCrossSection * c) * yr

def calcMaximumMagnetization(spin):
	"""
	Fitting function from Narayan et al. (2022), based on ideal GRMHD MAD tori.
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
	Result borrowed from Lowell et al. (2023).  They report that the HD parts of e and l are remarkably flat as a function of spin.
	"""

	return 0.86 - 2*spin*0.97

def sHD(spin, r_Edd, parameters=[0.01971853]):
	"""
	Direct fit to s with respect to Eddington ratio (RNC23).
	"""

	xi = parameters[0]*r_Edd
	return (calcSpinupNT(spin) + sHD_min(spin)*xi) / (1+xi)

def calcMagnetization(spin, r_Edd, parameters=[1.29, 1.88], mdot_Edd_hot_thin=3e-2):
	"""
	Fit to magnetization with respect to Eddington ratio (RNC23).
	"""

	mdot_Edd = r_Edd * 0.1 / radiativeEfficiencyThin(spin)
	if mdot_Edd > mdot_Edd_hot_thin:
		evolutionFactor = (r_Edd/parameters[1])**parameters[0]
		return calcMaximumMagnetization(spin) * evolutionFactor / (1.0 + evolutionFactor)
	else:
		return calcMaximumMagnetization(spin)

def jetEfficiencyFunction(phi, spin, kappa=0.05):
	"""
	Formula from Tchekhovskoy et al. (2010), fit to GRMHD.
	"""

	horizonAngularVelocity = Omega_H(spin)
	return kappa / (4*np.pi) * phi**2 * horizonAngularVelocity**2 * (1. + 1.38*horizonAngularVelocity**2 - 9.2*horizonAngularVelocity**4)

def etaEM(spin, r_Edd, mdot_Edd_hot_thin=3e-2):
	"""
	Feed RNC23 fitting function into Tchekhovskoy et al. (2010) fitting function.
	"""

	return jetEfficiencyFunction(calcMagnetization(spin, r_Edd, mdot_Edd_hot_thin=mdot_Edd_hot_thin), spin)

def k_ratio(spin):
	"""
	Fitting function from Lowell et al. (2023) for Omega_F / Omega_H.  Somewhat weird behavior, but matches results.
	"""

	if spin <= 0:
		return 0.23
	else:
		return np.minimum(0.35, 0.1 + 0.5*spin)

def calcJetPower(mass, spin, mdot_Edd, madness=1.0, mdot_Edd_hot_thin=3e-2, mad_superEddington=True, mad_subEddington=True):
	"""
	Use a fitting function for phi and connect to BZ.
	"""

	massInflowRate = calcEddingtonAccretionRateFiducial(mass) * mdot_Edd
	r_Edd = mdot_Edd * radiativeEfficiencyThin(spin) / 0.1

	if (mad_superEddington & (mdot_Edd > mdot_Edd_hot_thin)) | (mad_subEddington & (mdot_Edd <= mdot_Edd_hot_thin)):
		#Simply eta * Mdot * c^2
		return etaEM(spin, r_Edd, mdot_Edd_hot_thin=mdot_Edd_hot_thin) * madness * massInflowRate * c**2 * solarMass / yr
	else:
		return 0.0

def calcMassEvolutionRate(mass, spin, mdot_Edd, coherent=True, mdot_Edd_hot_thin=3e-2, mad_superEddington=True, mad_subEddington=True, madness=1.0):
	"""
	Definition of mdot_Edd.  Don't forget that f_Edd != mdot_Edd != r_Edd
	"""

	massInflowRate = calcEddingtonAccretionRateFiducial(mass) * mdot_Edd

	#Compute losses due to radiation and jets.  Note that this can be negative, correctly causing mass loss.
	epsilon = radiativeEfficiency(spin, mdot_Edd, mdot_Edd_hot_thin=mdot_Edd_hot_thin)
	r_Edd = mdot_Edd * radiativeEfficiencyThin(spin) / 0.1
	if (mad_superEddington & (mdot_Edd > mdot_Edd_hot_thin)) | (mad_subEddington & (mdot_Edd <= mdot_Edd_hot_thin)):
		eta = calcJetPower(mass, spin, mdot_Edd, madness=madness, mdot_Edd_hot_thin=mdot_Edd_hot_thin, mad_superEddington=mad_superEddington, mad_subEddington=mad_subEddington) / (massInflowRate * c**2 * solarMass / yr)
	else:
		eta = 0
	lossFactor = (1.0 - epsilon - eta)

	if not coherent:
		#Compute the same for the flipped orientation, average.
		epsilon_flip = radiativeEfficiency(-spin, mdot_Edd, mdot_Edd_hot_thin=mdot_Edd_hot_thin)
		r_Edd_flip = mdot_Edd * radiativeEfficiencyThin(-spin) / 0.1
		if (mad_superEddington & (mdot_Edd > mdot_Edd_hot_thin)) | (mad_subEddington & (mdot_Edd <= mdot_Edd_hot_thin)):
			eta_flip = calcJetPower(mass, -spin, mdot_Edd, madness=madness, mdot_Edd_hot_thin=mdot_Edd_hot_thin, mad_superEddington=mad_superEddington, mad_subEddington=mad_subEddington) / (massInflowRate * c**2 * solarMass / yr)
		else:
			eta_flip = 0
		lossFactor += (1.0 - epsilon_flip - eta_flip)
		lossFactor /= 2
	return massInflowRate * lossFactor

def calcSpinupNT(spin):
	"""
	Following equations of Shapiro (2005) for the spin evolution of a Novikov-Thorne thin disk.
	"""

	#Note:  Not allowing spins outside physical range.
	a = np.maximum(-1,np.minimum(1,spin))

	r = r_isco(spin)
	E = (r**2 - 2*r + a*np.sqrt(r)) / (r*np.sqrt(r**2 - 3*r + 2*a*np.sqrt(r)))
	l = np.sqrt(r) * (r**2 - 2*a*np.sqrt(r) + a**2) / (r*np.sqrt(r**2 - 3*r + 2*a*np.sqrt(r)))

	return l - 2*a*E

def calcSpinupMAD(spin):
	"""
	Fitting function from Narayan et al. (2022), ideal GRMHD tori.
	"""

	#Assuming coefficients from np.polyfit.  Rearranging so that the exponent increases.
	fitCoefficients = np.flipud(np.array([-4.02630245, 5.70845519, 9.43649497, -7.79639428, -12.53175763, 0.44763054]))
	order = np.arange(len(fitCoefficients))
	return np.sum(fitCoefficients * spin**order)

def calcSpinup(spin, mdot_Edd, madness=1.0, mdot_Edd_hot_thin=3e-2, mad_superEddington=True, mad_subEddington=True):
	"""
	Spinup parameter of Gammie et al. (2004), Shapiro et al. (2005).  Generalized using our model.
	"""

	r_Edd = mdot_Edd * radiativeEfficiencyThin(spin) / 0.1

	if mad_subEddington & (mdot_Edd < mdot_Edd_hot_thin):
		#You're a MAD ADAF
		s = calcSpinupMAD(spin) * madness + calcSpinupNT(spin) * (1.0-madness)
	else:
		if mad_superEddington & (mdot_Edd >= mdot_Edd_hot_thin):
			#Start with hydrodynamic spinup.
			s = sHD(spin, r_Edd)

			#Then, add electromagnetic spinup.
			if spin != 0:
				s += -etaEM(spin, r_Edd, mdot_Edd_hot_thin=mdot_Edd_hot_thin) * (1.0 / k_ratio(spin) / Omega_H(spin) - 2.0*spin) * np.sign(spin)
			if madness < 1.0:
				s = s * madness + calcSpinupNT(spin) * (1.0-madness)
		else:
			s = calcSpinupNT(spin)
	return s

def calcSpinEvolutionRate(mass, spin, mdot_Edd, madness=1.0, mdot_Edd_hot_thin=3e-2, mad_superEddington=True, mad_subEddington=True, coherent=True):
	"""
	Combine equations of Narayan et al. (2022) with new Mdot dependence.
	"""
	
	coherentSpinup = calcSpinup(spin, mdot_Edd, madness=madness, mdot_Edd_hot_thin=mdot_Edd_hot_thin, mad_superEddington=mad_superEddington, mad_subEddington=mad_subEddington) * mdot_Edd * calcEddingtonAccretionRateFiducial(mass) / mass
	if coherent:
		return coherentSpinup
	else:
		#Assume that spinflips happen all the time, on timescales short compared to the evolutionary timescale.  Careful--there's a sign flip convention required.
		incoherentSpinup = -calcSpinup(-spin, mdot_Edd, madness=madness, mdot_Edd_hot_thin=mdot_Edd_hot_thin, mad_superEddington=mad_superEddington, mad_subEddington=mad_subEddington) * mdot_Edd * calcEddingtonAccretionRateFiducial(mass) / mass
		return (coherentSpinup + incoherentSpinup)/2

class SpinEvolverRK45(object):
	"""
	Use Runge-Kutta-Fehlberg algorithm to integrate spin, mass, and jet power at once.
	"""

	def __init__(self, M0, a0, mdot_Edd0, nsteps=1e6, maximumTime_yr=13.8e9, maximumMass=np.inf, minimumSpin=-0.998, maximumSpin=0.998, allowedFractionalMassError=1e-10, allowedSpinError=1e-10, initialTimeStep_yr=1e6, minimumFractionalTimeResolution=0.001, \
	madness=1.0, mad_superEddington=True, mad_subEddington=True, mdot_Edd_hot_thin=3e-2, silent=True, coherent=True):

		#Some cleanup
		M0 = float(M0)
		a0 = float(a0)
		mdot_Edd0 = float(mdot_Edd0)
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
		self.mdot_Edd = np.zeros(nsteps)
		self.mdot_Edd[0] = np.nan
		self.f_Edd = np.zeros(nsteps)
		self.f_Edd[0] = np.nan
		self.L_bol = np.zeros(nsteps)
		self.L_bol[0] = np.nan
		self.jetPower = np.zeros(nsteps)
		self.jetPower[0] = calcJetPower(M0, a0, mdot_Edd0)
		self.madness = float(madness)
		self.mad_superEddington = bool(mad_superEddington)
		self.mad_subEddington = bool(mad_subEddington)
		self.mdot_Edd_hot_thin = float(mdot_Edd_hot_thin)
		self.coherent = bool(coherent)

		#Integration parameters
		self.timeStep = initialTimeStep_yr  #Will naturally get bigger.
		self.minimumFractionalTimeResolution = minimumFractionalTimeResolution 
		self.currentIndex = 0
		self.maximumSpin = maximumSpin
		self.minimumSpin = minimumSpin
		self.allowedFractionalMassError = allowedFractionalMassError
		self.allowedSpinError = allowedSpinError
		self.silent = silent

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

				#Note that we are assuming that eddingtonRatioFunction is in terms of mass, assuming a fiducial radiative efficiency of 10%.
				mdot_Edd = eddingtonRatioFunction(self.time[self.currentIndex] + self.timeStep*self._A[k], massPrediction, spinPrediction)
				k_list_mass[k] = self.timeStep * calcMassEvolutionRate(massPrediction, spinPrediction, mdot_Edd, coherent=self.coherent, mdot_Edd_hot_thin=self.mdot_Edd_hot_thin, mad_superEddington=self.mad_superEddington, mad_subEddington=self.mad_subEddington, madness=self.madness)
				k_list_spin[k] = self.timeStep * calcSpinEvolutionRate(massPrediction, spinPrediction, mdot_Edd, madness=self.madness, mdot_Edd_hot_thin=self.mdot_Edd_hot_thin, mad_superEddington=self.mad_superEddington, mad_subEddington=self.mad_subEddington, coherent=self.coherent)
			proposedMass = self.mass[self.currentIndex] + np.sum(self._CH * k_list_mass)
			truncationError_mass = np.abs(np.sum(self._CT * k_list_mass))
			if truncationError_mass > 0:
				stepsizeFactor_mass = 0.9 * (self.allowedFractionalMassError / (truncationError_mass/self.mass[self.currentIndex]))**0.2
			else:
				stepsizeFactor_mass = np.inf
			proposedSpin = np.maximum(self.minimumSpin, np.minimum(self.maximumSpin, self.spin[self.currentIndex] + np.sum(self._CH * k_list_spin)))
			truncationError_spin = np.abs(np.sum(self._CT * k_list_spin))
			if truncationError_spin > 0:
				stepsizeFactor_spin = 0.9 * (self.allowedSpinError / truncationError_spin)**0.2
			else:
				stepsizeFactor_spin = np.inf

			#Check if the errors are within tolerances
			if (stepsizeFactor_mass >= 1) & (stepsizeFactor_spin >= 1):
				stepTakenWithinError = True

			#Note that the timestep is only allowed to increase by a factor of 100
			self.timeStep *= np.min([stepsizeFactor_mass,stepsizeFactor_spin,100,(self.minimumFractionalTimeResolution*self.time[self.currentIndex]+self.timeStep)/self.timeStep])
			if not self.silent:
				print(f"t={self.time[self.currentIndex]:1.3e}, mdot_Edd={self.mdot_Edd[self.currentIndex]:1.3e}, M={self.mass[self.currentIndex]:1.3e}, a={self.spin[self.currentIndex]:1.3e}, P_jet={self.jetPower[self.currentIndex]:1.3e}")

		#If you succeeded, let's make some global changes.
		self.time[self.currentIndex+1] = self.time[self.currentIndex] + self.timeStep
		self.mass[self.currentIndex+1] = proposedMass
		self.spin[self.currentIndex+1] = proposedSpin
		if not self.coherent:
			#Randomly choose prograde or retrograde for the observed snapshot.
			self.spin[self.currentIndex+1] *= np.random.choice([-1,1])
		self.mdot_Edd[self.currentIndex+1] = eddingtonRatioFunction(self.time[self.currentIndex+1], self.mass[self.currentIndex+1], self.spin[self.currentIndex+1])
		self.f_Edd[self.currentIndex+1] = self.mdot_Edd[self.currentIndex+1] * radiativeEfficiency(self.spin[self.currentIndex+1], self.mdot_Edd[self.currentIndex+1], mdot_Edd_hot_thin=self.mdot_Edd_hot_thin) / 0.1
		self.L_bol[self.currentIndex+1] = self.f_Edd[self.currentIndex+1] * calcEddingtonLuminosity(self.mass[self.currentIndex+1])
		self.jetPower[self.currentIndex+1] = calcJetPower(self.mass[self.currentIndex+1], self.spin[self.currentIndex+1], self.mdot_Edd[self.currentIndex+1], madness=self.madness, mdot_Edd_hot_thin=self.mdot_Edd_hot_thin, mad_superEddington=self.mad_superEddington, mad_subEddington=self.mad_subEddington)
		self.currentIndex += 1
		if not self.silent:
			print(f"t={self.time[self.currentIndex]:1.3e}, mdot_Edd={self.mdot_Edd[self.currentIndex]:1.3e}, M={self.mass[self.currentIndex]:1.3e}, a={self.spin[self.currentIndex]:1.3e}, P_jet={self.jetPower[self.currentIndex]:1.3e}")

		#Finally, check termination conditions.
		if (self.time[self.currentIndex] > self.maximumTime_yr) | (self.mass[self.currentIndex] > self.maximumMass) | (self.currentIndex > self.maximumSteps-2): 
			self.finished = True

	def integrateAll(self, eddingtonRatioFunction):
		"""
		Calls evolve until it finishes.  eddingtonRatioFunction must return eddingtonRatio as a function of time and mass.
		"""

		self.mdot_Edd[0] = eddingtonRatioFunction(0, self.mass[0], self.spin[0])
		self.f_Edd[0] = self.mdot_Edd[0] * radiativeEfficiency(self.spin[0], self.mdot_Edd[0], mdot_Edd_hot_thin=self.mdot_Edd_hot_thin) / 0.1
		self.L_bol[0] = self.f_Edd[0] * calcEddingtonLuminosity(self.mass[0])
		self.jetPower[0] = calcJetPower(self.mass[0], self.spin[0], self.mdot_Edd[0], madness=self.madness)
		while not self.finished:
			self.evolve(eddingtonRatioFunction)

if __name__ == '__main__':
	#This is an example.

	simpleEddingtonRatioFunction = lambda t, m, a: 1e-2
	M0 = 1e9
	a0 = 0.998
	mdot_Edd0 = simpleEddingtonRatioFunction(0,M0,a0)
	integrator = SpinEvolverRK45(M0, a0, mdot_Edd0, maximumTime_yr=1e9, initialTimeStep_yr=1e7, minimumFractionalTimeResolution=1.0, silent=False)
	integrator.integrateAll(simpleEddingtonRatioFunction)
