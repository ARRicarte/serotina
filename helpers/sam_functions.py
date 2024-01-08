"""
ARR 11.09.15

Miscellaneous functions imported into sam.py or tree.py
"""

import numpy as np
import os
from .. import cosmology
from .. import constants
import cPickle as pickle
from scipy.integrate import quad
from scipy.interpolate import interp1d, interp2d, InterpolatedUnivariateSpline, RectBivariateSpline, griddata
from scipy.signal import fftconvolve
from .. import config
currentPath = os.path.abspath(os.path.dirname(__file__)) + '/'

def pSpin(spin, lbar=0.05, sigma=0.5, sigmaMax=6):
	"""
	This is a log-normal, de-normalized so that the max value is 1.
	"""
	return lbar/spin * np.exp(-np.log(spin/lbar)**2/2/sigma**2)

def drawRandomHaloSpin(nvals=1, sigma=0.5, sigmaMax=6):
	"""
	Draw from a log-normal distribution the value of a spin parameter of a halo
	using rejection-comparison.  Only going up to 6 sigma.

	Called by sam.py
	"""
	if nvals == 1:
		while True:
			proposition = np.random.random()*sigma*sigmaMax
			probability = pSpin(proposition)
			dice = np.random.random()
			if dice <= probability:
				break
		return proposition
	else:
		propositions = np.random.random(nvals)*sigma*sigmaMax
		probabilities = pSpin(propositions)
		dice = np.random.random(nvals)
		rejections = np.where(dice > probabilities)[0]
		while len(rejections) > 0:
			newProps = np.random.random(len(rejections))*sigma*sigmaMax
			newProbs = pSpin(newProps)
			dice = np.random.random(len(rejections))
			propositions[rejections] = newProps
			probabilities[rejections] = newProbs
			rejections = rejections[np.where(dice > newProbs)[0]]
		return propositions

def M2v(masses, redshifts):
	"""
	Convert masses into circular velocities with the spherical collapse model.

	See Barkana & Loeb (2001)

	Note:  Same as Volonteri+2013
	"""
	d = cosmology.Omega_m * (1+redshifts)**3 / (cosmology.Omega_m * (1+redshifts)**3 + \
	cosmology.Omega_l + (1.0 - cosmology.Omega_l - cosmology.Omega_m)*(1+redshifts)**2) - 1
	Delta_c = 18*np.pi**2 + 82*d - 39*d**2
	return 23.4 * (masses / 1e8 / cosmology.h)**(1.0/3.0) * (cosmology.Omega_m/(d+1) * Delta_c/(18*np.pi**2))**(1.0/6.0) * \
	np.sqrt((1.0+redshifts)/10)

#####################################################################################
##DEPRECATED:  This entire class of equations is flawed.  Use velocityDispersion()!##
#####################################################################################

def v2sigma_Ferrarese(v_c, gammaDelta=(0.55,0.84)):
	"""
	Calculate sigma from v_c, using Ferrarese 2002 by default.  This is parameterized as logv = gamma + delta*logsigma.
        """
	return 10**((np.log10(v_c)-gammaDelta[0])/gammaDelta[1])

def v2sigma_SIS(v_c):
	"""
	Return sigma from v_c with a singular isothermal sphere model
	"""
	return v_c / np.sqrt(2)

def v2sigma_dpl(v_c, lowerSlope=2.210, upperSlope=0.4810, vBreak=131.5, sigmaBreak=132.5):
	"""
	Return sigma from v_c based on Larkin & McLaughlin (2016).
	"""
        if not hasattr(v_c, '__len__'):
                v_c = np.array([v_c])
        output = np.zeros(len(v_c))
        output[v_c < vBreak] = sigmaBreak * (v_c[v_c < vBreak] / vBreak)**lowerSlope
        output[v_c >= vBreak] = sigmaBreak * (v_c[v_c >= vBreak] / vBreak)**upperSlope
        return output

with open(currentPath + '../lookup_tables/Larkin_McLaughlin_2016/v2sigma.pkl', 'r') as myfile:
	v2sigma_dict = pickle.load(myfile)

v_peak_ll16 = v2sigma_dict['V_peak']
sigma_ll16 = v2sigma_dict['sigma']

def v2sigma(v_peak, redshiftEvolution=0.25):
	"""
	Now using a dictionary made of modeling from LL16!
	"""
        if not hasattr(v_peak, '__len__'):
                v_peak = np.array([v_peak])

	return np.interp(v_peak, v_peak_ll16, sigma_ll16)

def v2M_BH(v_c, alphaBeta=(8.22,4.58), gammaDelta=(0.55,0.84)):
	"""
	Turn halo circular velocity into BH mass
	"""
	return 10**(alphaBeta[0] + alphaBeta[1] * (1.0/gammaDelta[1]*(np.log10(v_c) - gammaDelta[0]) - np.log10(200)))

########################################
##CORRECT SIGMA DEFINITION AS OF 06/09##
########################################

with open(currentPath + '../lookup_tables/velocityDispersion/velocityDispersion.pkl', 'r') as myfile:
	vd_dict = pickle.load(myfile)

velocityDispersionTable = vd_dict['sigma']
vd_redshift = vd_dict['redshift']
vd_logHaloMass = vd_dict['logHaloMass']

_logInterpVelocityDispersion = RectBivariateSpline(vd_logHaloMass, vd_redshift, velocityDispersionTable)

def velocityDispersion(m_halo, z):
	return _logInterpVelocityDispersion(np.log10(m_halo), z, grid=False)

"""
Time and redshift conversions.
"""

with open(currentPath + "../lookup_tables/tofz.pkl", 'rb') as myfile:
	data = pickle.load(myfile)
tarray = data['tarray']
zarray = data['zarray']

z2t_interp = interp1d(zarray, tarray)
t2z_interp = interp1d(tarray/1e9, zarray)

def z2t(z):
	"""
	Compute time as a function of redshift.  Results have been tabulated for the Planck2015 cosmology.

	Returns time in Gyr.
	"""
	return z2t_interp(z)/1e9

def t2z(t):
	"""
	Compute redshift as a function of time.
	"""
	return t2z_interp(t)


def v2t(m_halo, v):
	"""
	Compute dynamical time as a function of halo circular velocity.  In the SIS model, we're assuming that
	we want the time at 0.1 R_vir.

	Assuming v in km/s, outputs t in Gyr.
	"""
	
	return constants.G * m_halo * constants.M_sun / (v * 1e3)**3 / constants.yr * 0.1 / 1e9

def t_dyn(redshifts):
	"""
	Compute the dynamical time for a halo at redshift (z).  This is independent of mass in a spherical collapse model.
	Estimated at 0.1 R_vir.  See Barkana & Loeb (2001).

	Output in Gyr
	"""
	d = cosmology.Omega_m * (1+redshifts)**3 / (cosmology.Omega_m * (1+redshifts)**3 + \
        cosmology.Omega_l + (1.0 - cosmology.Omega_l - cosmology.Omega_m)*(1+redshifts)**2) - 1
        Delta_c = 18*np.pi**2 + 82*d - 39*d**2
	outTime = 0.003276 / cosmology.h * (cosmology.Omega_m/(d+1) * Delta_c/(18*np.pi**2))**-0.5 * ((1.0+redshifts)/10)**-1.5
	return outTime

def v2M(v_c):
	"""
	Convert velocity to black hole mass using the MM13 relation.

	Wants km s^-1 and returns solar masses.
	"""
	return 10**(8.32+5.64*np.log10(v2sigma(v_c)/200))

def calcDeltaVir(z):
	"""
	Compute the overdensity parameter Delta_vir at different redshift.
	"""
	
	Omega_m_at_z = cosmology.Omega_m*(1+z)**3 / (cosmology.Omega_m*(1+z)**3 + cosmology.Omega_l + \
	(1.0-cosmology.Omega_m-cosmology.Omega_l)*(1+z)**2)
	x = Omega_m_at_z - 1
	return (18*np.pi**2 + 82*x - 39*x**2) / Omega_m_at_z

def calcLumDist(z):
	"""
	Compute the luminosity distance to some redshift. So as not to waste time, this only works for flat universes.
	Units are in meters.
	"""
	d_H = constants.c * cosmology.t_H * constants.yr
	d_c = d_H * quad(lambda zprime: (cosmology.Omega_m*(1+zprime)**3 + cosmology.Omega_l)**-0.5, 0, z)[0]
	return d_c * (1 + z)

def H(z):
	"""
	Compute the Hubble constant as a function of redshift.
	(Ignoring Omega_k and Omega_r.)
	"""
	return cosmology.H_0 * np.sqrt(cosmology.Omega_m*(1+z)**3 + cosmology.Omega_l)

def eddingtonLimit(m_bh, efficiency=0.1):
	"""
	Returns the Eddington limit of accretion in units of solar masses per year.
	"""
	return 4 * np.pi * constants.G * m_bh * constants.m_p / (efficiency * constants.sigma_T * constants.c) * constants.yr

def eddingtonLum(m_bh):
        """
        Returns the Eddington luminosity in units of solar luminosities.
        """
        return 4 * np.pi * constants.G * m_bh * constants.M_sun * constants.m_p * constants.c / constants.sigma_T / constants.L_sun

class treeNavigator(object):

	"""
	Stores possible files in tree_output, and converts parameters to closest match.
	"""
	
	def __init__(self, path=None):
		"""
		Store masses and indices
		"""
		if path is None:
			path = config.treeInputPath
		
		self.path = path
		self.files = [file for file in os.listdir(path) if file[-7:] == '.bin.gz']
		#Get rid of extensions.  In the process I screw up the beginning of the name, since there's a decimal point.
		noExtensions = [name.split('.')[0:2] for name in self.files]
		#Patch the words back together.
		completeNames = [part[0]+'.'+part[1] for part in noExtensions]
		splitNames = [name.split('_') for name in completeNames]
		self.massStrings = [name[0][1:] for name in splitNames]
		self.n_mass = len(np.unique(self.massStrings))
		self.numberStrings = [name[1][1:] for name in splitNames]
		self.n_sample = len(np.unique(self.numberStrings))
		self.logUniqueMasses = np.log10(np.sort(np.unique(self.massStrings).astype(float)))
		self.logMassRange = [np.min(self.logUniqueMasses), np.max(self.logUniqueMasses)]

	def findClosestTree(self, mass, number):
		"""
		Convert parameters to file name.
		"""

		mString = min(self.massStrings, key=lambda x: abs(float(x)-mass))
		nString = min(self.numberStrings, key=lambda x: abs(int(x)-number))
		return self.path + '/m'+mString+'_'+'n'+nString+'.bin.gz'

#Halo mass function
with open(currentPath + '../lookup_tables/massFunction.pkl', 'r') as myfile:
	massFunctionDict = pickle.load(myfile)
numberDensityMasses = massFunctionDict['mass']
numberDensityValues = massFunctionDict['dndlog10m']
numberDensityRedshifts = massFunctionDict['redshift']
_calcNumberDensity_log = RectBivariateSpline(numberDensityRedshifts, np.log10(numberDensityMasses), numberDensityValues)

def calcNumberDensity(haloMass, redshift):
	"""
	Doing interpolation in log space
	"""
	return _calcNumberDensity_log(redshift, np.log10(haloMass*cosmology.h), grid=False)

def calcNumberDensity_genmf(logHaloMass, redshift, infile=currentPath + '../lookup_tables/massFunctions.pkl'):
	"""
	Calculate the number density of objects of mass haloMass at redshift redshift.
	This returns an interpolated value from genmfcode in log(dn [Mpc^-3])/dlogM
	"""
	from scipy.interpolate import RectBivariateSpline
	
	with open(infile, 'rb') as myfile:
		interpTable = pickle.load(myfile)

	massFunctions = np.flipud(interpTable['massFunctions'])
	massFunctions[np.where(~np.isfinite(massFunctions))[0]] = -99
	logM = np.flipud(interpTable['logM'])
	redshifts = interpTable['redshift']

	interpFunct = RectBivariateSpline(logM, redshifts, massFunctions)
	return interpFunct(logHaloMass, redshift)

"""
Cooling Function Stuff
"""

'''
with open('./cooling_curves/coolingFunction_SD93.pkl', 'rb') as myfile:
	interpTable = pickle.load(myfile)

logT = interpTable['logT']
logLambda = interpTable['logLambda']
FeH = interpTable['FeH']
FeH[0] = -7
#It's actually -infinity, but that's mayhem for interpolation.  Set to -7 by hand.
interpLambda = interp2d(logT, FeH, logLambda, bounds_error=False, fill_value=-np.inf)
#NOTE:  May have to be more careful about extrapolations.  fill_value should not be 0
#if, for example, FeH > 0.5

def coolingFunction(temperature, metallicity):
	"""
	Use the Sutherland & Dopita 1993 cooling curves in order to compute the cooling function.

	Table is in erg / cm^3 / s.  This function outputs Joules / m^3 / s.
	"""
	
	#It's possible that I should be more careful about what I use for X, but probably not.  
	#Assuming it's the same as solar.
	input_feh = max(np.log10(metallicity/0.02), -7)
	return 10**interpLambda(np.log10(temperature), input_feh) * constants.erg * (1e2)**3
'''

"""
Peak calculation stuff
"""

with open(currentPath + '../lookup_tables/sigmaPeaks.pkl', 'r') as myfile:
	sigmaData = pickle.load(myfile)
sigmaMasses = sigmaData['masses']
sigmaNus = sigmaData['nus']
sigmaRedshifts = sigmaData['redshifts']
_calcSigmaPeak_log = RectBivariateSpline(sigmaRedshifts, np.log10(sigmaMasses), sigmaNus)

def m2nu(haloMass, redshift):
	"""
	Calculate nu, the number of sigma this peak is, at a given halo mass and redshift.
	
	Does interpolation in log space of the mass.
	"""
	return _calcSigmaPeak_log(redshift, np.log10(haloMass), grid=False)

def nu2m(nu, redshift):
	"""
	The inverse is tricky, since each redshift has a different range of nu.
	"""

	redshiftProximity = np.argsort(np.abs(sigmaRedshifts-redshift))
	closeIndices = [np.where(redshiftProximity == 0)[0], np.where(redshiftProximity == 1)[0]]
	interpMasses = []
	for i in closeIndices:
		approximateMass = interp1d(np.squeeze(sigmaNus[i,:]), sigmaMasses)(nu)
		interpMasses.append(approximateMass)
	finalMass = (interpMasses[1]-interpMasses[0]) / (sigmaRedshifts[closeIndices[1]] - sigmaRedshifts[closeIndices[0]]) * \
	(redshift - sigmaRedshifts[closeIndices[0]]) + interpMasses[0]

	return finalMass

def bias(haloMass, redshift, a=0.707, b=0.5, c=0.6, deltac=1.686):
	nu = m2nu(haloMass, redshift)
	nu1 = np.sqrt(a) * nu
	return 1.0 + 1.0 / deltac * (nu1**2 + b*nu1**(2.0*(1.0-c)) - (nu1**(2*c)/np.sqrt(a)) / \
	(nu1**(2*c) + b*(1.0-c)*(1.0-c/2)))

"""
Binary Decay Stuff
"""

with open(currentPath + '../lookup_tables/binaryDecayTable.pkl', 'rb') as myfile:
	interpTable = pickle.load(myfile)
tau = interpTable['taus']
alpha = interpTable['alphas']
tau2alpha = InterpolatedUnivariateSpline(tau, alpha)
alpha2tau = InterpolatedUnivariateSpline(alpha, tau)

def calcHardeningRadius(m2, sigma_DM):
	"""
	Calculate the radius at which a binary becomes hard.  This is the radius used to initialize binaries
	that have coalesced due to dynamical friction.

	Assuming m2 in solar masses, sigma_DM in km/s.  Returning a_h in pc.
	"""
	return constants.G * (m2*constants.M_sun) / 4 / (sigma_DM*1e3)**2 / constants.pc

def calcDecay(a_now, Delta_t, sigma_DM, m1, m2, H=15.0, J=1.0):
	"""
	Calculate the separation that a MBHB should have after some time elapses after its formation.

	Assuming a in pc, Delta_t in Gyr, sigma_DM in km/s, m1 and m2 in solar masses.
	Returns a_new in pc
	"""
	a_h = calcHardeningRadius(m2, sigma_DM) * constants.pc
	t_0 = 9 * np.pi * constants.G**2 * J**2 * ((m1+m2)*constants.M_sun)**2 / (8 * H * (sigma_DM*1e3)**5 * a_h)
	
	alpha_now = a_h/(a_now*constants.pc)
	tau_now = alpha2tau(alpha_now)
	tau_new = tau_now + (Delta_t*1e9*constants.yr)/t_0
	alpha_new = tau2alpha(tau_new)
	return a_h / alpha_new / constants.pc

def calcTransitionRadius(a_now, sigma_DM, m1, m2, H=15.0, J=1.0):
	"""
	Calculate the radius at which gravitational radiation should take over.  Assuming circular orbit.  Condition is given
	by the radius at which da/dt is the same for both stellar scattering and gravitational radiation.
	
	Units are as in calcDecay()
	"""
	a_h = calcHardeningRadius(m2, sigma_DM) * constants.pc
	r_core = 3.0/4.0/(sigma_DM*1e3)**2 * constants.G * J * ((m1+m2)*constants.M_sun) * np.log(a_h/a_now)
	rho_core = (sigma_DM*1e3)**2 / (2*np.pi*constants.G*r_core**2)
	return ((64 * constants.G**2 * (sigma_DM*1e3) * (m1 * m2 * (m1+m2) * constants.M_sun**3)) / (5 * constants.c**5 * H * rho_core))**0.2 / constants.pc

def calcMdot(M, f_Edd, timeStep, alpha=-1, t_Q=1e9, efficiency=0.1):
	"""
	Calculate the accretion rate of a black hole using Volonteri+ 2006.
	Uses solar masses and years
	"""
	Mdot_Edd = eddingtonLimit(M, efficiency=efficiency)
	#Mdot_Edd = M / (constants.t_Sal/constants.yr)
	fdot_Edd = f_Edd**(1.0-alpha) / np.abs(alpha) / t_Q * (efficiency * Mdot_Edd * constants.M_sun / constants.yr * constants.c**2 / 1e9 / constants.L_sun)**(-alpha)
	if timeStep > 0:
		fnew_Edd = f_Edd + timeStep * fdot_Edd
		output = fnew_Edd * Mdot_Edd * (1.0-efficiency)/efficiency, fdot_Edd
	else:
		output = f_Edd * Mdot_Edd * (1.0-efficiency)/efficiency, fdot_Edd
	return output

kutta4 = \
[[0, 0, 0, 0, 0], \
[1.0/3.0, 1.0/3.0, 0, 0, 0], \
[2.0/3.0, -1.0/3.0, 1, 0, 0], \
[1, 1, -1, 1, 0], \
[0, 1.0/8.0, 3.0/8.0, 3.0/8.0, 1.0/8.0]]

def integrateM(M, f_Edd, timeStep, alpha=-1, t_Q=1e9, efficiency=0.1, tableau=kutta4):
	"""
	Integrate black hole mass using Volonteri+ 2006
	Uses solar masses and years

	Be careful...super-exponential growth leads to numerical instability if f_Edd ~ 1 and |alpha| is too large.
	"""
	karray_M = np.zeros(len(tableau)-1)
	karray_f = np.zeros(len(tableau)-1)
	for i in range(len(tableau)-1):
		M_temp = M + timeStep*sum([tableau[i][j+1]*karray_M[j] for j in range(i)])
		f_temp = f_Edd + timeStep*sum([tableau[i][j+1]*karray_f[j] for j in range(i)])
		karray_M[i], karray_f[i] = calcMdot(M_temp, f_temp, tableau[i][0]*timeStep, alpha=alpha, t_Q=t_Q, efficiency=efficiency)
	M_dot = sum([karray_M[i]*tableau[-1][i+1] for i in range(len(tableau)-1)])
	f_dot = sum([karray_f[i]*tableau[-1][i+1] for i in range(len(tableau)-1)])
	return M+M_dot*timeStep, f_Edd+f_dot*timeStep

"""
Metal pollution for seeding
"""

with open(currentPath + '../lookup_tables/scannapieco2003.pkl', 'r') as myfile:
	scannapieco2003 = pickle.load(myfile)

def calcUnpollutedProbability(masses, redshifts):
	"""
	Calculate the probability that a halo formed with some mass at some redshift is pristine of metals.
	"""

	#These things need to be arrays to be zipped.
	massArr = np.atleast_1d(masses)
	zArr = np.atleast_1d(redshifts)

	#To prevent the interpolation function from returning 0, I'm setting the values of the really small halos to the values
	#they'd have if they were the minimum value on the plot.
	massArr[massArr < 4e6] = 4e6
	zArr[zArr > 34] = 34
	return griddata(scannapieco2003['massAndRedshift'], scannapieco2003['efficiency'], zip(massArr,zArr), fill_value=0)

def determinePollution(masses, redshifts, seedingEfficiency=1.0):
	masses = np.atleast_1d(masses)
	redshifts = np.atleast_1d(redshifts)
	pollutionProbabilities = 1.0 - calcUnpollutedProbability(masses, redshifts)
	dice = np.random.random(masses.shape[0])
	isPolluted = dice < pollutionProbabilities

	#A second cut for seeding efficiency
	moreDice = np.random.random(masses.shape[0])
	isPolluted = isPolluted | (moreDice < (1.0 - seedingEfficiency))
	return isPolluted
	
"""
Hopkins luminosity functions
"""

with open(currentPath + '../lookup_tables/hopkins07_lum_funct.txt', 'r') as mydata:
	hopkinsTable = np.loadtxt(mydata)
hop_z = hopkinsTable[:,0]
hop_logphi = hopkinsTable[:,1]
hop_logL = hopkinsTable[:,2]
hop_g1 = hopkinsTable[:,3]
hop_g2 = hopkinsTable[:,4]
hopkinsInterpFuncts = [interp1d(hop_z,parameter) for parameter in [hop_logphi, hop_logL, hop_g1, hop_g2]]
interpAllHopkins = lambda z: [f(z) for f in hopkinsInterpFuncts]

def hopLumFunct(L,z):
	"""
	Returns in Mpc^{-3} logL^{-1}
	"""
	logphi, logLstar, g1, g2 = interpAllHopkins(z)
	return 10**logphi / ((L/10**logLstar)**g1 + (L/10**logLstar)**g2)

"""
Kelly & Shen 2013 Eddington ratio distribution
"""

with open(currentPath + '../lookup_tables/bh_data/Kelly_Shen_2013/KellyShen2013_fEdd.pkl', 'r') as mydata:
	ks13 = pickle.load(mydata)

log_f_Edd = ks13['log_f_Edd']
averageEddDistribution = ks13['averageDistribution']

p_log_f_Edd = interp1d(log_f_Edd, averageEddDistribution/np.max(averageEddDistribution), bounds_error=False, fill_value=0)

def drawEddingtonRatios(nvals):
	"""
	Use rejection-comparison to draw Eddington ratios
	"""

	prop_log_f_Edd = log_f_Edd[0] + np.random.random(nvals)*(log_f_Edd[-1]-log_f_Edd[0])
	probabilities = p_log_f_Edd(prop_log_f_Edd)
	dice = np.random.random(nvals)
	rejections = np.where(dice > probabilities)[0]
	while len(rejections) > 0:
		newProps = log_f_Edd[0] + np.random.random(len(rejections))*(log_f_Edd[-1]-log_f_Edd[0])
		newProbs = p_log_f_Edd(newProps)
		dice = np.random.random(len(rejections))
		prop_log_f_Edd[rejections] = newProps
		probabilities[rejections] = newProbs
		rejections = rejections[np.where(dice > newProbs)[0]]
	return 10**prop_log_f_Edd

def randomWalkfEdd(fEdd, timeSteps, fEdd_dexperGyr):
	"""
	Move the Eddington ratio like a walker of a Markov Chain
	"""

	nvals = len(fEdd)
	possibleRanges = [np.maximum(np.log10(fEdd) - fEdd_dexperGyr*timeSteps, np.full(nvals, log_f_Edd[0])), \
	np.minimum(np.log10(fEdd) + fEdd_dexperGyr*timeSteps, np.full(nvals, log_f_Edd[-1]))]
	proposedLogfEdd = np.random.random(size=nvals) * (possibleRanges[1] - possibleRanges[0]) + possibleRanges[0]
	probabilities = p_log_f_Edd(proposedLogfEdd) / np.maximum(p_log_f_Edd(possibleRanges[0]), p_log_f_Edd(possibleRanges[1]))
	dice = np.random.random(nvals)
	rejections, = np.where(dice > probabilities)
	while len(rejections) > 0:
		newProps = np.random.random(size=len(rejections)) * (possibleRanges[1][rejections] - possibleRanges[0][rejections]) + possibleRanges[0][rejections]
		#Division here to renormalize to joint distribution.  Assuming monotonic in interval.
		newProbs = p_log_f_Edd(newProps) / np.maximum(p_log_f_Edd(possibleRanges[0][rejections]), p_log_f_Edd(possibleRanges[1][rejections]))
		dice = np.random.random(len(rejections))
		proposedLogfEdd[rejections] = newProps
		probabilities[rejections] = newProbs
		rejections = rejections[np.where(dice > newProbs)[0]]
	return 10**proposedLogfEdd

"""
Eddington ratio distributions used in Tucci & Volonteri (2016)
"""

def _log_f_crit_typeI(z):
	return np.maximum(-1.9 + 0.45*z, np.log10(0.03))

def _sigma_typeI(z):
	return np.maximum(1.03 - 0.15*z, 0.6)

def distribution_typeI(log_f_Edd, z):
	"""
	Log-normal that is fit to Kelly & Shen (2013)

	Re-'normalized' so that its maximum value is 1 at each redshift for rejection-comparison.
	"""
	
	log_f_crit = _log_f_crit_typeI(z)
	sigma = _sigma_typeI(z)

	return np.exp(-(log_f_Edd - log_f_crit)**2 / 2 / sigma**2)

def distribution_typeII(log_f_Edd, z, log_f_range=(-3.9,0.0), slope0=-0.38, slopeEvolution=0.0):
	"""
	Agrees with low-z, made up for high-z to flatten it.

	Re-'normalized' so that its maximum value is 1 at each redshift for rejection-comparison.
	"""

	slope = slope0 + z*slopeEvolution

	#Note:  Jones+2016 use an exponential cutoff, but I don't like it--with a finite timestep you can have giant spurts.
	#This is the original code.
	#return (10**(log_f_Edd - log_f_range[0]))**slope * np.exp(-(10**log_f_Edd-10**log_f_range[0])/10**log_f_range[1])

	output = (10**log_f_Edd)**slope

	#Normalize by dividing by maximum value.
	output[slope < 0] /= (10**log_f_range[0])**slope[slope < 0]
	output[slope > 0] /= (10**log_f_range[-1])**slope[slope > 0]

	return output

def draw_typeI(nvals, z, maxSigma=2.5):
        """
        Use rejection-comparison to draw Eddington ratios
        """

	z = np.minimum(z, 4.0)

        logSpread = maxSigma*_sigma_typeI(z)
        logCenter = _log_f_crit_typeI(z)
        logBounds = np.vstack([logCenter-logSpread, logCenter+logSpread])
        logProposedfEdd = logBounds[0,:] + np.random.random(nvals)*(logBounds[1,:]-logBounds[0,:])
        probabilities = distribution_typeI(logProposedfEdd, z)
        dice = np.random.random(nvals)
        rejections = dice > probabilities
        while np.any(rejections):
		logProposedfEdd[rejections] = np.random.random(size=sum(rejections)) * (logBounds[1,rejections]-logBounds[0,rejections]) + logBounds[0,rejections]
		probabilities[rejections] = distribution_typeI(logProposedfEdd[rejections], z[rejections])
                dice = np.random.random(sum(rejections))
                rejections[rejections] = dice > probabilities[rejections]
        return 10**logProposedfEdd

def draw_typeII(nvals, z, logBounds=(-3.9,0.0), slope0=-0.38):
	#Note:  z doesn't actually do anything.

	cdf_values = np.random.random(size=nvals)
	return (((10**logBounds[1])**slope0 - (10**logBounds[0])**slope0) * cdf_values + (10**logBounds[0])**slope0)**(1.0/slope0)

'''
#Rejection comparison is unnecessarily slow.  Using CDF sampling now.
def draw_typeII_old(nvals, z, logBounds=(-3.9,0.0), slope0=-0.38):
        """
        Use rejection-comparison to draw Eddington ratios
        """

        z = np.minimum(z, 4.0)

        logProposedfEdd = logBounds[0] + np.random.random(nvals)*(logBounds[1]-logBounds[0])
        probabilities = distribution_typeII(logProposedfEdd, z, log_f_range=logBounds, slope0=slope0)
        dice = np.random.random(nvals)
        rejections = dice > probabilities
        while np.any(rejections):
                logProposedfEdd[rejections] = np.random.random(size=sum(rejections)) * (logBounds[1]-logBounds[0]) + logBounds[0]
                probabilities[rejections] = distribution_typeII(logProposedfEdd[rejections], z[rejections], log_f_range=logBounds, slope0=slope0)
                dice = np.random.random(sum(rejections))
                rejections[rejections] = dice > probabilities[rejections]
        return 10**logProposedfEdd
'''

def randomWalk_typeI(fEdd, timeSteps, fEdd_MyrPerDex, z, maxSigma=2.5):
        """
        Move the Eddington ratio like a walker of a Markov Chain
        """

	z = np.minimum(z, 4.0)

	#Propose, calculate probabilities
        nvals = len(fEdd)
	logSpread = maxSigma*_sigma_typeI(z)
	logCenter = _log_f_crit_typeI(z)
	logBounds = np.vstack([logCenter-logSpread, logCenter+logSpread])
        logPossibleRanges = [np.maximum(np.log10(fEdd) - timeSteps*1e3/fEdd_MyrPerDex, logBounds[0,:]), \
	np.minimum(np.log10(fEdd) + timeSteps*1e3/fEdd_MyrPerDex, logBounds[1,:])]
        logProposedfEdd = np.random.random(size=nvals) * (logPossibleRanges[1] - logPossibleRanges[0]) + logPossibleRanges[0]
        probabilities = distribution_typeI(logProposedfEdd, z)

	#Renormalize to 1 if the central Eddington ratio is not included. This is meant to save time.
	logCenters = _log_f_crit_typeI(z)
	tooLow = np.less(logPossibleRanges[1], logCenters)
	tooHigh = np.less(logCenters, logPossibleRanges[0])
	if np.any(tooLow):
		lowRenormalizations = np.ones(nvals)
		lowRenormalizations[tooLow] = distribution_typeI(logPossibleRanges[1][tooLow], z[tooLow])
		probabilities[tooLow] /= lowRenormalizations[tooLow]
	if np.any(tooHigh):
		highRenormalizations = np.ones(nvals)
		highRenormalizations[tooHigh] = distribution_typeI(logPossibleRanges[0][tooHigh], z[tooHigh])
		probabilities[tooHigh] /= highRenormalizations[tooHigh]

	#Rejection comparison
        dice = np.random.random(nvals)
        rejections = np.greater(dice, probabilities)
        while np.any(rejections):
		logProposedfEdd[rejections] = np.random.random(size=sum(rejections)) * (logPossibleRanges[1][rejections] - logPossibleRanges[0][rejections]) + logPossibleRanges[0][rejections]
		probabilities[rejections] = distribution_typeI(logProposedfEdd[rejections], z[rejections])

		#Renormalization again
		if np.any(tooLow & rejections):
			probabilities[tooLow & rejections] /= lowRenormalizations[tooLow & rejections]
		if np.any(tooHigh & rejections):
			probabilities[tooHigh & rejections] /= highRenormalizations[tooHigh & rejections]
                dice = np.random.random(sum(rejections))
                rejections[np.where(rejections)[0]] = np.greater(dice, probabilities[rejections])
        return 10**logProposedfEdd

def randomWalk_typeII(fEdd, timeSteps, fEdd_MyrPerDex, z, logBounds=(-3.9,0.0)):
        """
        Move the Eddington ratio like a walker of a Markov Chain
        """

	z = np.minimum(z, 4.0)

	#Propose, calculate probabilities
        nvals = len(fEdd)
        logPossibleRanges = [np.maximum(np.log10(fEdd) - timeSteps*1e3/fEdd_MyrPerDex, logBounds[0]), \
	np.minimum(np.log10(fEdd) + timeSteps*1e3/fEdd_MyrPerDex, logBounds[1])]
        logProposedfEdd = np.random.random(size=nvals) * (logPossibleRanges[1] - logPossibleRanges[0]) + logPossibleRanges[0]
        probabilities = distribution_typeII(logProposedfEdd, z)

	#Renormalize to the lowest Eddington ratio possible.  This is meant to save time.
	renormalizations = distribution_typeII(logPossibleRanges[0], z)
	probabilities /= renormalizations

	#Rejection comparison
        dice = np.random.random(nvals)
        rejections = np.greater(dice, probabilities)
        while np.any(rejections):
		logProposedfEdd[rejections] = np.random.random(size=sum(rejections)) * (logPossibleRanges[1][rejections] - logPossibleRanges[0][rejections]) + logPossibleRanges[0][rejections]
		probabilities[rejections] = distribution_typeII(logProposedfEdd[rejections], z[rejections])

		#Renormalization again
		probabilities[rejections] /= renormalizations[rejections]

                dice = np.random.random(sum(rejections))
                rejections[np.where(rejections)[0]] = np.greater(dice, probabilities[rejections])
        return 10**logProposedfEdd

"""
Virial Mass to Velocity Dispersion
"""

#Fit power law slopes to Larkin & McLaughlin (2016) to get a broken power law.

def M2sigma_z0(M_vir, lowerSlope=0.70, upperSlope=0.16, sigmaBreak=130.0, massBreak=7e11):
	"""
	Use Larkin & McLaughlin (2016) to estimate sigma of M
	"""

	if not hasattr(M_vir, '__len__'):
		M_vir = np.array([M_vir])
	output = np.zeros(len(M_vir))
	output[M_vir < massBreak] = sigmaBreak * (M_vir[M_vir < massBreak] / massBreak)**lowerSlope
	output[M_vir >= massBreak] = sigmaBreak * (M_vir[M_vir >= massBreak] / massBreak)**upperSlope
	return output

"""
A normalized Gaussian
"""

def makeGaussianSmoothingKernel(widthInBins, maxSigma=4):

	halfRangeInBins = np.floor(widthInBins * maxSigma)
	if halfRangeInBins < 1:
		return [1]
	else:
		gaussian = lambda x: np.exp(-0.5*(float(x)/widthInBins)**2) / widthInBins / np.sqrt(2*np.pi)
		binsSampled = np.linspace(-halfRangeInBins,halfRangeInBins,num=2*halfRangeInBins+1)
		return [gaussian(x) for x in binsSampled]

"""
Map from L_bol to M_1450 calculated from Hopkins template
"""

with open(currentPath + '../lookup_tables/hopkins_agn_template/LbolToM1450.pkl', 'r') as myfile:
	mapLbolToM1450 = pickle.load(myfile)

def LbolToM1450(Lbol):
	"""
	Lbol must be in solar luminosities.
	"""

	return np.interp(np.log10(Lbol), mapLbolToM1450[0,:], mapLbolToM1450[1,:], left=mapLbolToM1450[1,0], right=mapLbolToM1450[1,-1])

"""
More complicated determinations of the peak circular velocity of a halo.
"""

def r_vir(m_vir, redshift):	
	"""
	Returns value in meters.  Takes virial mass in solar masses.
	"""

	h_ratio2 = 1.0 + cosmology.Omega_m*((1.0+redshift)**3 - 1.0)
	Delta_vir = 18.0 * np.pi**2 - 82.0 * (1.0 - cosmology.Omega_m) / h_ratio2 - 39.0 * (1.0 - cosmology.Omega_m)**2 / h_ratio2**2

	return 1e3 * constants.pc * (1166.1 * cosmology.h**2 * Delta_vir * h_ratio2 * m_vir**-1)**(-1.0/3.0)

def v_ratio2(r_ratio):
	"""
	Used for v_peak.  Input needs to be the ratio of the radius to the radius at which the density profile has a
	slope of -2.
	"""

	return ((24.0 * (r_ratio)**(11.0/45.0)) / (13.0 + 11.0*(r_ratio)**(4.0/9.0)))**5

def concentration(m_vir, redshift):
	"""
	Returns r_vir / r_-2.  Requires m_vir in solar masses.

	Only calibrated to z < 5, so we assume no evolution beyond that.
	"""
	redshift = np.minimum(redshift, 5)
	return 10**(0.537 + 0.488*np.exp(-0.718*redshift**1.08) - (0.097-0.024*redshift)*np.log10((m_vir / (1e12/cosmology.h))))

def v_peak(m_vir, redshift):
	"""
	Calculated for Dehnen & McLaughlin halos.  Takes m_vir in solar masses, returns circular velocity in km/s
	"""

	v_ratio2_peak = 1.08787 #= v_ratio2(2.28732)
	v_ratio2_vir = v_ratio2(concentration(m_vir, redshift))
	return np.sqrt(constants.G * m_vir * constants.M_sun / r_vir(m_vir, redshift) * v_ratio2_peak / v_ratio2_vir) / 1e3

def Mcrit(z, M_crit0=1e12):
	"""
	Critical mass below which supernova outflows are buoyant according to Bower et al., (2017).  This is a really large mass; one needs super-Eddington accretion for z=6 quasars to grow.
	"""

	return M_crit0 * (cosmology.Omega_m * (1+z)**3 + cosmology.Omega_l)**(-1.0/8.0)

def lumFunctUeda(luminosities, redshift, A=3.26, logLstar=45.48, gamma1=0.89, gamma2=2.13, p1=5.33, p2=-1.5, p3=-6.2, beta1=0.46, zc1=1.86, zc2=3.0, \
	logLa1=46.20, logLa2=45.67, alpha1=0.29, alpha2=-0.1):
	"""
	Return luminosity function values in h^3 Mpc^-3

	redshift must be a scalar
	"""

	#Assuming luminosities are solar.  Converting to erg/s
	luminosities = luminosities * constants.L_sun * 1e7

	#Normalization at z=0
	realNorm = A * 1e-6 * (cosmology.h/0.7)**3

	#Redshift evolution has a long, annoying parametrization
	z1array = zc1 * (luminosities/10**logLa1)**alpha1
	z1array[luminosities > 10**logLa1] = zc1
	z2array = zc2 * (luminosities/10**logLa2)**alpha2
	z2array[luminosities > 10**logLa2] = zc2

	#First slope is also luminosity dependent
	realp1 = p1 + beta1*(np.log10(luminosities)-logLa2)

	evolutionFactor = (1.0+redshift)**realp1
	part2Mask = (z1array < redshift) & (z2array >= redshift)
	part3Mask = z2array < redshift
	evolutionFactor[part2Mask] = (1.0+z1array[part2Mask])**realp1[part2Mask] * ((1.0+redshift)/(1.0+z1array[part2Mask]))**p2
	evolutionFactor[part3Mask] = (1.0+z1array[part3Mask])**realp1[part3Mask] * ((1.0+z2array[part3Mask])/(1.0+z1array[part3Mask]))**p2 * ((1.0+redshift)/(1.0+z2array[part3Mask]))**p3

	Lstar = 10**logLstar
	return realNorm * ((luminosities/Lstar)**gamma1 + (luminosities/Lstar)**gamma2)**-1 * evolutionFactor

with open(currentPath + '../lookup_tables/analyticMassFunctions.pkl', 'r') as myfile:
	analyticMassFunction = pickle.load(myfile)

amf_redshifts = analyticMassFunction['redshift']
amf_masses = analyticMassFunction['m_bh']
amf_numberDensity = analyticMassFunction['numberDensity']

def analyticMassFunction(redshift):
	closestRedshiftIndex = np.argmin(np.abs(amf_redshifts-redshift))
	return amf_masses[closestRedshiftIndex,:], amf_numberDensity[closestRedshiftIndex,:]

def convolve(histogram, dexToConvolve, dexPerBin):
	binsToConvolve = dexToConvolve / dexPerBin
	kernel = makeGaussianSmoothingKernel(binsToConvolve)
	return fftconvolve(histogram, kernel, mode='same')

"""
LCDM Power Spectrum
"""

with open(currentPath + '../lookup_tables/powerSpectra/powerSpectra.pkl', 'r') as myfile:
	lcdm_ps = pickle.load(myfile)

lcdm_k = lcdm_ps['k'] * cosmology.h
lcdm_P = lcdm_ps['powerSpectrum'] / cosmology.h**3
lcdm_z = lcdm_ps['redshift']
_calcPowerSpectrum_log = RectBivariateSpline(lcdm_z, np.log10(lcdm_k), np.log10(lcdm_P))

def calcPowerSpectrum(redshift, k):
	#Note that the units returned are Mpc^3
	return 10**_calcPowerSpectrum_log(redshift, np.log10(k), grid=False)

"""
Delay time distribution
"""

with open(currentPath + '../lookup_tables/mergerDelayTimes/mergerDelayTimeDistribution.pkl', 'r') as myfile:
	delayTimeData = pickle.load(myfile)

_inverseDelayTimeCDF = interp1d(delayTimeData['CDF'], delayTimeData['delayTime'], bounds_error=False, fill_value=(0,np.inf))

def drawMergerDelayTimes(number):
	"""
	Returns values in Gyr
	"""
	randomCDFValues = np.random.random(number)
	return _inverseDelayTimeCDF(randomCDFValues)

"""
Merger Probabilities
"""

with open(currentPath + '../lookup_tables/mergerProbabilities/mergerProbabilityTable.pkl', 'r') as myfile:
	mergerProbabilityData = pickle.load(myfile)

_mergerProbabilityInterpolation = RectBivariateSpline(0.5*(np.log10(mergerProbabilityData['massRatioEdges'][1:]) + np.log10(mergerProbabilityData['massRatioEdges'][:-1])), \
0.5*(mergerProbabilityData['logStellarMassEdges'][1:] + mergerProbabilityData['logStellarMassEdges'][:-1]), mergerProbabilityData['mergerProbability'], \
kx=1, ky=1)

def mergerProbability(m_star, q):
	m_star = np.atleast_1d(m_star)
	q = np.atleast_1d(q)

	return np.minimum(1,np.maximum(0,_mergerProbabilityInterpolation(np.log10(q), np.log10(m_star), grid=False)))

def computeMergerTime(dynamicalTime, haloMassRatio, circularity=0.5, r_ratio=0.6):
	"""
	The dynamical friction time scale from Boylan-Kolchin et al. (2008).  Make sure the
	dynamical time is estimated at 0.1 Rvir
	"""

	return dynamicalTime * 0.216 * haloMassRatio**-1.3 / np.log(1+haloMassRatio**-1) \
	* np.exp(1.9*circularity) * (r_ratio)**1.0

def calcRecoilEscapeVelocity_permanent(M_halo, z):
	"""
	Recoil velocity required to escape from the center of the halo permanently from Choksi+ 2017
	"""

	log10V0z = 0.000216*z**3 - 0.00339*z**2 + 0.0581*z + 2.10
	alphaz = -6.58e-6*z**4 + 0.000353*z**3 - 0.00538*z**2 + 0.0342*z + 0.341
	return 10**log10V0z * (M_halo/1e10)**alphaz

def calcRecoilEscapeVelocity_01t(M_halo, z):
	"""
	Recoil velocity required to escape from the center until 0.1 t_kick from Choksi+ 2017
	"""

	log10V0z = 1.08e-5*z**3 + 0.000710*z**2 + 0.0224*z + 2.12
	alphaz = 5.49e-5*z**3 - 0.00183*z**2 + 0.0243*z + 0.341
	return 10**log10V0z * (M_halo/1e10)**alphaz
