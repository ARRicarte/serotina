"""
ARR 11.09.15

Miscellaneous functions imported into sam.py or tree.py
"""

import numpy as np
import os
from .. import cosmology
from .. import constants
import pickle
from scipy.integrate import quad
from scipy.interpolate import interp1d, interp2d, InterpolatedUnivariateSpline, RectBivariateSpline, griddata
from .. import config
currentPath = os.path.abspath(os.path.dirname(__file__)) + '/'

#####################
##Dark matter halos##
#####################

def darkMatterHaloSpinProbabilityDistribution(spin, lbar=0.05, sigma=0.5, sigmaMax=6):
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
			probability = darkMatterHaloSpinProbabilityDistribution(proposition)
			dice = np.random.random()
			if dice <= probability:
				break
		return proposition
	else:
		propositions = np.random.random(nvals)*sigma*sigmaMax
		probabilities = darkMatterHaloSpinProbabilityDistribution(propositions)
		dice = np.random.random(nvals)
		rejections = np.where(dice > probabilities)[0]
		while len(rejections) > 0:
			newProps = np.random.random(len(rejections))*sigma*sigmaMax
			newProbs = darkMatterHaloSpinProbabilityDistribution(newProps)
			dice = np.random.random(len(rejections))
			propositions[rejections] = newProps
			probabilities[rejections] = newProbs
			rejections = rejections[np.where(dice > newProbs)[0]]
		return propositions

def M2v(masses, redshifts):
	"""
	Convert masses into circular velocities with the spherical collapse model.

	See Barkana & Loeb (2001)
	"""
	d = cosmology.Omega_m * (1+redshifts)**3 / (cosmology.Omega_m * (1+redshifts)**3 + \
	cosmology.Omega_l + (1.0 - cosmology.Omega_l - cosmology.Omega_m)*(1+redshifts)**2) - 1
	Delta_c = 18*np.pi**2 + 82*d - 39*d**2
	return 23.4 * (masses / 1e8 / cosmology.h)**(1.0/3.0) * (cosmology.Omega_m/(d+1) * Delta_c/(18*np.pi**2))**(1.0/6.0) * \
	np.sqrt((1.0+redshifts)/10)

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

def calcDeltaVir(z):
	"""
	Compute the overdensity parameter Delta_vir at different redshift.
	"""
	
	Omega_m_at_z = cosmology.Omega_m*(1+z)**3 / (cosmology.Omega_m*(1+z)**3 + cosmology.Omega_l + \
	(1.0-cosmology.Omega_m-cosmology.Omega_l)*(1+z)**2)
	x = Omega_m_at_z - 1
	return (18*np.pi**2 + 82*x - 39*x**2) / Omega_m_at_z

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

"""
Halo mass function
"""

with open(currentPath + '../lookup_tables/massFunction.pkl', 'rb') as myfile:
	massFunctionDict = pickle.load(myfile, encoding='latin1')
numberDensityMasses = massFunctionDict['mass']
numberDensityValues = massFunctionDict['dndlog10m']
numberDensityRedshifts = massFunctionDict['redshift']
_calcHaloNumberDensity_log = RectBivariateSpline(numberDensityRedshifts, np.log10(numberDensityMasses), numberDensityValues)

def calcHaloNumberDensity(haloMass, redshift):
	"""
	The halo mass function. 

	Doing interpolation in log space
	"""
	return _calcHaloNumberDensity_log(redshift, np.log10(haloMass*cosmology.h), grid=False)

"""
LCDM Halo Power Spectrum
"""

with open(currentPath + '../lookup_tables/powerSpectra/powerSpectra.pkl', 'rb') as myfile:
	lcdm_ps = pickle.load(myfile, encoding='latin1')

lcdm_k = lcdm_ps['k'] * cosmology.h
lcdm_P = lcdm_ps['powerSpectrum'] / cosmology.h**3
lcdm_z = lcdm_ps['redshift']
_calcPowerSpectrum_log = RectBivariateSpline(lcdm_z, np.log10(lcdm_k), np.log10(lcdm_P))

def calcPowerSpectrum(redshift, k):
	#Note that the units returned are Mpc^3
	return 10**_calcPowerSpectrum_log(redshift, np.log10(k), grid=False)

############################################################
##Alternative, some arguably flawed, definitions of sigma.##
############################################################

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

with open(currentPath + '../lookup_tables/Larkin_McLaughlin_2016/v2sigma.pkl', 'rb') as myfile:
	v2sigma_dict = pickle.load(myfile, encoding='latin1')

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
##Sigma definition used in publication##
########################################

with open(currentPath + '../lookup_tables/velocityDispersion/velocityDispersion.pkl', 'rb') as myfile:
	vd_dict = pickle.load(myfile, encoding='latin1')

velocityDispersionTable = vd_dict['sigma']
vd_redshift = vd_dict['redshift']
vd_logHaloMass = vd_dict['logHaloMass']

_logInterpVelocityDispersion = RectBivariateSpline(vd_logHaloMass, vd_redshift, velocityDispersionTable)

def velocityDispersion(m_halo, z):
	return _logInterpVelocityDispersion(np.log10(m_halo), z, grid=False)

############################################
##Eddington luminosity and accretion rate.##
############################################

def eddingtonLum(m_bh):
	"""
	Returns the Eddington luminosity in units of solar luminosities.
	"""
	return 4 * np.pi * constants.G * m_bh * constants.M_sun * constants.m_p * constants.c / constants.sigma_T / constants.L_sun

def eddingtonLimit(m_bh, efficiency=0.1):
	"""
	Returns the Eddington limit of accretion in units of solar masses per year.
	"""
	return 4 * np.pi * constants.G * m_bh * constants.m_p / (efficiency * constants.sigma_T * constants.c) * constants.yr

##########################################################################
##Interpreter of dark matter merger trees formatted in a particular way.##
##########################################################################

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

#################################
##Overdensity Peak Calculations##
#################################

with open(currentPath + '../lookup_tables/sigmaPeaks.pkl', 'rb') as myfile:
	sigmaData = pickle.load(myfile, encoding='latin1')
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

#########################################################################
##Metal pollution estimates, optionally used for seeding, though dated.##
#########################################################################

with open(currentPath + '../lookup_tables/scannapieco2003.pkl', 'rb') as myfile:
	scannapieco2003 = pickle.load(myfile, encoding='latin1')

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
	
###########################################################
##Eddington ratio distributions: Tucci & Volonteri (2016)##
###########################################################

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

##############################
##Supernova-regulated growth##
##############################

def Mcrit(z, M_crit0=1e12):
	"""
	Critical mass below which supernova outflows are buoyant according to Bower et al., (2017).  This is a really large mass; one needs super-Eddington accretion for z=6 quasars to grow.
	"""

	return M_crit0 * (cosmology.Omega_m * (1+z)**3 + cosmology.Omega_l)**(-1.0/8.0)

#####################
##BH Merger Recipes##
#####################

#None of this is used by default.

#First, some recipes based on Tremmel et al. (2017).

"""
Delay time distribution
"""

with open(currentPath + '../lookup_tables/mergerDelayTimes/mergerDelayTimeDistribution.pkl', 'rb') as myfile:
	delayTimeData = pickle.load(myfile, encoding='latin1')

_inverseDelayTimeCDF = interp1d(delayTimeData['CDF'], delayTimeData['delayTime'], bounds_error=False, fill_value=(0,np.inf))

def drawMergerDelayTimes(number):
	"""
	Returns values in Gyr
	"""
	randomCDFValues = np.random.random(number)
	return _inverseDelayTimeCDF(randomCDFValues)


with open(currentPath + '../lookup_tables/mergerProbabilities/mergerProbabilityTable.pkl', 'rb') as myfile:
	mergerProbabilityData = pickle.load(myfile, encoding='latin1')

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

#######################
##Retrograde Flipping##
#######################

def computeAccretionAlignmentFlipTime(M_bh, z, parameters=[0.1]):

	M_bh = np.atleast_1d(M_bh)
	z = np.atleast_1d(z)

	#Scale the alignment flip time using a free parameter, times the dynamical time.  May add more parameters later.
	timescale = t_dyn(z) * parameters[0]

	#Wait times for Poisson distributed events follow an exponential.  
	waitTimes = -timescale*np.log(1.0 - np.random.uniform(size=len(M_bh)))

	return waitTimes

