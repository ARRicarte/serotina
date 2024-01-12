"""
Mostly other things to overplot on figures.
"""

import pickle
import numpy as np

"""
Map from L_bol to M_1450 calculated from Hopkins template
"""

with open(currentPath + '../lookup_tables/hopkins_agn_template/LbolToM1450.pkl', 'rb') as myfile:
    mapLbolToM1450 = pickle.load(myfile, encoding='latin1')

def LbolToM1450(Lbol):
    """
    Lbol must be in solar luminosities.
    """

    return np.interp(np.log10(Lbol), mapLbolToM1450[0,:], mapLbolToM1450[1,:], left=mapLbolToM1450[1,0], right=mapLbolToM1450[1,-1])

"""
Luminosity functions from Ueda.
"""

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

"""
Fake but intuitive BH mass functions assuming M_h, z -> sigma -> M_bh
"""

with open(currentPath + '../lookup_tables/analyticMassFunctions.pkl', 'rb') as myfile:
    analyticMassFunction = pickle.load(myfile, encoding='latin1')

amf_redshifts = analyticMassFunction['redshift']
amf_masses = analyticMassFunction['m_bh']
amf_numberDensity = analyticMassFunction['numberDensity']

def analyticMassFunction(redshift):
    closestRedshiftIndex = np.argmin(np.abs(amf_redshifts-redshift))
    return amf_masses[closestRedshiftIndex,:], amf_numberDensity[closestRedshiftIndex,:]

def M2sigma_z0(M_vir, lowerSlope=0.70, upperSlope=0.16, sigmaBreak=130.0, massBreak=7e11):
    """
    Use Larkin & McLaughlin (2016) to estimate sigma of virial mass
    """

    if not hasattr(M_vir, '__len__'):
        M_vir = np.array([M_vir])
    output = np.zeros(len(M_vir))
    output[M_vir < massBreak] = sigmaBreak * (M_vir[M_vir < massBreak] / massBreak)**lowerSlope
    output[M_vir >= massBreak] = sigmaBreak * (M_vir[M_vir >= massBreak] / massBreak)**upperSlope
    return output

