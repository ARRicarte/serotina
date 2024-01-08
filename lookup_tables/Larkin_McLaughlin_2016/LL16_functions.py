import constants
import cosmology
import numpy as np

def r_vir(m_vir, redshift):
        """
        Returns value in meters.  Takes virial mass in solar masses.
        """

        h_ratio2 = 1.0 + cosmology.Omega_m*((1.0+redshift)**3 - 1.0)
        Delta_vir = 18.0 * np.pi**2 - 82.0 * (1.0 - cosmology.Omega_m) / h_ratio2 - 39.0 * (1.0 - cosmology.Omega_m)**2 / h_ratio2**2

        return 1e3 * constants.pc * (1166.1 * cosmology.h**2 * Delta_vir * h_ratio2 * m_vir**-1)**(-1.0/3.0)

def m_ratio(r_ratio):
	"""
	Used for fraction of stars to dm in an effective radius
	"""

	return ((24.0 * (r_ratio)**(4.0/9.0)) / (13.0 + 11.0*(r_ratio)**(4.0/9.0)))**5

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

def m_star(m_vir):
	"""
	Calculated only for z=0.  Both should be in solar masses.
	"""

	return m_vir * 0.0564 * ((m_vir / 7.66e11)**-1.06 + (m_vir/7.66e11)**0.556)**-1

def star_fraction_in_r(r_fraction):
	"""
	The Hernquist profile, appropriate only for early-type galaxies.

	r_fraction is the radius divided by the effective radius
	"""

	return (r_fraction / (r_fraction + 1.0/1.81527))**2

def r_eff(m_star):
	"""
	True only for z=0, and fit by eye.

	m_star in solar masses, r_eff in kpc
	"""

	return 1.5 * (m_star / 2e10)**0.1 * (1.0 + (m_star/2e10)**5)**0.1

def sigma_ap(m_halo, ejectionFactor=1.724):
	"""
	Goes from halo mass to central velocity dispersion only at z=0.

	m_halo in solar masses, output in km/s
	"""

	Mstar = m_star(m_halo)
	Reff = r_eff(Mstar) 

	#Put everything in mks
	Reff *= 1e3 * constants.pc
	Mstar *= constants.M_sun
	Rvir = r_vir(m_halo,0)
	conc = concentration(m_halo,0)

	darkMassInReff = constants.M_sun * m_halo * m_ratio(Reff/Rvir*conc) / m_ratio(conc)
	fStarVir = Mstar / (m_halo*constants.M_sun) * star_fraction_in_r(Rvir/Reff)
	fracInReff = fStarVir * star_fraction_in_r(1.0) / star_fraction_in_r(Rvir/Reff) / darkMassInReff * (m_halo*constants.M_sun)

	return np.sqrt(constants.G * Mstar / Reff) * 0.389 * np.sqrt(ejectionFactor + 0.86/fracInReff) / 1e3

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

def testplot():
	import matplotlib.pyplot as plt
	haloMasses = np.logspace(6,16,1000)
	sigmas = sigma_ap(haloMasses)
	mstars = m_star(haloMasses)
	velocities = v_peak(haloMasses,0)
	test = v2sigma_dpl(velocities)

	plt.loglog(sigmas, velocities, label='Analytic')
	plt.loglog(test, velocities, color='r', ls='--', label='DPL Fit')
	with open('vpeak_sigma_LL16_fig5.1_blue.dat', 'r') as myfile:
		data = np.loadtxt(myfile)
	plt.loglog(data[:,0], data[:,1], ls=':', color='orange', label='Trace')
	plt.xlabel('$\sigma$')
	plt.ylabel('$V_p$')
	plt.legend()
	plt.show()
	
def makeSaveFile(output='v2sigma.pkl'):
	import cPickle as pickle

	haloMasses = np.logspace(6,16,1000)
        sigmas = sigma_ap(haloMasses)
        velocities = v_peak(haloMasses,0)
	outDict = {'V_peak': velocities, 'sigma': sigmas}
	with open(output, 'w') as myfile:
		pickle.dump(outDict, myfile)

if __name__ == '__main__':
	testplot()
	#makeSaveFile()
