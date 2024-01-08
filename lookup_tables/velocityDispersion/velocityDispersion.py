import constants
import smhm
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

def star_fraction_in_r(r_fraction):
	"""
	The Hernquist profile, appropriate only for early-type galaxies.

	r_fraction is the radius divided by the effective radius
	"""

	return (r_fraction / (r_fraction + 1.0/1.81527))**2

def r_eff(m_star, redshift):
	"""
	Mosleh+16, red galaxies.  Redshift extension using power law from Huertas-Company et al., (2013)	

	m_star in solar masses, r_eff in kpc
	"""

	#Redshift evolution devised to reproduce trends of Huertas-Company
	
	#Matches trends exactly
	#logMCrit = 10.75
	#redshiftEvolutionSlopes = np.maximum(0.0, 0.66/0.85*(np.log10(m_star)-logMCrit)+0.34)

	#Contorted to make sigma bigger at high z
	#logMCrit = 11.0 - 1.5 * np.log10((1.0 + redshift))
	#redshiftEvolutionSlopes = np.maximum(0.0, 0.9/(11.6-logMCrit) * (np.log10(m_star)-logMCrit)+0.1)

	#Assuming no growth below critical mass.  This version prevents sigma from shrinking too much.
	logMCrit = 10.75
	redshiftEvolutionSlopes = np.maximum(0.0, 1.0/0.85*(np.log10(m_star)-logMCrit))

	redshiftFactor = (1.0 + redshift)**-redshiftEvolutionSlopes

	return 10**-0.314 * (m_star)**0.042 * (1.0 + m_star/10**10.537)**0.76 * redshiftFactor

def sigma_ap(m_halo, redshift, ejectionFactor=1.724):
	"""
	Goes from halo mass to central velocity dispersion only at z=0.

	m_halo in solar masses, output in km/s
	"""

	Mstar = 10**smhm.logMstar(np.log10(m_halo), redshift)
	Reff = r_eff(Mstar, redshift) 

	#Put everything in mks
	Reff *= 1e3 * constants.pc
	Mstar *= constants.M_sun
	Rvir = r_vir(m_halo,redshift)
	conc = concentration(m_halo,redshift)

	darkMassInReff = constants.M_sun * m_halo * m_ratio(Reff/Rvir*conc) / m_ratio(conc)
	fStarVir = Mstar / (m_halo*constants.M_sun) * star_fraction_in_r(Rvir/Reff)
	fracInReff = fStarVir * star_fraction_in_r(1.0) / star_fraction_in_r(Rvir/Reff) / darkMassInReff * (m_halo*constants.M_sun)

	return np.sqrt(constants.G * Mstar / Reff) * 0.389 * np.sqrt(ejectionFactor + 0.86/fracInReff) / 1e3

def testplot():
	import matplotlib.pyplot as plt

	m_halo = np.logspace(9,15,100)
	redshift = np.linspace(0,5,6)
	
	fig, ax = plt.subplots()
	ax.set_xscale('log')
	ax.set_yscale('log')

	for z in redshift:
		v_array = v_peak(m_halo, z)
		s_array = sigma_ap(m_halo, z)
		ax.plot(m_halo, s_array)
		
	ax.plot(v_array, v_array/np.sqrt(2), ls='--', color='k')
	ax.set_xlabel(r'$V_c \ [\mathrm{km} \, \mathrm{s}^{-1}]$', fontsize=14)
	ax.set_ylabel(r'$\sigma \ [\mathrm{km} \, \mathrm{s}^{-1}]$', fontsize=14)

	fig.tight_layout()
	fig.show()

if __name__ == '__main__':
	testplot()
