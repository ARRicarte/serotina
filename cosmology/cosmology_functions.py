"""
Cosmological distance and time functions backed by the active astropy cosmology.

All functions use the cosmology object from cosmology.py.  To switch cosmologies
call serotina.set_cosmology() which will also rebuild the z<->t lookup tables.
"""

import numpy as np
from scipy.interpolate import interp1d
from . import cosmology as cosm
import astropy.units as u


def _rebuild_tables():
    """Build z<->t interpolation tables from the current cosmology.

    Called automatically on import and whenever set_cosmology() is called.
    Uses a table rather than calling astropy directly because z2t/t2z are
    invoked inside the SAM's inner evolution loop with arrays.
    """
    global z2t_interp, t2z_interp
    z_arr = np.linspace(0, 20, 10000)
    t_arr = cosm.cosmo.age(z_arr).to('Gyr').value   # monotonically decreasing
    z2t_interp = interp1d(z_arr, t_arr, bounds_error=False, fill_value='extrapolate')
    # For t2z we need t as the x-axis (must be increasing), so reverse both arrays.
    t2z_interp = interp1d(t_arr[::-1], z_arr[::-1], bounds_error=False, fill_value='extrapolate')


_rebuild_tables()


def z2t(z):
    """Redshift -> age of universe in Gyr."""
    return z2t_interp(z)


def t2z(t):
    """Age of universe in Gyr -> redshift."""
    return t2z_interp(t)


def computeLuminosityDistance(z):
    """Luminosity distance in Mpc."""
    return cosm.cosmo.luminosity_distance(z).to('Mpc').value


def computeComovingDistance(z):
    """Comoving distance in Mpc."""
    return cosm.cosmo.comoving_distance(z).to('Mpc').value


def computeAngularDistance(z):
    """Angular diameter distance in Mpc."""
    return cosm.cosmo.angular_diameter_distance(z).to('Mpc').value


def E(z):
    """Dimensionless Hubble parameter E(z) = H(z) / H0."""
    return cosm.cosmo.efunc(z)


def computeH(z):
    """Hubble parameter H(z) in km/s/Mpc."""
    return cosm.cosmo.H(z).value


def computedVdz(z):
    """Comoving volume element dV/dz over the full sky in Mpc³."""
    return cosm.cosmo.differential_comoving_volume(z).to('Mpc3 / sr').value * 4 * np.pi


def computedzdt(z):
    """dz/dt in yr⁻¹ (positive convention used for rate calculations)."""
    return cosm.cosmo.H(z).to(1 / u.yr).value * (1.0 + z)
