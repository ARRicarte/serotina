"""
Cosmological parameters and the active cosmology object.

By default, uses Planck 2015 values (Planck Collaboration 2015, Table 4, TT+lowP)
matching the original hardcoded constants in serotina.

To run with a different cosmology, call set_cosmology() before initializing the SAM:

    import serotina
    from astropy.cosmology import Planck18
    serotina.set_cosmology(Planck18)
"""

from astropy.cosmology import FlatLambdaCDM

# Exact Planck 2015 values from the original cosmology.py (ARR 11.06.15)
cosmo = FlatLambdaCDM(H0=67.74, Om0=0.3089, Ob0=0.0486, name='Planck2015_serotina')


def set_cosmology(new_cosmo):
    """Replace the active cosmology with any astropy cosmology object.

    Also refreshes the module-level constant aliases and rebuilds the z<->t
    lookup tables in cosmology_functions so all downstream code sees the change.
    """
    global cosmo
    cosmo = new_cosmo
    _refresh_constants()
    from . import cosmology_functions
    cosmology_functions._rebuild_tables()


def _refresh_constants():
    global Omega_m, Omega_l, H_0, h, Omega_b
    Omega_m = cosmo.Om0
    Omega_l = cosmo.Ode0
    H_0 = cosmo.H0.value
    h = H_0 / 100.0
    Omega_b = cosmo.Ob0 if cosmo.Ob0 is not None else 0.0486


_refresh_constants()

# Not derivable from a standard FlatLambdaCDM object; kept for backward compatibility.
sigma_8 = 0.8159
n = 0.9667
