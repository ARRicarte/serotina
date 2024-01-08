import numpy as np

def calcISCO(spin):
	"""
	Return the ISCO in geometrized units for a Kerr black hole.
	"""

	#Split magnitude and direction.
	spin = np.atleast_1d(spin)
	spin_mag = np.abs(spin)
	mu = np.sign(spin)

	#Following Bardeen 1970...
	z1 = 1.0 + (1.0-spin_mag**2)**(1.0/3.0) * ((1.0+spin_mag)**(1.0/3.0) + (1.0-spin_mag)**(1.0/3.0))
	z2 = np.sqrt(3*spin_mag**2 + z1**2)
	return 3 + z2 - mu*((3-z1) * (3 + z1 + 2*z2))**0.5
