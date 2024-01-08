def calcISCO(a, mu=1):
	"""
	Return the ISCO in geometrized units.
	"""
	z1 = 1.0 + (1.0-a**2)**(1.0/3.0) * ((1.0+a)**(1.0/3.0) + (1.0-a)**(1.0/3.0))
	z2 = (3*a**2 + z1**2)**0.5
	return 3 + z2 - mu*((3-z1)*(3+z1+2*z2))**0.5
