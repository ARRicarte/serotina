import numpy as np

#Fitted coefficients
s4 = -0.129
s5 = -0.384
t0 = -2.686
t2 = -3.454
t3 = 2.353

def calcRemnantSpin(m1, m2, a1, a2, theta1=None, theta2=None, phi1=None, phi2=None, spinMax=1.0):
	"""
	Rezzolla et al., (2008)

	Assumptions:  mass loss to GW is negligible, spin vector is sum of initial spins
	plus a vector l that is aligned with the orbit.
	
	#Note:  Lousto et al. 2010 have formulae for mass and spin of remnants.
	"""

	#Input sanitization.  Want vectors.
	m1 = np.atleast_1d(m1).astype(float)
	m2 = np.atleast_1d(m2).astype(float)
	a1 = np.atleast_1d(a1).astype(float)
	a2 = np.atleast_1d(a2).astype(float)

	#Choose random angles by default.
	#Note even sampling of theta with respect to solid angle.  Negative thetas correspond to retrograde orbits.
	if theta1 is None:
		theta1 = np.arccos(1.0-2*np.random.random(len(m1))) * np.random.choice([1.0,-1.0], len(m1))
	if theta2 is None:
		theta2 = np.arccos(1.0-2*np.random.random(len(m1))) * np.random.choice([1.0,-1.0], len(m1))
	if phi1 is None:
		phi1 = 2*np.pi*np.random.random(len(m1))
	if phi2 is None:
		phi2 = 2*np.pi*np.random.random(len(m1))
	theta1 = np.atleast_1d(theta1)
	theta2 = np.atleast_1d(theta2)
	phi1 = np.atleast_1d(phi1)
	phi2 = np.atleast_1d(phi2)

	#Determine mass ratio (and which hole is bigger)
	q = m2 / m1
	orderBackwards = q > 1
	q[orderBackwards] = 1/q[orderBackwards]

	#Symmetric mass ratio; useful construction.
	nu = q/(1+q)**2

	#We'll need to flip the order of spins and angles too.  From now on, we don't actually need the masses though, so they are unchanged.
	a1_substitute = a2[orderBackwards]
	a2_substitute = a1[orderBackwards]
	a1[orderBackwards] = a1_substitute
	a2[orderBackwards] = a2_substitute

	theta1_substitute = theta2[orderBackwards]
	theta2_substitute = theta1[orderBackwards]
	theta1[orderBackwards] = theta1_substitute
	theta2[orderBackwards] = theta2_substitute

	phi1_substitute = phi2[orderBackwards]
	phi2_substitute = phi1[orderBackwards]
	phi1[orderBackwards] = phi1_substitute
	phi2[orderBackwards] = phi2_substitute

	#Construct spin vectors
	a_vector_1 = np.vstack([a1 * np.sin(theta1) * np.cos(phi1), a1 * np.sin(theta1) * np.sin(phi1), a1 * np.cos(theta1)])
	a_vector_2 = np.vstack([a2 * np.sin(theta2) * np.cos(phi2), a2 * np.sin(theta2) * np.sin(phi2), a2 * np.cos(theta2)])

	#alpha = angle between a1 and a2, beta = angle between a1 and l, gamma = angle between a2 and l
	cosAlpha = np.ones_like(a1)
	twoNonzeroSpins = (a1 != 0) & (a2 != 0)
	if np.any(twoNonzeroSpins):
		#This is just an explicit dot product.
		cosAlpha[twoNonzeroSpins] = (a_vector_1[0,twoNonzeroSpins]*a_vector_2[0,twoNonzeroSpins] + a_vector_1[1,twoNonzeroSpins]*a_vector_2[1,twoNonzeroSpins] + a_vector_1[2,twoNonzeroSpins]*a_vector_2[2,twoNonzeroSpins]) / a1[twoNonzeroSpins] / a2[twoNonzeroSpins]
	cosBeta = np.cos(theta1)
	cosGamma = np.cos(theta2)

	#Magnitude of vector of orbital angular momentum
	l = s4 / (1+q**2)**2 * (a1**2 + a2**2*q**4 + 2*a1*a2*q**2*cosAlpha) + \
	(s5*nu + t0 + 2)/(1+q**2) * (a1*cosBeta + a2*q**2*cosGamma) + \
	2*np.sqrt(3) + t2*nu + t3*nu**2

	#Final equation
	a_fin = np.zeros_like(m1)

	#Sometimes I have a negative number under the radical.  For safety, I just set spin to the average...
	#I'm guessing that the fitting function happens to break with the parameters given.
	underRadical = a1**2 + a2**2*q**4 + 2*a2*a1*q**2*cosAlpha + \
	2*(a1*cosBeta + a2*q**2*cosGamma)*l*q + l**2*q**2
	a_fin[underRadical>=0] = 1.0 / (1+q[underRadical>0])**2 * np.sqrt(underRadical[underRadical>0])
	a_fin[underRadical<0] = (a1[underRadical<0]+a2[underRadical<0]/2)

	#There's no physical reason to have a maximum spin, but for numerical reasons, it may be necessary to set it equal to that from accretion.
	a_fin[a_fin > spinMax] = spinMax

	return a_fin
