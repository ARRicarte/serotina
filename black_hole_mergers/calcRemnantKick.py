import numpy as np

#Fitted Constants
A = 1.2e4   #km/s
B = -0.93   #unitless
H = 6.9e3   #km/s
K = 6.0e4   #km/s
xi = 145.0*np.pi/180.0  #The angle between the unequal mass and spin contribution to the recoil velocity in the orbital plane.  Lousto liked 145 degrees.

def calcRemnantKick(m1, m2, a1, a2, theta1=None, theta2=None, phi1=None, phi2=None):
	"""
	Lousto et al. (2010,2012)

	Angles are in a coordinate system where the orbital angular momentum is z.
	We don't care about the direction; we return only the magnitude.

	arg m1:  Mass, arbitrary units
	arg m2:  Mass, arbitrary units
	arg a1:  spin, unitless
	arg a2:  spin, unitless
	kwarg theta1:  angle between orbital angular momentum vector and spin vector, defaults to random, radians 
	kwarg phi1:  angle of the spin vector projected onto the orbital plane, defaults to random, radians
	kwarg theta2:  angle between orbital angular momentum vector and spin vector, defaults to random, radians 
	kwarg phi2:  angle of the spin vector projected onto the orbital plane, defaults to random, radians
	"""

	#Input sanitization.  Want vectors.
	m1 = np.atleast_1d(m1).astype(float)
	m2 = np.atleast_1d(m2).astype(float)
	a1 = np.atleast_1d(a1).astype(float)
	a2 = np.atleast_1d(a2).astype(float)

	#Choose random angles by default.
	if theta1 is None:
		theta1 = 2*np.pi*np.random.random(len(m1))
	if theta2 is None:
		theta2 = 2*np.pi*np.random.random(len(m1))
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
	eta = q/(1+q)**2
	
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

	#Reconstruct spin vectors based on input angles.
	a1_par = a1*np.cos(theta1)
	a1_perp = a1*np.sin(theta1)
	a1_vector = np.vstack([a1_perp*np.cos(phi1), a1_perp*np.sin(phi1), a1_par])
	a2_par = a2*np.cos(theta2)
	a2_perp = a2*np.sin(theta2)
	a2_vector = np.vstack([a2_perp*np.cos(phi2), a2_perp*np.sin(phi2), a2_par])

	#Recoil coming from asymmetric mass distribution.
	v_m = A*eta**2*(1-q)/(1+q) * (1+B*eta)

	#Recoil coming from spins.  Zero for Schwarzschild.
	v_perp = H*eta**2/(1+q) * (a2_par - q*a1_par)
	Delta_vector = a2_vector - q*a1_vector   #Note:  I'm omitting coefficients because I don't need the real magnitude...
	magnitudeForRenormalization = np.array([np.sqrt(np.dot(Delta_vector[:,i],Delta_vector[:,i])) for i in range(Delta_vector.shape[1])])
	Delta_hat = Delta_vector / magnitudeForRenormalization
	cosine_Theta_Delta = Delta_hat[0,:]	#Choosing the infall direction to be the x-direction, this is the result of a projection
	v_par = K*eta**2/(1+q) * (a2_perp - q*a1_perp) * cosine_Theta_Delta

	#Put everything together.
	kick_vector = np.vstack([v_m + v_perp*np.cos(xi), v_perp*np.sin(xi), v_par])

	return np.array([np.sqrt(np.dot(kick_vector[:,i],kick_vector[:,i])) for i in range(kick_vector.shape[1])])
