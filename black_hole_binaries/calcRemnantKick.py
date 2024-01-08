import numpy as np

#Fitted Constants
A = 1.2e4 
B = -0.93 
H = 6.9e3 
K = 6.0e4 
xi = 145.0*np.pi/180.0

def calcRemnantKick(m1, m2, a1, a2, theta1=None, theta2=None, phi1=None, phi2=None):
	"""
	Lousto et al., (2012)

	Angles are in a coordinate system where the orbital angular momentum is z.

	#NOTE:  NOT YET VECTORIZED

	"""

	#Determine mass ratio (and which hole is bigger)
	q = m2 / m1
	if q > 1:
		q = q**-1
		a1, a2 = a2, a1
	eta = q/(1+q)**2

	if theta1 is None:
		theta1 = 2*np.pi*np.random.random()
	if theta2 is None:
		theta2 = 2*np.pi*np.random.random()
	if phi1 is None:
		phi1 = 2*np.pi*np.random.random()
	if phi2 is None:
		phi2 = 2*np.pi*np.random.random()

	#Determine orientation of spin vector
	a1_par = a1*np.cos(theta1)
	a1_perp = a1*np.sin(theta1)
	a1_vector = np.array([a1_perp*np.cos(phi1), a1_perp*np.sin(phi1), a1_par])
	a2_par = a2*np.cos(theta2)
	a2_perp = a2*np.sin(theta2)
	a2_vector = np.array([a2_perp*np.cos(phi2), a2_perp*np.sin(phi2), a2_par])

	#Velocity Components
	v_m = A*eta**2*(1-q)/(1+q) * (1+B*eta)
	if (a1 != 0) or (a2 != 0):
		v_perp = H*eta**2/(1+q) * (a2_par - q*a1_par)

		Delta_vector = a2_vector - q*a1_vector   #Note:  I'm omitting coefficients because I don't need the magnitude...
		Delta_hat = Delta_vector / np.sqrt(np.dot(Delta_vector,Delta_vector))
		cosine_Theta_Delta = Delta_hat[0]        #Choosing the infall direction to be the x-direction, this is the result of a projection
		v_par = K*eta**2/(1+q) * (a2_perp - q*a1_perp) * cosine_Theta_Delta
	else:
		v_perp = 0
		v_par = 0

	kick_vector = np.array([v_m + v_perp*np.cos(xi), v_perp*np.sin(xi), v_par])

	return np.sqrt(np.dot(kick_vector,kick_vector))
