"""
Mostly following the equations of Lousto et al. (2010).

I was unable to implement their remnant spin formulae, and I suspect an error in the published equations.
For this reason, alternative formulae from Rezzolla et al. (2008) are included for remnant spin.
"""
import numpy as np

########################
##Lousto et al. (2010)##
########################

#Fitted constants for recoil.
A = 1.2e4   #km/s
B = -0.93   #unitless
H = 6.9e3   #km/s
K = 6.0e4   #km/s
xi = 145.0*np.pi/180.0  #The angle between the unequal mass and spin contribution to the recoil velocity in the orbital plane.  Lousto liked 145 degrees.

#Fitted constants for remnant mass
E2 = 0.341
E3 = 0.522
ES = 0.673
EDelta = -0.3689 #Fit consistent with 0
EA = -0.0136	 #Fit consistent with 0
EB = 0.045
EC = 0.0
ED = 0.2611	  #Fit consistent with 0
EE = 0.0959
EF = 0.0

#Fitted constants for remnant spin
J2 = -2.81
J3 = 1.69
JA = -2.9667
JB = -1.7296
JDelta = 0.0
JM = 0.0
JMDelta = 0.0  #This is not even discussed in the paper, but the others are 0, so I set it to 0.
JS = 0.0
JMS = 0.0

##########################
##Rezzolla et al. (2008)##
##########################

#Fitted coefficients
s4 = -0.129
s5 = -0.384
t0 = -2.686
t2 = -3.454
t3 = 2.353

def calcRemnantProperties(m1, m2, a1, a2, theta1=None, theta2=None, phi1=None, phi2=None, spinMax=1.0, zeroMysteryThetas=False, spinMode='Rezzolla08'):

	"""
	The authors expect unreliable results as a>0.8.
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
	theta1 = np.atleast_1d(theta1).astype(float)
	theta2 = np.atleast_1d(theta2).astype(float)
	phi1 = np.atleast_1d(phi1).astype(float)
	phi2 = np.atleast_1d(phi2).astype(float)

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

	#Construct spin vectors
	a1_par = a1*np.cos(theta1)
	a1_perp = a1*np.sin(theta1)
	a_vector_1 = np.vstack([a1_perp*np.cos(phi1), a1_perp*np.sin(phi1), a1_par])
	a2_par = a2*np.cos(theta2)
	a2_perp = a2*np.sin(theta2)
	a_vector_2 = np.vstack([a2_perp*np.cos(phi2), a2_perp*np.sin(phi2), a2_par])
	a1_dot_a2 = np.array([np.dot(a_vector_1[:,i],a_vector_2[:,i]) for i in range(a_vector_1.shape[1])])

	#Construct weird "Thetas" that appear in the paper.  Taking angles with respect to x-axis.
	Delta_vector = a_vector_2*m2 - a_vector_1*m1  #Extra normalization factors omitted. Also equal to what I would have called "Minus_vector"
	Delta_norm = np.linalg.norm(Delta_vector, axis=0)
	noDelta = Delta_norm == 0
	Theta_Delta = np.zeros_like(a1)
	if np.any(noDelta):
		Theta_Delta[noDelta] = 0
	if np.any(~noDelta):
		Delta_hat_x = Delta_vector[0,~noDelta] / Delta_norm[~noDelta]
		Theta_Delta[~noDelta] = np.arccos(Delta_hat_x)

	S_vector = a_vector_1*m1**2 + a_vector_2*m2**2
	S_norm = np.linalg.norm(S_vector, axis=0)
	noS = S_norm == 0
	Theta_S = np.zeros_like(a1)
	if np.any(noS):
		Theta_S[noS] = 0
	if np.any(~noS):
		S_hat_x = S_vector[0,~noS] / S_norm[~noS]
		Theta_S[~noS] = np.arccos(S_hat_x)

	Plus_vector = a_vector_2*m2 + a_vector_1*m1
	Plus_norm = np.linalg.norm(Plus_vector, axis=0)
	noPlus = Plus_norm == 0
	Theta_plus = np.zeros_like(a1)
	if np.any(noPlus):
		Theta_plus[noPlus] = 0
	if np.any(~noPlus):
		Plus_hat_x = Plus_vector[0,~noPlus] / Plus_norm[~noPlus]
		Theta_plus[~noPlus] = np.arccos(Plus_hat_x)

	#These thetas are fundamentally unknown until someone fits many more GR simulations.  The authors recommend randomizing.
	if zeroMysteryThetas:
		Theta1 = np.zeros(len(m1))
		Theta3 = np.zeros(len(m1))
		Theta5 = np.zeros(len(m1))
	else:
		Theta1 = np.random.random(len(m1))*2*np.pi
		Theta3 = np.random.random(len(m1))*2*np.pi
		Theta5 = np.random.random(len(m1))*2*np.pi

	###################
	##Recoil Velocity##
	###################

	#Recoil coming from asymmetric mass distribution.
	v_m = A*eta**2*(1-q)/(1+q) * (1+B*eta)

	#Recoil coming from spins.  Zero for Schwarzschild.
	v_perp = H*eta**2/(1+q) * (a2_par - q*a1_par)
	v_par = K*eta**2/(1+q) * (a2_perp - q*a1_perp) * np.cos(Theta_Delta)  #Their equation continues with a "KS", but I am omitting it for simplicity in favor of a fit with only "K".

	#Put everything together.
	kick_vector = np.vstack([v_m + v_perp*np.cos(xi), v_perp*np.sin(xi), v_par])
	kick_speed = np.linalg.norm(kick_vector, axis=0)

	################
	##Remnant Mass##
	################

	#Estimate of energy radiated at ISCO
	E_isco = 1.0 - np.sqrt(8)/3 + 0.103803*eta \
	+ 1.0/(36.0*np.sqrt(3)*(1+q)**2) * (q*(1.0+2.0*q)*a1_par + (2+q)*a2_par)
	- 5.0/(324.0*np.sqrt(2)*(1+q)**2)*(a2**2 - 3*a2_par**2 - 2*q*(a1_dot_a2-3*a1_par*a2_par) + q**2*(a1**2-3*a1_par**2))

	frac_radiated = eta*E_isco + E2*eta**2 + E3*eta**3 \
	+ eta**2/(1+q)**2 * (ES*(a2_par + q**2*a1_par) + EDelta*(1-q)*(a2_par - q*a1_par) + EA*np.linalg.norm(a_vector_2+q*a_vector_1, axis=0)**2 \
	+ EB*(a2_perp+q*a1_perp)**2 * (np.cos(Theta_plus) + EC)**2 + ED*np.linalg.norm(a_vector_2-q*a_vector_1, axis=0)**2 \
	+ EE*(a2_perp-q*a1_perp)**2 * (np.cos(Theta_Delta-Theta3)**2 + EF))

	m_remnant = (1.0 - frac_radiated) * (m1 + m2)

	################
	##Remnant Spin##
	################

	if spinMode == 'Lousto10':
		#These equations are not working, seemingly always returning spins > 1, and are not used in the literature.  I suspect an error.  Returning nans to remind myself there is an issue here.
		a_remnant_magnitude = np.full(len(a1), np.nan)
		"""
		a_remnant = np.zeros_like(a_vector_1)
		J_vector_ISCO = np.zeros_like(a_vector_1)

		J_vector_ISCO[2,:] += 2*np.sqrt(3) - 1.5255862*eta - 1.0/(9*np.sqrt(2)*(1+q)**2) * (q*(7+8*q)*a1_par + (8+7*q)*a2_par) \
		+ 2.0/(9*np.sqrt(3)*(1+q)**2)*(a2**2 - 3*(a2_par**2) - 2*q*(a1_dot_a2 - 3*a1_par*a2_par) + q**2*(a1**2-3*a1_par**2))
		J_vector_ISCO -= 1.0/(9*np.sqrt(2)*(1+q)**2)*(q*(1+4*q)*a_vector_1 + (4+q)*a_vector_2) + 1/(eta * (1+q**2))*(a_vector_2+q**2*a_vector_1)

		a_remnant += eta*J_vector_ISCO
		a_remnant[2,:] += J2*eta**2 + J3**eta**3
		a_remnant[2,:] += eta**2/(1+q)**2 * (JA*(a2_par+q**2*a1_par) + JB*(1-q)*(a2_par-q*a1_par))  #I could keep track of a sign flip here if I wanted.
		a_remnant[0,:] += (1-q) * np.linalg.norm(a_vector_2[:2,:]-q*a_vector_1[:2,:], axis=0) * np.sqrt(JDelta*np.cos(2*Theta_Delta) + JMDelta)  #Is JMDelta a typo?
		a_remnant[0,:] += np.linalg.norm(a_vector_2[:2,:]+q**2*a_vector_1[:2,:], axis=0)*np.sqrt(JS*np.cos(Theta_S-Theta5) + JMS)  #I'm throwing all the perpendicular components into x.  Unsure if this is intended.
		a_remnant *= (1.0 - frac_radiated)**(-2)
		a_remnant_magnitude = np.linalg.norm(a_remnant, axis=0)
		"""
	elif spinMode == 'Rezzolla08':
		#These equations assume that mass loss is negligible.

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
		(s5*eta + t0 + 2)/(1+q**2) * (a1*cosBeta + a2*q**2*cosGamma) + \
		2*np.sqrt(3) + t2*eta + t3*eta**2

		#Final equation
		a_remnant_magnitude = np.zeros_like(m1)

		#Sometimes I have a negative number under the radical.  For safety, I just set spin to the average...
		#I'm guessing that the fitting function happens to break with the parameters given.
		underRadical = a1**2 + a2**2*q**4 + 2*a2*a1*q**2*cosAlpha + \
		2*(a1*cosBeta + a2*q**2*cosGamma)*l*q + l**2*q**2
		a_remnant_magnitude[underRadical>=0] = 1.0 / (1+q[underRadical>0])**2 * np.sqrt(underRadical[underRadical>0])
		a_remnant_magnitude[underRadical<0] = (a1[underRadical<0]+a2[underRadical<0]/2)

	#There's no physical reason to have a maximum spin, but for numerical reasons, it may be necessary to set it equal to that from accretion. The fitting functions can also give unphysical spin.
	a_remnant_magnitude[a_remnant_magnitude > spinMax] = spinMax

	return m_remnant, a_remnant_magnitude, kick_speed
