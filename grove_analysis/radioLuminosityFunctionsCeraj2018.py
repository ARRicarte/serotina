"""
Fitting function to radio luminosity function provided here.

This includes a high-luminosity cutoff, even though it's not supported by the data at all.

Ceraj et al. (2018) have both a pure density evolution (PDE) and pure luminosity evolution (PLE) fit.
I only include PLE, which seems to have less tension at the high-luminosity end than PDE in their figure 5.
"""

#These values were only for "High luminosity," based on L_x>1e43, found in the main text.  Not corrected for incompleteness.  Don't use.
"""
alpha1 = 3.97
beta1 = -0.92
phi_star = 10**(-7.47)
L_star = 10**26.47
alpha = -0.66
beta = -2.0 #This one is fixed.
"""

#These values were for the full sample described in Appendix A.
alpha1 = 1.97
beta1 = -0.46
phi_star = 10**(-5.1)
L_star = 10**24.59
alpha = -1.27
beta = -0.49

def phi0(L):
	return phi_star / ((L_star/L)**alpha + (L_star/L)**beta)

def phi(L, z):
	return phi0(L / (1.0+z)**(alpha1+z*beta1))
