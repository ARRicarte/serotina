"""
Shen et al. (2020) have a global fit to the bolometric luminosity function out to z=7.
We implement their formulae here.
"""

import numpy as np

#There are two sets of parameters, because the behavior of the faint end is sensitive to how you extrapolate based on flux-limited data.
#In global fit A, the faint end slope has a flexible evolutionary pattern.
#In global fit B, the faint end slope is restricted to evolve monotonically as a power law.
#Give L in units of solar luminosities, not erg/s.  Returned units are number per dex per cMpc^-3

parameters = {}
parameters['A'] = {}
parameters['A']['a0'] = 0.8569
parameters['A']['a1'] = -0.2614
parameters['A']['a2'] = 0.0200
parameters['A']['b0'] = 2.5375
parameters['A']['b1'] = -1.0425
parameters['A']['b2'] = 1.1201
parameters['A']['c0'] = 13.0088
parameters['A']['c1'] = -0.5759
parameters['A']['c2'] = 0.4554
parameters['A']['d0'] = -3.5426
parameters['A']['d1'] = -0.3936

parameters['B'] = {}
parameters['B']['a0'] = 0.3653
parameters['B']['a1'] = -0.6006
parameters['B']['b0'] = 2.4709
parameters['B']['b1'] = -0.9963
parameters['B']['b2'] = 1.0716
parameters['B']['c0'] = 12.9656
parameters['B']['c1'] = -0.5758
parameters['B']['c2'] = 0.4698
parameters['B']['d0'] = -3.6276
parameters['B']['d1'] = -0.3444

#Reference redshift.
z_ref = 2.0

#Chebyshev polynomial functions.
T0 = lambda x: 1.0
T1 = lambda x: x
T2 = lambda x: 2.0*x**2-1.0

def gamma1(z, fit):
	
	if fit == 'A':
		return parameters[fit]['a0']*T0(1.0+z) + parameters[fit]['a1']*T1(1.0+z) + parameters[fit]['a2']*T2(1.0+z)
	elif fit == 'B':
		return parameters[fit]['a0']*((1.0+z)/(1.0+z_ref))**parameters[fit]['a1']

def gamma2(z, fit):

	z_quantity = (1.0+z)/(1.0+z_ref)
	return 2 * parameters[fit]['b0'] / (z_quantity**parameters[fit]['b1'] + z_quantity**parameters[fit]['b2'])

def L_star(z, fit):

	z_quantity = (1.0+z)/(1.0+z_ref)
	return 10**(2 * parameters[fit]['c0'] / (z_quantity**parameters[fit]['c1'] + z_quantity**parameters[fit]['c2']))

def phi_star(z, fit):

	return 10.0**(parameters[fit]['d0']*T0(1.0+z) + parameters[fit]['d1']*T1(1.0+z))

def dn_dlogL(L, z, fit):

	L_star_z = L_star(z, fit)
	return phi_star(z, fit) / ((L/L_star_z)**(gamma1(z, fit)) + (L/L_star_z)**(gamma2(z, fit)))
