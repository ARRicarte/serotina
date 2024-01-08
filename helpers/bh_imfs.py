"""
Functions to generate both pop III remnant and DCBH IMFs

ARR: 05.21.16
"""

import cPickle as pickle
import numpy as np
from scipy.interpolate import interp1d
import os
currentPath = os.path.abspath(os.path.dirname(__file__)) + '/'

########
##DCBH##
########

#Open pickle with IMF data, convert into a nice function. 
table = currentPath + '../lookup_tables/bh_imfs/imf_ln07.pkl'
with open(table, 'r') as myfile:
	data = pickle.load(myfile)

M_max_dcbh = data['M_BH'][-1]
M_min_dcbh = data['M_BH'][0]
y_dcbh = 10**(data['log(dn/dlog(M_BH))']-np.max(data['log(dn/dlog(M_BH))']))
p_log_m_bh = interp1d(np.log10(data['M_BH']), y_dcbh, bounds_error=False, fill_value=0)

#Define a normalized dn/dlogm
deltas = np.diff(np.log10(data['M_BH']))
trapy = 0.5*(y_dcbh[1:]+y_dcbh[:-1])
normalization = np.sum(deltas*trapy)
imf_dcbh = lambda logm: p_log_m_bh(logm) / normalization

def drawDCBHMass(n_vals=1):
	"""
	Draw from a pickle which has tabulated results from Lodato & Natarajan (2007)
	"""

	#Propose, and do rejection-comparison on each element
	log_M_prop = np.random.rand(n_vals) * (log_m_bh[-1] - log_m_bh[0]) + log_m_bh[0]
	probability = p_log_m_bh(log_M_prop)
	dice = np.random.rand(n_vals)
        rejections = np.where(dice > probability)[0]
	while len(rejections) > 0:
                newProps = np.random.random(len(rejections)) * (log_m_bh[-1] - log_m_bh[0]) + log_m_bh[0]
		newProbs = p_log_m_bh(newProps)
		dice = np.random.rand(len(rejections))
                log_M_prop[rejections] = newProps
                probability[rejections] = newProbs
                rejections = rejections[np.where(dice > newProbs)[0]]
        return 10**log_M_prop

###########
##Pop III##
###########

#Originally, I was going to use the Hirano+ 2014 Pop III IMFs, but their numbers for high-mass BHs
#are so poor (~10 of them) that there's no point.  I'm just drawing from a power law now, as in 
#Volonteri & Natarajan 2009

#Define a normalized dn/dlogm
#M_min_popIII=125
#M_max_popIII=1000
M_min_popIII=30
M_max_popIII=100
slope=-0.3
imf_popIII_nolimits = lambda logm: -slope*np.log(10) / (M_min_popIII**slope - M_max_popIII**slope) * 10**(slope*logm)

def imf_popIII(logm):
	"""
	Need to make a version that chops off at M_min_popIII and M_max_popIII
	"""
	if (logm < np.log10(M_max_popIII)) & (logm > np.log10(M_min_popIII)):
		return imf_popIII_nolimits(logm)
	else:
		return 0

def drawPopIIIMass(n_vals=1):
	"""
	Draw from a power-law distribution.
	"""

	#Propose, and do rejection-comparison on each element
	M_prop = np.random.rand(n_vals) * (M_max_popIII-M_min_popIII) + M_min_popIII
	probability = (M_prop/M_min_popIII)**(slope-1)
	dice = np.random.rand(n_vals)
	rejections = np.where(dice > probability)[0]
	while len(rejections) > 0:
		newProps = np.random.random(len(rejections)) * (M_max_popIII-M_min_popIII) + M_min_popIII
		newProbs = (newProps/M_min_popIII)**(slope-1)
		dice = np.random.rand(len(rejections))
		M_prop[rejections] = newProps
		probability[rejections] = newProbs
		rejections = rejections[np.where(dice > newProbs)[0]]
	return M_prop
