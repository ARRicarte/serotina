import numpy as np
from .calcISCO import calcISCO

def calcSpinEvolutionFromAccretion(spin, m_ratio, spinMax=0.998):
	"""
	m_ratio is the amount that the black hole grew.  Using equations from Bardeen 1970.  This only accounts for thin disk accretion.  
	During this period, accretion proceeds in the same direction.  Initially retrograde accretion becomes prograde if m_ratio is large enough.
	"""

	spin = np.atleast_1d(spin).astype(float)
	m_ratio = np.atleast_1d(m_ratio).astype(float)
	alignment = np.atleast_1d(alignment).astype(int)

	r_ISCO = calcISCO(spin)
	newSpin = np.zeros_like(spin)
	spunToMaximum = m_ratio >= r_ISCO**0.5
	newSpin[spunToMaximum] = spinMax
	newSpin[~spunToMaximum] = np.maximum(np.minimum(np.sqrt(r_ISCO[~spunToMaximum])/3/m_ratio[~spunToMaximum] * (4 - (3*r_ISCO[~spunToMaximum]/m_ratio[~spunToMaximum]**2 - 2)**0.5), spinMax), -spinMax)

	return newSpin


