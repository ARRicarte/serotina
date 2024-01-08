from .calcISCO import *

def calcSpinEvolutionFromAccretion(spinNow, m_ratio, mu=1, spinMax=0.998):
	"""
	Assuming that m_ratio is the amount that the black hole grew.
	"""

	#Calculate the ISCO radius, depending on whether we're prograde or retrograde
	r_ISCO = calcISCO(spinNow, mu=mu)

	#Update spin, limited to the Thorne limit.
	if m_ratio >= r_ISCO**0.5:
		newSpin = spinMax
	else:
		newSpin = r_ISCO**0.5 / 3 / m_ratio * (4 - (3 * r_ISCO / m_ratio**2 - 2)**0.5)
	return max(min(abs(newSpin), spinMax), 0)


