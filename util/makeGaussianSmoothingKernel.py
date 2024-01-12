import numpy as np

def makeGaussianSmoothingKernel(widthInBins, maxSigma=4):
	"""
	A kernel that can be used for convolution.
	"""

	halfRangeInBins = int(np.floor(widthInBins * maxSigma))
	if halfRangeInBins < 1:
		return [1]
	else:
		gaussian = lambda x: np.exp(-0.5*(float(x)/widthInBins)**2) / widthInBins / np.sqrt(2*np.pi)
		binsSampled = np.linspace(-halfRangeInBins,halfRangeInBins,num=int(np.floor(2*halfRangeInBins)+1))
		return [gaussian(x) for x in binsSampled]
