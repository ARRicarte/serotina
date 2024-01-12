from scipy.signal import fftconvolve
def convolve(histogram, dexToConvolve, dexPerBin):
	binsToConvolve = dexToConvolve / dexPerBin
	kernel = makeGaussianSmoothingKernel(binsToConvolve)
	return fftconvolve(histogram, kernel, mode='same')

