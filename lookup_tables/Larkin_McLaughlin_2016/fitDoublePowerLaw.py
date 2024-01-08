from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

def brokenPowerLaw(x, normalization, slope1, slope2, x_break):
	output = np.zeros(len(x))
	output[x < x_break] = normalization * (x/x_break)**slope1
	output[x >= x_break] = normalization * (x/x_break)**slope2
	return output

def brokenPowerLaw_logfitting(logx, normalization, slope1, slope2, logx_break):
	output = np.zeros(len(logx))
	output[logx < logx_break] = np.log10(normalization * (10**logx[logx < logx_break]/10**logx_break)**slope1)
	output[logx >= logx_break] = np.log10(normalization * (10**logx[logx >= logx_break]/10**logx_break)**slope2)
	return output

def powerLaw(x, normalization, slope):
	return normalization * x**slope

def powerLaw_logfitting(logx, normalization, slope):
	return np.log10(normalization * (10**logx)**slope)

def fitDoublePowerLaw(xdata, ydata, plot=True, p0=None):
	if p0 is None:
		p0 = [np.nanmean(np.log10(ydata)), 0.0, 0.0, np.nanmean(np.log10(xdata))]
	idealParameters = curve_fit(brokenPowerLaw_logfitting, np.log10(xdata), np.log10(ydata), p0=p0)[0]
	print "Your power law has the following parameters:"
	print "  Normalization:", idealParameters[0]
	print "  slope1:", idealParameters[1]
	print "  slope2:", idealParameters[2]
	print "  x_break:", idealParameters[3]
	
	if plot:
		plt.loglog(xdata, ydata, linewidth=2, linestyle='-', color='b', label="Data")
		plt.xlabel('x', fontsize=18)
		plt.ylabel('y', fontsize=18)
		plt.loglog(xdata, 10**brokenPowerLaw_logfitting(np.log10(xdata), idealParameters[0], idealParameters[1], idealParameters[2], \
		idealParameters[3]), linewidth=2, color='r', linestyle='--', label="Fit")
		plt.show()

	return idealParameters

def fitPowerLaw(xdata, ydata, plot=True, p0=None):
	if p0 is None:
		p0 = [np.nanmean(ydata), 0]
	idealParameters = curve_fit(powerLaw_logfitting, np.log10(xdata), np.log10(ydata), p0=p0)[0]
	print "Your power law has the following parameters:"
        print "  Normalization:", idealParameters[0]
        print "  Slope:", idealParameters[1]

	if plot:
		plt.loglog(xdata, ydata, linewidth=2, linestyle='-', color='b', label="Data")
                plt.xlabel('x', fontsize=18)
                plt.ylabel('y', fontsize=18)
		plt.show()

if __name__ == '__main__':
	with open("./vpeak_sigma_LL16_fig5.1_blue.dat", 'r') as myfile:
		data = np.loadtxt(myfile)
	fitDoublePowerLaw(data[:,1], data[:,0], p0=(500.0, -1.0, -4.0, 2.0), plot=True)
	"""
	lowerEnd = data[:,0] < 100
	upperEnd = data[:,0] > 200
	fitPowerLaw(data[lowerEnd,0], data[lowerEnd,1])
	fitPowerLaw(data[upperEnd,0], data[upperEnd,1])
	"""
