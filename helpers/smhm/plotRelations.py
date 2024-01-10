import matplotlib.pyplot as plt
from . import smhm_m13 as relation
#from . import smhm_b13 as relation
import numpy as np

def plotRelations(redshifts, logHaloMasses=np.linspace(8,15,100), ratio=False):
	for z in redshifts:
		logStellarMasses = relation.logMstar(logHaloMasses, z)
		if z > 8:
			lineStyle = ':'
		else:
			lineStyle = '-'
		if ratio:
			curve = logStellarMasses - logHaloMasses
		else:
			curve = logStellarMasses
		plt.plot(logHaloMasses, curve, lw=2, color=plt.cm.rainbow(float(redshifts.index(z))/(len(redshifts))), \
		ls=lineStyle, label="z={0}".format(z))
	plt.legend(loc='lower right')
	plt.xlabel("$\log_{10} (M_h/M_\odot)$", fontsize=16)
	if ratio:
		ylabel="$\log_{10}(M_*/M_h)$"
	else:
		ylabel="$\log_{10} (M_*/M_\odot)$"
	plt.ylabel(ylabel, fontsize=16)
	plt.xlim(logHaloMasses[0],logHaloMasses[-1])
	plt.show()

if __name__ == '__main__':
	redshifts = [0,1,2,3,4,5,6,7,8,15,20]
	plotRelations(redshifts, ratio=True)
