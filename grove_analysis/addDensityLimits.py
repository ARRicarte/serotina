"""
ARR:  05.24.17

Called by plotBlackHoleDensity.py
"""

import matplotlib.pyplot as plt
import numpy as np
import os
currentPath = os.path.abspath(os.path.dirname(__file__)) + '/'

def addDensityLimits(ax, labelOldConstraints=False):

	path = currentPath + "../lookup_tables/bh_data/blackHoleDensityLimits/"

	if labelOldConstraints:
		oldLabels = ["Hopkins et al. (2007)", "Salvaterra et al. (2012)", "Shankar et al. (2009)", \
		"Treister et al. (2009)", 'Treister et al. (2013)']
	else:
		oldLabels = [None]*5

	#Plot Hopkins 2007
	with open(path+"hopkins2007.dat", 'r') as myfile:
		data = np.loadtxt(myfile)
		yerr = [data[:,1]-data[:,2], data[:,3]-data[:,1]]
		er_hopkins = ax.errorbar(data[:,0], data[:,1], yerr=yerr, ls='None', lw=2, \
		fmt='o', color='k', markeredgecolor='k', label=oldLabels[0])

	#Plot Salvaterra 2012
	with open(path+"salvaterra2012.dat", 'r') as myfile:
		data = np.loadtxt(myfile)
		er_salvaterra = ax.errorbar(data[0]+2, np.log10(data[1]), xerr=2, uplims=True, yerr=0.3, \
		lw=2, color='k', label=oldLabels[1])
		er_salvaterra[-1][0].set_linestyle('--')

	#Plot Shankar 2009
	with open(path+"shankar2009.dat", 'r') as myfile:
		data = np.loadtxt(myfile)
		ax.fill_between([data[0], data[0]+0.5], [np.log10(data[1])]*2, [np.log10(data[2])]*2, \
		alpha=0.5, color='k', label=oldLabels[2])

	#Plot Treister et al. (2009)
	with open(path+"Treister2009.dat", "r") as myfile:
		data = np.loadtxt(myfile)
		xerr = [data[:,0]-data[:,2],data[:,3]-data[:,0]]
		er_treister1 = ax.errorbar(data[:,0], data[:,1], xerr=xerr, lw=2, color='k', \
		fmt='o', ls='None', markeredgecolor='k', label=oldLabels[3])
		er_treister1[-1][0].set_linestyle(':')

	#Plot Treister et al. (2013)
	with open(path+"Treister2013.dat", 'r') as myfile:
		data = np.loadtxt(myfile)
		er_treister2 = ax.errorbar(data[:,0]+0.5, np.log10(data[:,1]), xerr=0.5, yerr=0.3, ls='None', lw=2, \
		uplims=True, color='k', label=oldLabels[4])
		er_treister2[-1][0].set_linestyle('-.')

	#Plot our data
	with open(path+"cappelluti2016.dat", 'r') as myfile:
		data = np.loadtxt(myfile, dtype=str)
		densities = data[:,2].astype(float)
		#Aligned these by hand
		spectraColors = ['k']*4
		spectraNames = ['Standard, Low Z', 'Slim, Low Z', 'Standard, High Z', 'Slim, High Z']
		for i in range(len(densities)):
			ax.errorbar([7.5], np.log10(densities[i]), xerr=1.5, uplims=True, yerr=0.3, \
			lw=2, color=spectraColors[i], ls='None')
