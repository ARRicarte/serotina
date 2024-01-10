import numpy as np
from ..helpers import sam_functions as sf
import matplotlib.pyplot as plt
import pickle
import os
import gzip
from ..helpers import smhm
currentPath = os.path.abspath(os.path.dirname(__file__)) + '/'

def calculateCoevolution(ensemble, logBinEdges, z, n_tree=15, n_bootstrap=1000, z_host=0, \
	percentileRange=[5,95], ratio=True, stellar=True, eddMin=0):
	print("Analyzing file {0}.".format(ensemble))
	
	#Weird lists can be indexed list[treeIndex][binIndex]
	sortedValues = [[[] for dummy in range(len(logBinEdges)-1)] for anotherDummy in range(n_tree)]
	haloMassWeights = [[[] for dummy in range(len(logBinEdges)-1)] for anotherDummy in range(n_tree)]
	individualFiles = os.listdir(ensemble)

	#Loop over all host halo masses.
	for f_index in range(len(individualFiles)):
		file = individualFiles[f_index]
		with gzip.open(ensemble+'/'+file, 'rb') as myfile:
			data = pickle.load(myfile)
		hostHaloMass = 10**float(file.split('_')[-1].split('m')[1].split('n')[0])
		print("   M_h = {0}".format(np.log10(hostHaloMass)))

		#Note:  Not worrying about things like sampling factor & n_halo in the weight; it gets normalized away.
		weight = sf.calcNumberDensity(hostHaloMass, z_host).tolist()

		#Extract important data.
		logMBH = np.log10(data['m_bh'])
		logMHalo = np.log10(data['m_halo'])
		redshifts = data['redshift']
		treeIndices = data['treeIndex']
		eddRatios = data['eddRatio']

		uniqueRedshifts = np.unique(redshifts)
		closestRedshift = uniqueRedshifts[np.argmin(np.abs(uniqueRedshifts - z))]
		redshiftMask = redshifts == closestRedshift

		#Fill sorted lists based on tree index and bin index.
		for i in range(n_tree):
			tree_mask = (treeIndices == i+1)
			for j in range(len(logBinEdges)-1):
				bh_mask = (logBinEdges[j] <= logMBH) & (logMBH < logBinEdges[j+1])
				eddMask = eddRatios >= eddMin
				combinedMask = bh_mask & tree_mask & redshiftMask & eddMask
				if any(combinedMask):
					logMatchingHaloMasses = logMHalo[combinedMask]
					if stellar:
						logHostMasses = smhm.logMstar(logMatchingHaloMasses, redshifts[combinedMask])
					else:
						logHostMasses = logMatchingHaloMasses
					if ratio:
						logMatchingBHMasses = logMBH[combinedMask]
						sortedValues[i][j].extend(10**(logMatchingBHMasses-logHostMasses))
					else:
						sortedValues[i][j].extend(logHostMasses)
					haloMassWeights[i][j].extend([weight]*len(logHostMasses))

	#Now that we have assembled organized lists of halo masses and weights, we'll bootstrap to get appropriate ranges.
	sortedValues = np.array(sortedValues)
	haloMassWeights = np.array(haloMassWeights)
	
	outputPieces = np.zeros((len(logBinEdges)-1, n_bootstrap))
	for boot in range(n_bootstrap):
		randomFileNumbers = np.random.randint(0, n_tree, n_tree)
		for i in range(len(logBinEdges)-1):
			combinedWeights = np.concatenate(haloMassWeights[randomFileNumbers,i])
			if any(combinedWeights > 0):
				outputPieces[i,boot] = np.average(np.concatenate(sortedValues[randomFileNumbers,i]), \
				weights=combinedWeights)

	output = np.percentile(outputPieces, percentileRange, axis=1)
	return output

def plotCoevolution(logBinEdges, valueRanges, labels=None, redshifts=None, colors=None, plotShape=(2,3), \
	xlabel=r'$\log_{10}M_\bullet \ [M_\odot]$', ylabel=r'$<\log_{10}M_h|\log_{10}M_\bullet> \ [M_\odot]$', \
	ratio=True, figSize=(8,5), output=None, stellar=True):
	n_ensemble = len(valueRanges)
	n_redshift = len(valueRanges[0])

	fig, axarr = plt.subplots(plotShape[0], plotShape[1], sharex=True, sharey=True, figsize=(8,5))

	xaxis = 0.5 * (logBinEdges[1:] + logBinEdges[:-1])
	if ratio:
		if stellar:
			ylim = (-4,1)
		else:
			ylim = (-6.0,-2.8)
	else:
		ylim = (9,15)

	#Add data
	for ensemble in range(n_ensemble):
		for z_index in range(n_redshift):
			i = int(z_index / plotShape[1])
			j = z_index % plotShape[1]
			if ratio:
				plottableLow = np.log10(valueRanges[ensemble][z_index][0,:])
				plottableHigh = np.log10(valueRanges[ensemble][z_index][1,:])
				axarr[i,j].text(4.0, ylim[1]-0.4, r'$z = {0}$'.format(redshifts[z_index]), fontsize=11) 
			else:
				plottableLow = valueRanges[ensemble][z_index][0,:]
				plottableHigh = valueRanges[ensemble][z_index][1,:]
				axarr[i,j].text(2.5, 14, r'$z = {0}$'.format(redshifts[z_index]), fontsize=11) 
			axarr[i,j].fill_between(xaxis, plottableLow, plottableHigh, \
			color=colors[ensemble], alpha=0.7, label=labels[ensemble])
	
	#Loop through once more for some formatting
	for i in range(plotShape[0]):
		for j in range(plotShape[1]):
			if i==plotShape[0]-1:
				axarr[i,j].set_xlabel(xlabel, fontsize=11)
			if j==0:
				axarr[i,j].set_ylabel(ylabel, fontsize=11)
			if (i==0) & (j==plotShape[1]-1):
				axarr[i,j].legend(frameon=False, loc='upper right', fontsize=8)
			axarr[i,j].set_xlim(logBinEdges[0],logBinEdges[-1])
			axarr[i,j].set_ylim(ylim)

	fig.subplots_adjust(wspace=0, hspace=0)
	if output is not None:
		fig.savefig(output)
	else:
		fig.show()

if __name__ == '__main__':
	#Stuff you probably don't need to change
	ensemblePath = './ensemble_output'
	n_tree = 15
	n_bootstrap = 1000
	percentileRange = [5,95]
	logBinEdges = np.linspace(3.5,10,13)
	ratio = True
	stellar = True
	ylabel = r'$<\log_{10}(M_\bullet/M_*)>$'

	#Stuff you care about
	ensembleList = ['powerlaw_dcbh_072717', 'powerlaw_popIII_073117']
	redshiftPanels = [6.0, 8.0, 10.0]#[0.2, 2.0, 4.0, 6.0, 8.0, 10.0]
	ensembleLabels = ['Heavy', 'Light']
	
	#I was dumb and decided to loop over z instead of doing it all at once.  Good god, this is slow.

	valueRanges = [[calculateCoevolution(ensemble, logBinEdges, z, ensemblePath=ensemblePath, n_tree=n_tree, n_bootstrap=n_bootstrap, \
	percentileRange=percentileRange, ratio=ratio, stellar=stellar) for z in redshiftPanels] for ensemble in ensembleList]
	plotCoevolution(logBinEdges, valueRanges, labels=ensembleLabels, colors=['b', 'g'], redshifts=redshiftPanels, ratio=ratio, \
	ylabel=ylabel, stellar=stellar)
