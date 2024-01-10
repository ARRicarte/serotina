import numpy as np
import os
from .. import sam_functions as sf
import matplotlib.pyplot as plt
import pickle
import gzip
from scipy.optimize import curve_fit
from ..helpers import smhm
currentPath = os.path.abspath(os.path.dirname(__file__)) + '/'

def line(data, alpha, beta):
	return alpha + beta*data

def sigmaToM(sigmaList, alphaBeta=(8.32,5.35)):
	return 10**(alphaBeta[0] + alphaBeta[1]*np.log10(sigmaList/200))

def calcMSigmaz0(ensembles, alphaBeta=[(8.32,5.35)], plotData=True, dwarfsToo=True, labels=None, \
	colors=None, figsize=(6,6), output=None, iso=None, includeSatellites=False, minSigmaFit=200, fitALine=False, xlim=(20,500), ylim=(1e3,1e11), \
	outFiles=None):

	n_ensembles = len(ensembles)
	fig, axarr = plt.subplots(np.min((n_ensembles,2)), int(np.ceil(float(n_ensembles)/2)), figsize=figsize, sharex=True, sharey=False)
	if iso is None:
		iso = [False]*len(ensembles)

	axarr = np.atleast_2d(axarr)

	for e_index in range(n_ensembles):
		i = int(e_index / axarr.shape[1])
		j = e_index % axarr.shape[1]
		m_bh = np.array([])
		sigma = np.array([])

		ensemble_folder = ensembles[e_index]
		ensemble_files = [file for file in os.listdir(ensemble_folder) if file[-5:]=='.pklz']

		for file in ensemble_files:
			with gzip.open(ensemble_folder+'/'+file, 'rb') as myfile:
				megaDict = pickle.load(myfile)
				selection = megaDict['redshift'] == 0
				if not includeSatellites:
					selection = selection & (megaDict['satelliteToCentral'] == -1)
				m_bh = np.concatenate((m_bh, megaDict['m_bh'][selection]))
				if 'sigma' in megaDict.keys():
					sigma = np.concatenate((sigma, megaDict['sigma'][selection]))
				elif iso[e_index]:
					sigma = np.concatenate((sigma, sf.v2sigma_SIS(sf.M2v(megaDict['m_halo'][selection], 0))))
				else:
					sigma = np.concatenate((sigma, sf.velocityDispersion(megaDict['m_halo'][selection], 0)))

		axarr[i,j].scatter(sigma, m_bh, s=20, alpha=0.7, color=colors[e_index])
		if alphaBeta is not None:
			extremeSigmas = np.array([1,1e10])
			oplotMasses = [sigmaToM(s, alphaBeta=alphaBeta[e_index]) for s in extremeSigmas]
			axarr[i,j].plot(extremeSigmas, oplotMasses, linewidth=2, linestyle=':', color='k', zorder=2, label='Burst Limit')

		if plotData:
			with open(currentPath + '../lookup_tables/bh_data/Saglia16.pkl', 'rb') as myfile:
				data = pickle.load(myfile, encoding='latin1')
			data_sigma = np.array([10**logs[0] for logs in data['logsigma']])
			data_M = np.array([10**logm[0] for logm in data['logM_BH']])
			data_sigma_err = np.transpose(np.array([[10**logs[0]*(1.0-10**(-logs[1])),10**logs[0]*(10**logs[1]-1)] for logs in data['logsigma']]))
			data_M_err = np.transpose(np.array([[10**logm[0]*(1.0-10**(-logm[1])),10**logm[0]*(10**logm[1]-1)] for logm in data['logM_BH']]))
			axarr[i,j].errorbar(data_sigma, data_M, xerr=data_sigma_err, yerr=data_M_err, fmt='none', ecolor='k', alpha=1.0, zorder=1, label='Saglia+16')

			if dwarfsToo:
				with open(currentPath + '../lookup_tables/bh_data/msigma_dwarfs.dat', 'r') as myfile:
					small_data = np.loadtxt(myfile, dtype=str, delimiter=',')[:,1:].astype(float)
				axarr[i,j].errorbar(small_data[:,3], small_data[:,0], yerr=[small_data[:,1],small_data[:,2]], xerr=[small_data[:,4],small_data[:,5]], \
				fmt='o', color='slategrey', ecolor='slategrey', label='Other')

		#Let's fit a line, but only past minSigmaFit km/s
		if fitALine:
			alphaBetaFit, covarianceMatrix = curve_fit(line, np.log10(sigma[(m_bh>0) & (sigma>minSigmaFit)]/200.0), np.log10(m_bh[(m_bh>0) & (sigma>minSigmaFit)]), p0=(8,4))
			print("A linear regression yields alpha = {0} +/- {1}, beta = {2} +/- {3}".format(alphaBetaFit[0], covarianceMatrix[0][0]**0.5, alphaBetaFit[1], covarianceMatrix[1][1]**0.5))
			
			#Then, let's try and estimate the intrinsic scatter already present in this relation.
			logPerfectMasses = alphaBetaFit[0] + alphaBetaFit[1] * np.log10(sigma[(m_bh>0) & (sigma>minSigmaFit)]/200.0)
			scatter = np.std(np.log10(m_bh[(m_bh>0) & (sigma>minSigmaFit)]) - logPerfectMasses)
			print("Intrinsic scatter is estimated at {0} dex.".format(scatter))

			offsets = np.array([m_bh[k] / sigmaToM(sigma[k], alphaBeta=alphaBeta[e_index]) for k in range(len(m_bh)) if sigma[k]>minSigmaFit])
			print("On average, the ratio between output M and expected M is {0}.".format(10**np.average(np.log10(offsets[offsets != 0]))))
			print("Alternatively, the median ratio is {0}.".format(10**np.median(np.log10(offsets[offsets != 0]))))

		axarr[i,j].text(25, 2e10, labels[e_index], fontsize=12)
		axarr[i,j].set_xlim((xlim[0],xlim[1]))
		axarr[i,j].set_ylim((ylim[0],ylim[1]))
		axarr[i,j].set_xscale('log')
		axarr[i,j].set_yscale('log')
		if i == axarr.shape[0]-1:
			axarr[i,j].set_xlabel(r'$\sigma \ [\mathrm{km} \, \mathrm{s}^{-1}]$', fontsize=14)
		if (j == 0) | (axarr.shape[0]==1):
			axarr[i,j].set_ylabel(r'$M_\bullet \ [M_\odot]$', fontsize=14)
		if (i == axarr.shape[0]-1) & (j==axarr.shape[1]-1):
			axarr[i,j].legend(frameon=False, loc='lower right', fontsize=10)

		if outFiles is not None:
			outarray = np.transpose(np.vstack((sigma, m_bh)))
			header = 'sigma [km s^-1], m_bh [M_sun]'
			with open(outFiles[e_index], 'w') as myfile:
				np.savetxt(myfile, outarray, header=header, delimiter=',')

	fig.tight_layout()
	fig.subplots_adjust(wspace=0, hspace=0)
	for i in range(axarr.shape[0]):
		for j in range(axarr.shape[1]):
			if (j > 0) & (axarr.shape[0] > 1):
				axarr[i,j].set_yticks([])
			elif i != axarr.shape[0]-1:
				currentyTicks = axarr[i,j].get_yticks()
				newyTicks = [tick for tick in currentyTicks if ((tick > ylim[0]) & (tick <= ylim[1]))]
				axarr[i,j].set_yticks(newyTicks)

	if output is not None:
		fig.savefig(output)
	else:
		fig.show()

if __name__== '__main__':
	'''
	alphaBeta = [(8.45,5.0), (8.45,5.0), (8.45,5.0), (8.45,5.0)]
	colors = ['r', 'orange', 'purple', 'c']
	ensembles = ['powerLaw_popIII_pmerge0.1_072018', 'agnms_popIII_pmerge0.1_072018', 'powerLaw_dcbh_pmerge0.1_072018', 'agnms_dcbh_pmerge0.1_072018']
        labels = ['Light-PL', 'Light-MS', 'Heavy-PL', 'Heavy-MS']
	iso = [False,False,False,False]
	'''
	alphaBeta = [(8.45,5.0), (8.45,5.0)]
	colors = ['firebrick', 'royalblue']
	ensembles = ['blq_dcbh_pmerge0.1_072018', 'blq_popIII_pmerge0.1_072018']
	labels = ['Heavy', 'Light']
	iso = [False, False]

	#alphaBeta = [(8.45,5.0)]*2
	'''
	alphaBeta = [(8.32,5.35)]*3
        ensembles = ['powerLaw_dcbh_060518', 'powerLaw_dcbh_fmax0.3_060518', 'powerLaw_dcbh_beta4_060618']
        colors = ['b', 'g', 'r']
        labels = ['Heavy-1', 'Heavy-0.3', 'Heavy-beta4']
	iso = [False]*3
	'''

	'''
	alphaBeta = [(8.5,4.0)]*2
	ensembles = ['powerLaw_dcbh_beta4_spin_060818/', 'powerLaw_dcbh_beta4_060818']
	colors = ['b', 'g']
	labels = ['Heavy-spin', 'Heavy']
	iso = [False, False]
	'''

	#calcMSigmaz0(ensembles, colors=colors, labels=labels, alphaBeta=alphaBeta, iso=iso, minSigmaFit=220, fitALine=True)
	#calcMSigmaz0(ensembles, colors=colors, labels=labels, alphaBeta=None, iso=iso, minSigmaFit=220, fitALine=False, dwarfsToo=False)
	#outFiles = ['./data_products/MSigma_'+label+'.txt' for label in labels]
	#calcMSigmaz0(ensembles, colors=colors, labels=labels, alphaBeta=None, iso=iso, minSigmaFit=220, fitALine=False, outFiles=outFiles)
	calcMSigmaz0(ensembles, colors=colors, labels=labels, alphaBeta=None, iso=iso, fitALine=False, figsize=(5,8))
