"""
ARR: 07.05.17

Plots the fraction of mass gained in mergers as a function of BH mass
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import gzip
import pickle
currentPath = os.path.abspath(os.path.dirname(__file__)) + '/'

def plotMergedMassFraction(ensembles, labels=None, \
	colors=None, figsize=(5,4), output=None):

	n_ensembles = len(ensembles)
	fig, axarr = plt.subplots(n_ensembles, 1, figsize=figsize, sharex=True)

	if not hasattr(axarr, '__len__'):
		axarr = np.array([axarr])

	for e_index in range(n_ensembles):
		m_bh = []
		m_merged = []
		m_halo = []

		ensemble_folder = ensembles[e_index]
		ensemble_files = [file for file in os.listdir(ensemble_folder) if file[-5:]=='.pklz']

		for file in ensemble_files:
			with gzip.open(ensemble_folder+'/'+file, 'rb') as myfile:
				megaDict = pickle.load(myfile)
				z0 = megaDict['redshift'] == 0
				m_bh.extend(megaDict['m_bh'][z0])
				m_merged.extend(megaDict['m_merged'][z0])
				m_halo.extend(megaDict['m_halo'][z0])

		m_bh = np.array(m_bh)
		m_merged = np.array(m_merged)
		m_halo = np.array(m_halo)

		axarr[e_index].scatter(m_halo, m_merged/m_bh, s=20, alpha=0.6, color=colors[e_index])
		if labels is not None:
			axarr[e_index].text(5e13, 1.3e-2, labels[e_index], fontsize=11)

		axarr[e_index].set_xlim((3e11,2e15))
		axarr[e_index].set_ylim((1e-4,1e0))
		axarr[e_index].set_xscale('log')
		axarr[e_index].set_yscale('log')
		if e_index == n_ensembles-1:
			axarr[e_index].set_xlabel(r'$M_h \ [M_\odot]$', fontsize=15)
		axarr[e_index].set_ylabel(r'$M_{\bullet,\mathrm{BH \ Mergers}}/M_\bullet$', fontsize=15)

	fig.tight_layout()
	fig.subplots_adjust(wspace=0, hspace=0)

	if output is not None:
		fig.savefig(output)
	else:
		fig.show()

if __name__ == '__main__':
	ensembles = ['powerLaw_dcbh_beta5_060818']
	colors = ['r']
	labels = ['']

	plotMergedMassFraction(ensembles, colors=colors, labels=labels)
