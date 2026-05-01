"""
ARR:  07.18.2024

Plot spin versus black hole mass.
"""

import numpy as np
import os
import gzip
import pickle
import matplotlib.pyplot as plt
from ..helpers import sam_functions as sf
from .. import cosmology
from .. import constants
from .. import util
from ..cosmology import cosmology_functions as cf
from scipy import stats
import warnings
currentPath = os.path.abspath(os.path.dirname(__file__)) + '/'

def plotSpinMassRelation(ensembles, redshift=0, output=None, figshape=None, figsize=(4,4), labels=None, xlim=(1e5,5e10), ylim=(0,1), \
	includeSatellites=False, fig_axarr=None, thin_fEdd=(3e-2,1.0), onlyThin=False):
	
	if fig_axarr is None:
		if figshape is None:
			figshape = (1, len(ensembles))

		fig, axarr = plt.subplots(figshape[0], figshape[1], figsize=figsize, twinx=False, twiny=False)
		axarr = np.atleast_2d(axarr)
	else:
		fig, axarr = fig_axarr

	for e_index in range(len(ensembles)):
		spin = np.array([])
		mass = np.array([])
		eddingtonRatio = np.array([])
		i = int(e_index / axarr.shape[1])
		j = e_index % axarr.shape[1]

		ensemble_folder = ensembles[e_index]
		ensemble_files = [file for file in os.listdir(ensemble_folder) if file[-5:]=='.pklz']

		for file in ensemble_files:
			with gzip.open(ensemble_folder+'/'+file, 'rb') as myfile:
				megaDict = pickle.load(myfile)

			uniqueRedshifts = np.unique(megaDict['redshift'])
			closestRedshift = uniqueRedshifts[np.argmin(np.abs(uniqueRedshifts - redshift))]
			selection = megaDict['redshift'] == closestRedshift
			if not includeSatellites:
				selection = selection & (megaDict['satelliteToCentral'] == -1)
			if np.any(selection):
				spin = np.concatenate([spin, megaDict['spin_bh'][selection]])
				mass = np.concatenate([mass, megaDict['m_bh'][selection]])
				eddingtonRatio = np.concatenate([eddingtonRatio, megaDict['L_bol'][selection] / sf.eddingtonLum(megaDict['m_bh'][selection])])
			
		retrograde = spin < 0
		thin = (eddingtonRatio>=thin_fEdd[0]) & (eddingtonRatio <=thin_fEdd[1])
		if np.any(retrograde & thin):
			axarr[i,j].scatter(mass[retrograde & thin], np.abs(spin[retrograde & thin]), facecolor='blue', edgecolor='k', s=250, marker='*')
		if np.any(~retrograde & thin):
			axarr[i,j].scatter(mass[~retrograde & thin], np.abs(spin[~retrograde & thin]), facecolor='red', edgecolor='k', s=250, marker='*')
		if not onlyThin:
			if np.any(retrograde & ~thin):
				axarr[i,j].scatter(mass[retrograde & ~thin], np.abs(spin[retrograde & ~thin]), color='blue', s=20, alpha=0.3, zorder=-1)
			if np.any(~retrograde & ~thin):
				axarr[i,j].scatter(mass[~retrograde & ~thin], np.abs(spin[~retrograde & ~thin]), color='red', s=20, alpha=0.3, zorder=-1)

	for row in range(axarr.shape[0]):
		for col in range(axarr.shape[1]):
			ax = axarr[row,col]
			ax.set_xlim(xlim)
			ax.set_ylim(ylim)
			ax.set_xscale('log')
			if col == 0:
				ax.set_ylabel(r"$|a_\bullet|$", fontsize=13)
			else:
				ax.set_yticklabels([])
			if row == (axarr.shape[1]-1):
				ax.set_xlabel(r"$M_\bullet \ [M_\odot]$", fontsize=13)
			else:
				ax.set_xticklabels([])
	fig.tight_layout()

	if output is not None:
		fig.save(output)
		plt.close(fig)
	else:
		if fig_axarr is None:
			fig.show()
