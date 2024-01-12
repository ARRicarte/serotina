"""
ARR:  03.13.17

Create 2D maps of M-Sigma for the whole ensemble
"""

import numpy as np
import os
import gzip
import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from ..helpers import sam_functions as sf
from .. import cosmology
from .. import constants
currentPath = os.path.abspath(os.path.dirname(__file__)) + '/'

class EnsembleMSigma(object):

	def __init__(self, ensemble_folder):
		"""
		Just save the names of the files
		"""

		self.ensemble_folder = ensemble_folder
		self.ensemble_files = [file for file in os.listdir(ensemble_folder) if file[-5:]=='.pklz']
		self.densityMap = None

	def assembleData(self, n_mass=20, n_sample=15, logMassRange=(11,15), redshiftsDesired=[4.75,3.2,1.8,1.0,0.6,0], z_host=0):
		"""
		Read in data, saving mass, velocity dispersion, redshift, and bolometric luminosity.
		"""
		
		masses = []
		luminosities = []
		weights = []
		sigmas = []
		redshifts = []
		usedRedshifts = []

		samplingFactor = float(n_mass) / (logMassRange[1] - logMassRange[0])

		oneComplete = False
		for f_index in range(len(self.ensemble_files)):
			file = self.ensemble_files[f_index]
			hostHaloMass = 10**float(file.split('_')[-1].split('m')[1].split('n')[0])
			nHalos = int(file.split('_')[-1].split('n')[1].split('.')[0])
			with gzip.open(self.ensemble_folder+'/'+file, 'rb') as myfile:
				megaDict = pickle.load(myfile)
			uniqueRedshifts = np.unique(megaDict['redshift'])
			
			nonZeroMask = (megaDict['m_bh'] != 0)
		
			for z_index in range(len(redshiftsDesired)):
				closestRedshift = uniqueRedshifts[np.argmin(np.abs(uniqueRedshifts - redshiftsDesired[z_index]))]
				if not oneComplete:
					usedRedshifts.append(closestRedshift)
				weight = sf.calcHaloNumberDensity(hostHaloMass, z_host) / nHalos / samplingFactor
				redshiftMask = (megaDict['redshift'] == closestRedshift)
				
				combinedMask = nonZeroMask & redshiftMask

				masses.extend(megaDict['m_bh'][combinedMask])
				luminosities.extend(megaDict['L_bol'][combinedMask])
				weights.extend([weight]*sum(combinedMask))
				sigmas.extend(sf.v2sigma(sf.v_peak(megaDict['m_halo'][combinedMask], megaDict['redshift'][combinedMask])))
				redshifts.extend([closestRedshift]*sum(combinedMask))
			oneComplete = True

		self.masses = np.array(masses)
		self.luminosities = np.array(luminosities)
		self.weights = np.array(weights)
		self.sigmas = np.array(sigmas)
		self.redshifts = np.array(redshifts)
		self.uniqueRedshifts = np.array(usedRedshifts)

	def makeMSigmaDensityMaps(self, logMRange=(2,11), logSigmaRange=(0,3), density=75, minimumLuminosity=0.0, maximumLuminosity=np.inf):
		"""
		Return density maps in the M-sigma plane for each redshift.
		"""

		logMEdges = np.linspace(logMRange[0], logMRange[1], density+1)
		logSigmaEdges = np.linspace(logSigmaRange[0], logSigmaRange[1], density+1)

		maps = []

		for z_index in range(len(self.uniqueRedshifts)):
			z_mask = (self.redshifts == self.uniqueRedshifts[z_index])
			l_mask = (minimumLuminosity <= self.luminosities) & (self.luminosities <= maximumLuminosity)
			combinedMask = z_mask & l_mask
			MSigmaMap = np.histogram2d(np.log10(self.masses[combinedMask]), np.log10(self.sigmas[combinedMask]), weights=self.weights[combinedMask], \
			bins=[logMEdges,logSigmaEdges]) / np.diff(logMEdges)[0] / np.diff(logSigmaEdges)[0]
			maps.append(MSigmaMap[0])
		self.densityMaps = [maps, logMEdges, logSigmaEdges]

	def plotMSigmaDensityMaps(self, plotShape=(2,3), colorMap='viridis', showColorBar=True, cbrange=None, output=None):
		if self.densityMaps is None:
			self.makeMSigmaDensityMaps()

		fig, axarr = plt.subplots(plotShape[0], plotShape[1], sharex=True, sharey=True, figsize=(7,5))
		if cbrange is None:
			squishedMaps = np.array(self.densityMaps[0])
			cbrange = [np.log10(np.min(squishedMaps[squishedMaps>0])), np.log10(np.max(squishedMaps))]
		for z_index in range(len(self.uniqueRedshifts)):
			i = int(z_index / 3)
			j = z_index % 3

			axarr[i,j].imshow(np.log10(self.densityMaps[0][z_index]), origin='lower', cmap=colorMap, \
			extent=[self.densityMaps[2][0], self.densityMaps[2][-1], self.densityMaps[1][0], self.densityMaps[1][-1]], \
			aspect='auto', vmax=cbrange[1], vmin=cbrange[0], interpolation='bilinear')

			axarr[i,j].text(0.2,10,r'$z={0:2.1f}$'.format(self.uniqueRedshifts[z_index]), fontsize=11)

			if i==1:
				axarr[i,j].set_xlabel(r'$\log(\sigma \ [\mathrm{km} \, \mathrm{s}^{-1}])$', fontsize=11)
			if j==0:
				axarr[i,j].set_ylabel(r'$\log(M_\bullet \ [M_\odot])$', fontsize=11)

		#Squish plots together, make space for a color bar
		fig.subplots_adjust(hspace=0, wspace=0, top=0.80)
		
		#Color bar time
		if showColorBar:
			cbar_ax = fig.add_axes([0.12, 0.87, 0.8, 0.05])
			sm = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap(colorMap), norm=plt.Normalize(vmin=cbrange[0], vmax=cbrange[1]))
			sm._A = []
			cb = fig.colorbar(sm, ticks=cbrange, cax=cbar_ax, orientation='horizontal')
			cb.set_label(r'$\log(\mathrm{dN/dVd\log{M_\bullet}d\log{\sigma}} \ [\mathrm{Mpc}^{-3}])$', fontsize=11, labelpad=5)
			cbar_ax.xaxis.set_label_position('top')

		if any(output):
			fig.savefig(output)
		else:
			fig.show()

if __name__ == '__main__':
	plotter = EnsembleMSigma('randombumps_popIII_042017')
	redshifts = [0.2, 2.0, 4.0, 6.0, 8.0, 10.0]
	plotter.assembleData(redshiftsDesired=redshifts)
	plotter.makeMSigmaDensityMaps(minimumLuminosity=0, maximumLuminosity=np.inf, density=75)
	plotter.plotMSigmaDensityMaps(colorMap='viridis', cbrange=(-5.0,0.7))
