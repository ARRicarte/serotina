"""
ARR: 11.12.15

Analyzes SAM output
"""

import matplotlib.pyplot as plt
import numpy as np
from ..helpers import sam_functions as sf
from scipy.integrate import simps
from scipy.interpolate import interp1d
from .. import constants
import cPickle as pickle
import gzip
import os
import pdb
from ..util import folderToMovie
currentPath = os.path.abspath(os.path.dirname(__file__)) + '/'

#TODO:  Turn into a bunch of static methods so that __init__ doesn't crash 
#every time I change the keys I save.

class SAM_Analysis(object):

	def __init__(self, pickledSAMOutput):
		"""
		Read pickled output.
		"""
		with gzip.open(pickledSAMOutput, 'rb') as myfile:
			data = pickle.load(myfile)
		self.m_bh = data['m_bh_snap']
		self.spin_bh = data['spin_bh_snap']
		self.eddRatio = data['eddRatio_snap']
		self.m_halo = data['m_halo_snap']
		self.time = data['time']
		self.redshift = data['redshift']
		self.indices = data['indices_snap']
		self.elliptical = data['elliptical_snap']
		self.L_bol = data['L_bol_snap']
		self.bh_id = data['bh_id_snap']
		self.satelliteToCentral = data['satelliteToCentral_snap']
		self.sigma = data['sigma_snap']

	def traceBH(self, bh_id):
		"""
		Find the index of a bh_id at each level
		"""
		trace = np.zeros(len(self.time), dtype=int)
		exists = np.zeros(len(self.time), dtype=bool)
		nlev = len(self.time)
		for level in range(nlev):
			exists_bh = self.bh_id[level].count(bh_id)
			if exists_bh == 1:
				nextIndex = self.bh_id[level].index(bh_id)
			else:
				nextIndex = -1
			trace[level] = nextIndex
			exists[level] = exists_bh
		return trace, exists

	def plotGrowthTracks(self, showLegend=False, depth=0, alphaBeta=(8.22,4.58)):
		"""
		Plot tracks on the M-sigma relation
		"""

		fig, ax1 = plt.subplots()

		#Indices of the main black hole
		mainBH_indices, mainBH_exists = self.traceBH(self.bh_id[-1][0])

		#Plot Main Progenitor
		masses = np.array([self.m_bh[i][mainBH_indices[i]] for i in range(len(self.time))])
		sigmas = np.array([sf.v2sigma(self.v_c[i][mainBH_indices[i]]) for i in range(len(self.time))])
		ax1.loglog(sigmas[mainBH_exists], masses[mainBH_exists], linewidth=2, linestyle='-', color='k')
		ax1.set_xlabel(r'$\sigma \, [\mathrm{km/s}]$', fontsize=18)
		ax1.set_ylabel(r'$M_\mathrm{BH} \, (M_\odot)$', fontsize=18)
		ax1.set_xlim(3,1e3)
		ax1.set_ylim(1e2,1e10)

		#Plot M-sigma relation
		extremeVs = np.array([1,1e10])
		extremeSigmas = sf.v2sigma(extremeVs)
		oplotMasses = sf.v2M_BH(extremeVs)
		ax1.loglog(extremeSigmas, oplotMasses, linewidth=1, color='g')

		#Add Satellites
		if depth > 0:
			moreBHs = [i for i in self.bh_id[-1-depth] if i != self.bh_id[-1][0]]
			for bh in moreBHs:
				otherBH_indices, otherBH_exists = self.traceBH(bh)
				masses = np.array([self.m_bh[i][otherBH_indices[i]] for i in range(len(self.time))])
				sigmas = np.array([sf.v2sigma(self.v_c[i][mainBH_indices[i]]) for i in range(len(self.time))])
				ax1.loglog(sigmas[otherBH_exists], masses[otherBH_exists], linewidth=1, linestyle='--', color='k')
		if showLegend:
			plt.legend(loc=3, frameon=False)
		plt.show()

	def plotGrowthHistory(self, plotTime=False, showLegend=False, depth=0, alphaBeta=(8.22,4.58), plotSpin=True):
		"""
		Plot the evolution of a SAM's parts as a function of redshift.
		"""

		fig, ax1 = plt.subplots()
		if plotTime:
			xaxis = self.time
			xlabel = r'$t_\mathrm{ABB} \, (\mathrm{Gyr})$'
		else:
			xaxis = self.redshift
			xlabel = r'$z$'

		#Indices of the main black hole
		mainBH_indices, mainBH_exists = self.traceBH(self.bh_id[-1][0])

		#Plot Main Progenitor Mass
		maxis = np.array([self.m_bh[i][mainBH_indices[i]] for i in range(len(self.time))])
		ax1.semilogy(xaxis[mainBH_exists], maxis[mainBH_exists], linewidth=2, label="Mass", linestyle='-', color='k')
		maxis_expect = sf.v2M_BH(np.array([self.v_c[i][mainBH_indices[i]] for i in range(len(self.time))]), alphaBeta=alphaBeta)
		ax1.semilogy(xaxis[mainBH_exists], maxis_expect[mainBH_exists], linewidth=2, label="Expectation", linestyle='-', color='slategrey')
		ax1.set_xlabel(xlabel, fontsize=18, color='k')
		ax1.set_ylabel(r'$M_\mathrm{BH} \, (M_\odot)$', fontsize=18)
		ax1.set_ylim(1e2,1e10)

		#Plot Main Progenitor Spin
		if plotSpin:
			ax2 = ax1.twinx()
			aaxis = np.array([self.spin_bh[i][mainBH_indices[i]] for i in range(len(self.time))])
			ax2.plot(xaxis[mainBH_exists], aaxis[mainBH_exists], linewidth=2, label="Spin Parameter", linestyle='-', color='r')
			ax2.set_ylabel(r'$a$', fontsize=18, color='r')
			for tl in ax2.get_yticklabels():
				tl.set_color('r')
			ax2.set_ylim(0,1)

		#Add some satellites.  depth is the number of levels in the tree you go back to acquire satellites.
		if depth > 0:
			moreBHs = [i for i in self.bh_id[-1-depth] if i != self.bh_id[-1][0]]
			for bh in moreBHs:
				otherBH_indices, otherBH_exists = self.traceBH(bh)
				maxis = np.array([self.m_bh[i][otherBH_indices[i]] for i in range(len(self.time))])
				ax1.semilogy(xaxis[otherBH_exists], maxis[otherBH_exists], linewidth=1, linestyle='--', color='k')
				maxis_expect = sf.v2M_BH(np.array([self.v_c[i][otherBH_indices[i]] for i in range(len(self.time))]), alphaBeta=alphaBeta)
				ax1.semilogy(xaxis[otherBH_exists], maxis_expect[otherBH_exists], linewidth=1, linestyle='--', color='slategrey')
				if plotSpin:
					aaxis = np.array([self.spin_bh[i][otherBH_indices[i]] for i in range(len(self.time))])
					ax2.plot(xaxis[otherBH_exists], aaxis[otherBH_exists], linewidth=1, linestyle='--', color='r')

		plt.gca().invert_xaxis()
		if showLegend:
			plt.legend(loc=3, frameon=False)
		plt.show()

	def plotEvolution(self, mp4name, colorMode='eddingtonRatio', shapeMode='satellites', plotData=False, alphaBeta=(8.22,4.58), \
		showBar=True, alphaScatter=0.7, relationLabel='Feedback Limit', framerate=10, xlim=(3,5e2), ylim=(1e2,1e11), textloc=(4.0e0,1e10), \
		showSupernovaBarrier=False, temporaryFrameFolder='./frames/', dpi=300):
		"""
		Plot the evolution of M-sigma
		"""

		if temporaryFrameFolder[-1] != '/':
			temporaryFrameFolder += '/'

		if not os.path.isdir(temporaryFrameFolder):
			print("Creating directory "+temporaryFrameFolder+" to store movie frames.")
			os.system("mkdir "+temporaryFrameFolder)
		else:
			print("Deleting all frames in "+temporaryFrameFolder)
			os.system("rm "+temporaryFrameFolder+"frame*png")

		fig, ax = plt.subplots(1, figsize=(7,6))
		plt.clf()
		plt.xlabel(r'$\log(\sigma \, [\mathrm{km \, s^{-1}}])$', fontsize=16)
		plt.ylabel(r'$\log(M_\mathrm{BH} \, [\mathrm{M}_\odot])$', fontsize=16)
		plt.xscale('log')
		plt.yscale('log')
		plt.xlim(xlim)
		plt.ylim(ylim)

		if alphaBeta is not None:
			#Overplot observed relations
			extremeSigmas = np.array([1,1e10])
			oplotMasses = 10**(alphaBeta[0] + alphaBeta[1]*(np.log10(extremeSigmas/200)))
			plt.plot(extremeSigmas, oplotMasses, linewidth=2, linestyle=':', label=relationLabel, color='k', zorder=2)
			plt.legend(loc='lower right', frameon=False)

		if plotData:
			with open(currentPath + '../lookup_tables/bh_data/Saglia16.pkl', 'r') as myfile:
				data = pickle.load(myfile)
			data_sigma = np.array([10**logs[0] for logs in data['logsigma']])
			data_M = np.array([10**logm[0] for logm in data['logM_BH']])
			data_sigma_err = np.transpose(np.array([[10**logs[0]*(1.0-10**(-logs[1])),10**logs[0]*(10**logs[1]-1)] for logs in data['logsigma']]))
			data_M_err = np.transpose(np.array([[10**logm[0]*(1.0-10**(-logm[1])),10**logm[0]*(10**logm[1]-1)] for logm in data['logM_BH']]))
			plt.errorbar(data_sigma, data_M, xerr=data_sigma_err, yerr=data_M_err, fmt='none', ecolor='k', alpha=0.4, zorder=1)

		#Set up
		starPoints = None
		circlePoints = None
		supernovaBarrier = None
		ztext = plt.text(textloc[0], textloc[1], r'$z ='+('%03.2f'%self.redshift[-1])+'$', fontsize=16)

		#Color bar with code that I don't understand so that I can make it before the plot has any values.
		if colorMode == 'eddingtonRatio':
			ratioRange = (-3,0)
			if showBar:
				sm = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap('jet_r'), norm=plt.Normalize(vmin=ratioRange[0], vmax=ratioRange[1]))
				sm._A = []
				cb = plt.colorbar(sm, ticks=ratioRange)
				cb.set_label(r'$\log(f_\mathrm{Edd})$', fontsize=14)
		elif colorMode == 'luminosity':
			lrange = (7,15)
			if showBar:
				sm = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap('jet_r'), norm=plt.Normalize(vmin=lrange[0], vmax=lrange[1]))
				sm._A = []
				cb = plt.colorbar(sm, ticks=lrange)
				cb.set_label(r'$\log(L_\mathrm{Bol}/L_\odot)$', fontsize=14)
		elif colorMode == 'spin':
			arange = (0,1)
			if showBar:
				sm = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap('jet_r'), norm=plt.Normalize(vmin=arange[0], vmax=arange[1]))
				sm._A = []
				cb = plt.colorbar(sm, ticks=arange)
				cb.set_label(r'$a$', fontsize=14)

		plt.tight_layout()
		plt.draw()

		#Clear frames
		for zindex in range(len(self.redshift)):
			#Make an image of M-sigma at this redshift.
			print "Making frame"+str(zindex).zfill(3)

			#These are the plotted parameters.
			m_bhcut = np.array(self.m_bh[zindex])
			sigmacut = np.array(self.sigma[zindex])
			lbolcut = np.array(self.L_bol[zindex])
			eddRatiocut = lbolcut / sf.eddingtonLum(np.array(self.m_bh[zindex]))
			acut = np.array(self.spin_bh[zindex])

			#Rename old points
			oldStarPoints = starPoints
			oldCirclePoints = circlePoints
			oldSupernovaBarrier = supernovaBarrier

			#Ellipticals and non-circles are plotted separately.
			if shapeMode == 'ellipticals':
				shapeCut = np.array(self.elliptical[zindex])
			elif shapeMode == 'satellites':
				shapeCut = np.array(self.satelliteToCentral[zindex]) != -1
			circles, = np.where(shapeCut)
			stars, = np.where(~shapeCut)

			if colorMode == 'eddingtonRatio':
				#Color code by Eddington ratio
				e_colors = plt.cm.jet_r((np.log10(eddRatiocut[circles])-ratioRange[0])/(ratioRange[1]-ratioRange[0]))
				s_colors = plt.cm.jet_r((np.log10(eddRatiocut[stars])-ratioRange[0])/(ratioRange[1]-ratioRange[0]))
			elif colorMode == 'luminosity':
				#Color code by bolometric luminosity
				e_colors = plt.cm.jet_r((np.log10(lbolcut[circles])-lrange[0])/(lrange[1]-lrange[0]))
				s_colors = plt.cm.jet_r((np.log10(lbolcut[stars])-lrange[0])/(lrange[1]-lrange[0]))
			elif colorMode == 'spin':
				#Color code by dimensionless BH spin
				e_colors = plt.cm.jet_r((acut[circles])-arange[0])/(arange[1]-arange[0])
				s_colors = plt.cm.jet_r((acut[stars])-arange[0])/(arange[1]-arange[0])	

			#Create collections.
			if len(circles) > 0:
				if colorMode != 'spin':
					e_colors[eddRatiocut[circles]==0] = [0,0,0,1]
				circlePoints = plt.scatter(sigmacut[circles], m_bhcut[circles], marker='o',\
				color=e_colors, alpha=alphaScatter, s=100, zorder=3)
			else:
				circlePoints = None
			if len(stars) > 0:
				if colorMode != 'spin':
					s_colors[eddRatiocut[stars]==0] = [0,0,0,1]
				starPoints = plt.scatter(sigmacut[stars], m_bhcut[stars], marker='*',\
				color=s_colors, alpha=alphaScatter, s=100, zorder=4)
			else:
				starPoints = None

			if showSupernovaBarrier:
				Mcrit = sf.Mcrit(self.redshift[zindex])
				sigmaCrit = sf.velocityDispersion(Mcrit, self.redshift[zindex])
				supernovaBarrier = plt.plot([sigmaCrit,sigmaCrit], [1e-50,1e50], lw=2, ls='--', color='k', label='SN')

			#Clear old points
			if oldCirclePoints is not None:
				oldCirclePoints.remove()
			if oldStarPoints is not None:
				oldStarPoints.remove()
			if oldSupernovaBarrier is not None:
				oldSupernovaBarrier[0].remove()

			#Rename text
			ztext.set_text(r'$z ='+('%03.2f'%self.redshift[zindex])+'$')

			#Create figure.
			plt.draw()
			plt.savefig(temporaryFrameFolder + 'frame'+str(zindex).zfill(3)+'.png', dpi=dpi)

		#Create movie.
		folderToMovie(temporaryFrameFolder, mp4name)
		print("Movie saved to "+mp4name)

	def plotEvolutionMultipanel(self, zArr, boxsize=3, pdfname=None, pngname=None, colorMode='eddingtonRatio', colorbar=True, showMorphology=False, \
		plotData=True, shapeMode='ellipticals'):
		"""
		Multipanel PDF version of plotEvolution()
		"""

		#Inherit arrangement of panels from input
		zArr = np.array(zArr)
		if len(zArr.shape) == 1:
			zArr = np.expand_dims(zArr,axis=0)
		shape = zArr.shape

		#Plot Setup
		if colorbar:
			fig, axarr = plt.subplots(shape[0], shape[1], figsize=(boxsize*shape[1],boxsize*shape[0]*4.0/3.0), sharex=True, sharey=True)
		else:
			fig, axarr = plt.subplots(shape[0], shape[1], figsize=(boxsize*shape[1],boxsize*shape[0]), sharex=True, sharey=True)
		if len(axarr.shape) == 1:
			axarr = np.expand_dims(axarr,axis=0)
		
		#Color bar with code that I don't understand so that I can make it before the plot has any values.
		if colorMode == 'eddingtonRatio':
			ratioRange = (-4,0)
		elif colorMode == 'luminosity':
			lrange = (7,15)

		#Observed Relations
		relations = [(8.32,5.35)]
		relationLineStyles = [':']
		names = ['']

		#Differentiate galaxy symbols if we're showing morphology
		if showMorphology:
			scatterSymbols = ['o', '*']
		else:
			scatterSymbols = ['o', 'o']

		if plotData:
			with open('./bh_data/Saglia16.pkl', 'r') as myfile:
				data = pickle.load(myfile)
			data_sigma = np.array([10**logs[0] for logs in data['logsigma']])
			data_M = np.array([10**logm[0] for logm in data['logM_BH']])
			data_sigma_err = np.transpose(np.array([[10**logs[0]*(1.0-10**(-logs[1])),10**logs[0]*(10**logs[1]-1)] for logs in data['logsigma']]))
			data_M_err = np.transpose(np.array([[10**logm[0]*(1.0-10**(-logm[1])),10**logm[0]*(10**logm[1]-1)] for logm in data['logM_BH']]))

		#Loop over axes
		for i in range(shape[0]):
			for j in range(shape[1]):
				zindex = np.argmin(np.abs(self.redshift-zArr[i,j]))
				if i == shape[0]-1:
					axarr[i,j].set_xlabel(r'$\log(\sigma \, [\mathrm{km \, s^{-1}}])$', fontsize=16)
				if j == 0:
					axarr[i,j].set_ylabel(r'$\log(M_\mathrm{BH} \, [\mathrm{M}_\odot])$', fontsize=16)
				axarr[i,j].set_xscale('log')
				axarr[i,j].set_yscale('log')
				axarr[i,j].set_xlim(1,5e2)
				axarr[i,j].set_ylim(1e2,1e10)

				#Overplot observed relations
				if plotData:
					axarr[i,j].errorbar(data_sigma, data_M, xerr=data_sigma_err, yerr=data_M_err, fmt='none', ecolor='k', alpha=0.4, zorder=1)
				else:
					for rel in range(len(relations)):
						extremeSigmas = np.array([1,1e10])
						oplotMasses = 10**(relations[rel][0] + relations[rel][1]*(np.log10(extremeSigmas/200)))
						axarr[i,j].plot(extremeSigmas, oplotMasses, linewidth=2, linestyle=relationLineStyles[rel], label=names[rel], color='k')
				if (i == 0) & (j == 0):
					axarr[i,j].legend(loc=4, prop={'size':10}, frameon=False)
				ztext = axarr[i,j].text(1.3e0,1e9,r'$z ='+('%03.1f'%self.redshift[zindex])+'$', fontsize=16)

				#Actual data points.
				m_bhcut = np.array(self.m_bh[zindex])
				sigmacut = np.array(self.sigma[zindex])
				eddRatiocut = np.array(self.eddRatio[zindex])
				lbolcut = np.array(self.L_bol[zindex]) * eddRatiocut

				#Ellipticals and non-circles are plotted separately.
				if shapeMode == 'ellipticals':
					shapeCut = self.elliptical[zindex]
				circles = np.where(shapeCut)[0]
				stars = np.where([not x for x in shapeCut])[0]

				if colorMode == 'eddingtonRatio':
					#Color code by Eddington ratio
					e_colors = plt.cm.jet_r((np.log10(eddRatiocut[circles])-ratioRange[0])/(ratioRange[1]-ratioRange[0]))
					s_colors = plt.cm.jet_r((np.log10(eddRatiocut[stars])-ratioRange[0])/(ratioRange[1]-ratioRange[0]))
				elif colorMode == 'luminosity':
					#Color code by bolometric luminosity
					e_colors = plt.cm.jet_r((np.log10(lbolcut[circles])-lrange[0])/(lrange[1]-lrange[0]))
					s_colors = plt.cm.jet_r((np.log10(lbolcut[stars])-lrange[0])/(lrange[1]-lrange[0]))

				#Create collections.
				if len(circles) > 0:
					e_colors[eddRatiocut[circles]==0] = [0,0,0,1]
					axarr[i,j].scatter(sigmacut[circles], m_bhcut[circles], marker=scatterSymbols[0],\
					color=e_colors, s=25, alpha=0.7)
				if len(stars) > 0:
					s_colors[eddRatiocut[stars]==0] = [0,0,0,1]
					axarr[i,j].scatter(sigmacut[stars], m_bhcut[stars], marker=scatterSymbols[1],\
					color=s_colors, s=25, alpha=0.7)
		plt.tight_layout()
		if colorbar:
			plt.subplots_adjust(hspace=0,wspace=0,top=0.75)
			cbar_ax = fig.add_axes([0.12, 0.9, 0.8, 0.05])
			#Color bar with code that I don't understand so that I can make it before the plot has any values.
			if colorMode == 'eddingtonRatio':
				sm = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap('jet_r'), norm=plt.Normalize(vmin=ratioRange[0], vmax=ratioRange[1]))
				sm._A = []
				cb = fig.colorbar(sm, ticks=ratioRange, cax=cbar_ax, orientation='horizontal')
				cb.set_label(r'$\log(f_\mathrm{Edd})$', fontsize=14, labelpad=-1)
			elif colorMode == 'luminosity':
				sm = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap('jet_r'), norm=plt.Normalize(vmin=lrange[0], vmax=lrange[1]))
				sm._A = []
				cb = fig.colorbar(sm, ticks=lrange, cax=cbar_ax, orientation='horizontal')
				cb.set_label(r'$\log(L_\mathrm{Bol}/L_\odot)$', fontsize=14, labelpad=-1)
		else:
			plt.subplots_adjust(hspace=0,wspace=0)

		if pdfname is not None:
			plt.savefig(pdfname+'.pdf', dpi=1200)
		
		if pngname is not None:
			plt.savefig(pngname+'.png', dpi=1200)

		plt.show()

	def binner(self, data, bins):
		"""
		Returns bin index given data and bins.
		"""
		return np.floor((data-bins[0])/(bins[-1]-bins[0])*(len(bins)-1))

	def plotMergeRatio(self, zmax=3.5, mergeTime=2, nbins=5, binLims=(42,48)):
		"""
		Plot the fraction of AGN in mergers as a function of luminosity
		"""
		allLums = []
		allMerging = []
		for zindex in np.where(self.redshift < zmax)[0]:
			eddRatiocut = np.array(self.eddRatio[zindex])
			lbolcut = np.array(self.L_Edd[zindex]) * eddRatiocut * constants.L_sun * 1e7 #Switch to erg/s
			logLums = np.log10(lbolcut[lbolcut > 10**binLims[0]])
			lastMMcut = np.array(self.lastMajorMerger[zindex])
			isMerging = self.time[zindex] - lastMMcut <= mergeTime
			isMerging = isMerging[lbolcut > 10**binLims[0]]
			allLums.extend(logLums)
			allMerging.extend(isMerging)
		allLums = np.array(allLums)
		allMerging = np.array(allMerging)
		lbins = np.linspace(binLims[0],binLims[1],nbins+1)
		quasarBins = self.binner(allLums, lbins)
		countQuasars = [sum(quasarBins==i) for i in range(nbins)]
		mergeRatios = [float(sum(allMerging[quasarBins==i]))/max(countQuasars[i],1) for i in range(nbins)] #Note: max only here to prevent nans
		plt.errorbar(0.5*(lbins[:-1]+lbins[1:]), mergeRatios, yerr=[np.sqrt(countQuasars[i])/max(countQuasars[i],1) for i in range(nbins)], fmt='o')

		#Include results from Treister et al. 2012
		xaxis = np.linspace(42,48,100)
		plt.plot(xaxis, (xaxis-43.2)/4.5, linewidth=2, linestyle='-', color='k')
		plt.plot(xaxis, (10**xaxis/3e46)**0.4, linewidth=2, linestyle='--', color='k')
		plt.xscale('log')
		plt.yscale('log')
		plt.ylim(0.01,1)
		plt.xlim(42,48)
		plt.xticks(range(42,49,2), range(42,49,2))
		plt.xlabel(r'$\log(L_\mathrm{Bol}) \, \mathrm{[erg \, s^{-1}]}$', fontsize=16)
		plt.ylabel(r'$\mathrm{Fraction \enspace of \enspace AGN \enspace in \enspace Mergers}$', fontsize=16)
		plt.show()

	def plotStellar2Halo(self, mp4name='test', colorMode='eddingtonRatio', ffmpeg_alias='ffmpeg'):
		"""
		Plot the evolution of M-sigma
		"""

		fig, ax = plt.subplots(1, figsize=(7,6))
		plt.clf()
		plt.xlabel(r'$\log(M_\mathrm{Halo} \, [\mathrm{M}_\odot])$', fontsize=16)
		plt.ylabel(r'$\log(M_\mathrm{*} \, [\mathrm{M}_\odot])$', fontsize=16)
		plt.xscale('log')
		plt.yscale('log')
		plt.xlim(1e7,1e15)
		plt.ylim(1e4,1e13)

		#Overplot observed relations
		'''
		names = ['Feedback Limit']
		for rel in range(len(relations)):
			oplotMasses = 10**(relations[rel][0] + relations[rel][1]*(np.log10(extremeSigmas/200)))
			plt.plot(extremeSigmas, oplotMasses, linewidth=2, linestyle=relationLineStyles[rel], label=names[rel], color='k')
		plt.legend(loc=4)
		'''
		ztext = plt.text(4.3e0,1e9,r'$z ='+('%03.2f'%self.redshift[-1])+'$', fontsize=16)

		#Color bar with code that I don't understand so that I can make it before the plot has any values.
		if colorMode == 'eddingtonRatio':
			ratioRange = (-5,0)
			sm = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap('jet_r'), norm=plt.Normalize(vmin=ratioRange[0], vmax=ratioRange[1]))
			sm._A = []
			cb = plt.colorbar(sm, ticks=ratioRange)
			cb.set_label(r'$\log(f_\mathrm{Edd})$', fontsize=14)
		elif colorMode == 'luminosity':
			lrange = (7,15)
			sm = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap('jet_r'), norm=plt.Normalize(vmin=lrange[0], vmax=lrange[1]))
			sm._A = []
			cb = plt.colorbar(sm, ticks=lrange)
			cb.set_label(r'$\log(L_\mathrm{Bol}/L_\odot)$', fontsize=14)

		withBHPoints = None
		withoutBHPoints = None
		plt.tight_layout()
		plt.draw()

		#Clear frames
		os.system('rm ./frames/*')
		for zindex in range(len(self.redshift)):
			#Make an image of M-sigma at this redshift.
			print "Making frame"+str(zindex).zfill(3)

			#These are the plotted parameters.
			m_starcut = np.array(self.m_stars[zindex])
			m_halocut = np.array(self.m_halo[zindex])
			eddRatiocut = np.array(self.eddRatio[zindex])
			lbolcut = np.array(self.L_Edd[zindex]) * eddRatiocut
			hasBHcut = np.array(self.hasBH[zindex])

			#Rename old points
			oldWithBHPoints = withBHPoints
			oldWithoutBHPoints = withoutBHPoints

			#Those with and without BHs are plotted separately.
			withBH = np.where(hasBHcut)[0]
			withoutBH = np.where([not x for x in hasBHcut])[0]

			if colorMode == 'eddingtonRatio':
				#Color code by Eddington ratio
				with_colors = plt.cm.jet_r((np.log10(eddRatiocut[withBH])-ratioRange[0])/(ratioRange[1]-ratioRange[0]))
				without_colors = plt.cm.jet_r((np.log10(eddRatiocut[withoutBH])-ratioRange[0])/(ratioRange[1]-ratioRange[0]))
			elif colorMode == 'luminosity':
				#Color code by bolometric luminosity
				with_colors = plt.cm.jet_r((np.log10(lbolcut[withBH])-lrange[0])/(lrange[1]-lrange[0]))
				without_colors = plt.cm.jet_r((np.log10(lbolcut[withoutBH])-lrange[0])/(lrange[1]-lrange[0]))

			#Create collections.
			if len(withoutBH) > 0:
				without_colors[eddRatiocut[withoutBH]==0] = [0,0,0,1]
				withoutBHPoints = plt.scatter(m_halocut[withoutBH], m_starcut[withoutBH], marker='o',\
				color='white', edgecolor=without_colors, alpha=0.5, s=100)
			else:
				withoutBHPoints = None
			if len(withBH) > 0:
				with_colors[eddRatiocut[withBH]==0] = [0,0,0,1]
				withBHPoints = plt.scatter(m_halocut[withBH], m_starcut[withBH], marker='o',\
				color=with_colors, edgecolor=with_colors, alpha=0.5, s=100)
			else:
				withBHPoints = None

			#Clear old points
			if oldWithBHPoints is not None:
				oldWithBHPoints.remove()
			if oldWithoutBHPoints is not None:
				oldWithoutBHPoints.remove()

			#Rename text
			ztext.set_text(r'$z ='+('%03.1f'%self.redshift[zindex])+'$')

			#Create figure.
			plt.draw()
			plt.savefig('./frames/frame'+str(zindex).zfill(3)+'.png')

		#Create movie.
		os.system(ffmpeg_alias+' -framerate 5 -i ./frames/frame%03d.png -s:v 840x720 -c:v libx264 \
		-profile:v high -crf 23 -pix_fmt yuv420p -r 30 ./mp4s/'+mp4name+'.mp4')
		print "Movie saved to ./mp4s/"+mp4name+".mp4."

if __name__ == "__main__":
	analysis = SAM_Analysis(inpickle='test.pklz')
	#analysis.plotGrowthHistory(plotTime=False, depth=10, plotSpin=False)
	#analysis.plotGrowthTracks(depth=10)
	analysis.plotEvolution(colorMode='eddingtonRatio', plotData=True, alphaBeta=(8.45,5.0), showBar=True, framerate=40, shapeMode='ellipticals', showSupernovaBarrier=False)
	#analysis.plotStellar2Halo()
	#analysis.plotEvolutionMultipanel([[12,8,6],[4,2,1]], colorMode='eddingtonRatio', pdfname='test', colorbar=True, showMorphology=True)
	#analysis.plotEvolutionMultipanel([[6,2,0.5]], colorMode='eddingtonRatio', pdfname='test', colorbar=True, showMorphology=True, plotData=False)
