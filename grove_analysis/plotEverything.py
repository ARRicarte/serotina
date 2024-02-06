"""
ARR:  05.02.16

Make every plot for a list of ensembles.
"""

#import matplotlib
#matplotlib.use('Agg')
import numpy as np
import os
from . import calcMSigmaz0 as ms
from . import plotBolometricLuminosityFunctions as lum
from . import plotMassFunctions as mass
from . import plotOccupationFractions as of
from .ensembleMSigma import EnsembleMSigma as ems
from . import plotAssemblyHistory as pah
from . import plotCoevolution as coev
from . import plotHaloMassFunctions as haloMass
from . import plotBlackHoleDensity as amd
from . import plotMergedMassFraction as pmf
from . import calcGravitationalWaveEvents as gwe
from . import plotSpinDistributions as psd

def plotEverything(ensembles, colors, labels, outFolder, \
	numberOfDexToConvolve=0.3, logHostMassRange=(10.6,15.0), n_mass=23, n_sample=20, iso=None, alphaBeta=None, \
	msigma0=False, lum_funct=False, mass_funct=False, threshold_occupation=False, \
	blq_mass_funct=False, mass_funct_local=False, occupation=False, mass_weighted_occupation=False, \
	mergerFraction=False, ms_evolution=False, lum_past6=False, mass_past6=False, coevolution=False, \
	illustrateConvolution=False, assemblyHistoryUnnormed=False, assemblyHistoryNormed=False, \
	relativeMsigma=False, accretedMassDensity=False, halo_mass_funct=False, blackHoleMergers=False, spinDistributions=False, occupationIndex=3):

	#Make a folder
	if outFolder[-1] != '/':
		outFolder += '/'
	if os.path.exists(outFolder):
		print("Using existing folder " + outFolder + "for plotting output.")
	else:
		print("Creating " + outFolder + " for plotting output.")
		os.mkdir(outFolder)

	#Special default
	if alphaBeta is None:
		alphaBeta = [[8.45,5.0]]*len(ensembles)

	#M-sigma
	if msigma0:
		print("The M-sigma Relation")
		ms.calcMSigmaz0(ensembles, colors=colors, labels=labels, output=outFolder+'msigma.pdf', alphaBeta=alphaBeta, iso=iso, includeSatellites=False)

	#Bolometric luminosity functions
	if lum_funct:
		print("Bolometric Luminosity Functions")
		redshiftSlices = [0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; xlim=(1e9,1e15); ylim=(1e-9,1e0); figsize=(8,8); plotUnconvolved=False
		lumFuncts = [lum.calcBolometricLuminosityFunctions(e, redshiftSlices=redshiftSlices, logRange=logHostMassRange, n_mass=n_mass, n_sample=n_sample, logAGNLumBins=np.linspace(6,15,101)) for e in ensembles]
		lum.plotLuminosityFunctions(lumFuncts, labels=labels, colors=colors, redshiftSlices=redshiftSlices, plotUnconvolved=plotUnconvolved, showCompleteness=False, \
		figsize=figsize, xlim=xlim, ylim=ylim, numberOfDexToConvolve=numberOfDexToConvolve, showLynx=False, showOnoue=False, output=outFolder+'lum.pdf')

	#Occupation Fractions
	if occupation:
		print("Occupation Fractions")
		redshifts = [0.0,2.0,6.0]
		curveLabels = ['z=0','z=2','z=6']
		specialColors = ['royalblue', 'forestgreen', 'firebrick']
		of_calculations = [of.computeOccupationFractions(ensembles[occupationIndex], redshifts=redshifts, nbins=20, weightByMass=False, takeLog=False)]
		of.plotOccupationFractions(of_calculations, colors=specialColors, labels=curveLabels, figsize=(4,4), output=outFolder+'occupation.pdf', redshifts=redshifts, patchy=True)

	#Mass-weighted Occupation Fraction
	if mass_weighted_occupation:
		print("Mass-weighted Occupation Fractions")
		curveLabels = ['z=0','z=2','z=6']
		redshifts = [0.0,2.0,6.0]
		specialColors = ['royalblue', 'forestgreen', 'firebrick']
		of_calculations = [of.computeOccupationFractions(e, redshifts=redshifts, nbins=20, weightByMass=True, takeLog=True) for e in ensembles]
		of.plotEffectiveOccupationFraction(of_calculations, colors=specialColors, curveLabels=curveLabels, redshifts=redshifts, figshape=(2,2), figsize=(6,6), textLabel=labels, output=outFolder+'occupation_effective.pdf', patchy=True)

	#Threshold Occupation
	if threshold_occupation:
		print("Threshold Occupation")
		curveLabels = ['z=0','z=2','z=6']
		redshifts = [0]
		specialColors = ['royalblue', 'forestgreen', 'firebrick']
		of_calculations = [of.computeOccupationFractions(e, redshifts=redshifts, nbins=20, weightByMass=False, takeLog=False, minimumMass=3e5) for e in ensembles]
		of.plotOccupationFractions_minMass(of_calculations, colors=specialColors, curveLabels=curveLabels, redshifts=redshifts, figshape=(2,2), figsize=(6,6), textLabel=labels, output=outFolder+'occupation_threshold.pdf', patchy=True)

	#Mass functions
	if mass_funct:
		print("Mass Functions")
		redshiftSlices = [0.6, 1.0, 2.0, 3.0, 4.0, 5.0]; xlim=(5e6,2e10); ylim=(1e-6,1e-1); figsize=(8,5)
		massFuncts = [mass.calcMassFunctions(e, redshiftSlices=redshiftSlices, logRange=logHostMassRange, n_mass=n_mass, n_sample=n_sample) for e in ensembles]
		mass.plotMassFunctions(massFuncts, labels=labels, colors=colors, redshiftSlices=redshiftSlices, figsize=figsize, xlim=xlim, ylim=ylim, \
		output=outFolder+'mass.pdf', showAnalytic=False, numberOfDexToConvolve=numberOfDexToConvolve, showCompleteness=False)

	#Mass functions for broad-line quasars
	if blq_mass_funct:
		print("Mass Functions for Broad Line Quasars")
		redshiftSlices = [0.6, 1.0, 1.6, 2.15, 3.2, 4.75]; xlim=(1e8,1e11); ylim=(2e-8,4e-3); figsize=(8,5); eddMin=1e-2
		massFuncts = [mass.calcMassFunctions(e, redshiftSlices=redshiftSlices, weightByObscuration=True, logRange=logHostMassRange, n_mass=n_mass, n_sample=n_sample, eddMin=eddMin) for e in ensembles]
		mass.plotMassFunctions(massFuncts, labels=labels, colors=colors, redshiftSlices=redshiftSlices, figsize=figsize, xlim=xlim, ylim=ylim, showCompleteness=False, \
		includeQuasars=True, quasarScaling=1, includeMH=False, showAnalytic=False, numberOfDexToConvolve=numberOfDexToConvolve, output=outFolder+'mass_blq.pdf')

	#Mass functions
	if mass_funct_local:
		print("Local Mass Functions")
		redshiftSlices = [0.0]; xlim=(1e6,2e10); ylim=(1e-5,1e-1); figsize=(4,4)
		massFuncts = [mass.calcMassFunctions(e, redshiftSlices=redshiftSlices, logRange=logHostMassRange, n_mass=n_mass, n_sample=n_sample) for e in ensembles]
		mass.plotMassFunctions(massFuncts, labels=labels, colors=colors, redshiftSlices=redshiftSlices, figsize=figsize, xlim=xlim, ylim=ylim, \
		includeQuasars=False, includeMH=True, numberOfDexToConvolve=numberOfDexToConvolve, output=outFolder+'mass_local.pdf')

	#Fraction of mass acquired by mergers
	if mergerFraction:
		print("Fraction of Mass from BH Mergers")
		pmf.plotMergedMassFraction(ensembles, colors=colors, labels=labels, output=outFolder+'mergerFraction.pdf')

	#Evolution on the M-sigma plane
	if ms_evolution:
		print("M-sigma Evolution")
		redshifts = [0.2, 2.0, 4.0, 6.0, 8.0, 10.0]
		cbranges = [[-5.0,0.7], [-5.0,-1.0]]
		for e_index in range(len(ensembles)):
			plotter = ems(ensembles[e_index])
			plotter.assembleData(redshiftsDesired=redshifts)

			#All BHs
			plotter.makeMSigmaDensityMaps(minimumLuminosity=0, maximumLuminosity=np.inf, density=75)
			plotter.plotMSigmaDensityMaps(colorMap='viridis', cbrange=cbranges[e_index], output=outFolder+'ms_evolution_'+ensembles[e_index]+'_all.pdf')

			#Just luminous BHs
			plotter.makeMSigmaDensityMaps(minimumLuminosity=1e10, maximumLuminosity=np.inf, density=75)
			plotter.plotMSigmaDensityMaps(colorMap='viridis', cbrange=cbranges[e_index], output=outFolder+'ms_evolution_'+ensembles[e_index]+'_1e10.pdf')

	#Luminosity Functions Beyond 6
	if lum_past6:
		print("Luminosity Functions Beyond 6")
		redshiftSlices = [6.0, 7.0, 8.0, 9.0, 10.0, 12.0]; figsize=(8,5); xlim=(1e7,1e15); ylim=(1e-9,1e0); plotUnconvolved=False
		lumFuncts = [lum.calcBolometricLuminosityFunctions(e, redshiftSlices=redshiftSlices, n_mass=n_mass, n_sample=n_sample, logRange=logHostMassRange) for e in ensembles]
		lum.plotLuminosityFunctions(lumFuncts, labels=labels, colors=colors, redshiftSlices=redshiftSlices, plotUnconvolved=plotUnconvolved, showCompleteness=False, \
		figsize=figsize, xlim=xlim, ylim=ylim, numberOfDexToConvolve=numberOfDexToConvolve, output=outFolder+'lum_past6.pdf', showLynx=True, showOnoue=False)

	#Mass Functions Beyond 6
	if mass_past6:
		print("Mass Functions Past 6.")
		redshiftSlices = [6.0, 7.0, 8.0, 9.0, 10.0, 12.0]; xlim=(5e4,1e9); ylim=(1e-6,1e0); figsize=(8,5)
		massFuncts = [mass.calcMassFunctions(e, redshiftSlices=redshiftSlices, n_mass=n_mass, n_sample=n_sample, logRange=logHostMassRange) for e in ensembles]
		mass.plotMassFunctions(massFuncts, labels=labels, colors=colors, redshiftSlices=redshiftSlices, figsize=figsize, xlim=xlim, ylim=ylim, showCompleteness=False, \
		numberOfDexToConvolve=numberOfDexToConvolve, output=outFolder+'mass_past6.pdf')

	#Coevolution
	if coevolution:
		print("Coevolution")
		n_bootstrap = 1000
		percentileRange = [14,86]
		logBinEdges = np.linspace(3.5,10,13)
		ratio = True
		stellar = True
		ylabel = r'$<\log_{10}(M_\bullet/M_*)>$'

		redshiftPanels = [0.5, 2.0, 4.0, 6.0, 8.0, 10.0]
		valueRanges = [[coev.calculateCoevolution(ensemble, logBinEdges, z, n_tree=n_sample, n_bootstrap=n_bootstrap, \
		percentileRange=percentileRange, ratio=ratio, stellar=stellar) for z in redshiftPanels] for ensemble in ensembles]
		coev.plotCoevolution(logBinEdges, valueRanges, labels=labels, colors=colors, redshifts=redshiftPanels, ratio=ratio, \
		ylabel=ylabel, output=outFolder+'coevolution.pdf')

	#Convolution Illustration
	if illustrateConvolution:
		print("Convolution")
		redshiftSlices = [0.2]; xlim=(1e9,1e15); ylim=(1e-9,1e0); figsize=(4,4)
		#Assuming that the last ensemble is the one you want to plot
		lumFuncts = [lum.calcBolometricLuminosityFunctions(ensembles[-1], redshiftSlices=redshiftSlices, n_mass=n_mass, n_sample=n_sample, logRange=logHostMassRange)]
		lum.plotLuminosityFunctions(lumFuncts, labels=[labels[-1]], colors=[colors[-1]], redshiftSlices=redshiftSlices, plotUnconvolved=True, plotConvolved=True, \
		figsize=figsize, xlim=xlim, ylim=ylim, output=outFolder+'illustrateConvolution.pdf')

	#Accreted Mass Density
	if accretedMassDensity:
		print("Accreted Mass Density")
		densities = []
		densitiesAccreted = []
		luminosityDensities = []
		for e_index in range(len(ensembles)):
			redshifts, ranges, rangesAccreted, rangesLumDensity = amd.computeBlackHoleDensity(ensembles[e_index], n_mass=n_mass, n_sample=n_sample, logRange=logHostMassRange)
			densities.append(ranges)
			densitiesAccreted.append(rangesAccreted)
			luminosityDensities.append(rangesLumDensity)

		amd.plotBlackHoleDensities(redshifts, densitiesTotal=densities, densitiesAccreted=densitiesAccreted, labels=labels, colors=colors, numberOfDexToConvolve=numberOfDexToConvolve, \
		output=outFolder+'massDensity.pdf')
		amd.plotAccretionDensities(redshifts, luminosityDensities, labels=labels, colors=colors, numberOfDexToConvolve=numberOfDexToConvolve, \
		output=outFolder+'accretionDensity.pdf')

	#Assembly History, unnormalized
	if assemblyHistoryUnnormed:
		print("Assembly history, unnormalized")
		hostMasses = [1e11, 1e12, 1e13, 1e14, 1e15]
		colors = ['indigo', 'darkcyan', 'darkgreen', 'goldenrod', 'firebrick']
		assemblyHistories = []
		for e_index in range(1):
			redshifts, masses, ranges = pah.computeAssemblyHistories(ensembles[e_index], percentileRange=[14,86], hostMasses=hostMasses, normByZero=False)
			assemblyHistories.append(ranges)
		pah.plotAssemblyHistories(redshifts, assemblyHistories, labels=labels, colors=colors, masses=masses, normalized=False, output=outFolder+'assemblyHistory_unnormed.pdf', figsize=(4,4))

	#Assembly History, normalized
	if assemblyHistoryNormed:
		print("Assembly history, normalized")
		hostMasses = [1e11, 1e12, 1e13, 1e14, 1e15]
		colors = ['indigo', 'darkcyan', 'darkgreen', 'goldenrod', 'firebrick']
		assemblyHistories = []
		for e_index in range(1):
			redshifts, masses, ranges = pah.computeAssemblyHistories(ensembles[e_index], percentileRange=[14,86], hostMasses=hostMasses, normByZero=True)
			assemblyHistories.append(ranges)
		pah.plotAssemblyHistories(redshifts, assemblyHistories, labels=labels, colors=colors, masses=masses, normalized=True, output=outFolder+'assemblyHistory_normed.pdf', figsize=(4,4))

	#Relative M-sigma
	if relativeMsigma:
		print("Relative M-sigma")
		hostMasses = [1e11, 1e12, 1e13, 1e14, 1e15]
		colors = ['indigo', 'darkcyan', 'darkgreen', 'goldenrod', 'firebrick']
		normalize = False
		divideByMsigma = True
		for e_index in range(len(ensembles)):
			redshifts, masses, ranges = pah.computeAssemblyHistories(ensembles[e_index], percentileRange=[14,86], hostMasses=hostMasses, normByZero=False, divideByMsigma=True)
			assemblyHistories.append(ranges)
		pah.plotRelativeMsigma(redshifts, assemblyHistories, labels=labels, colors=colors, masses=masses)

	#Halo Mass Functions
	if halo_mass_funct:
		print("Halo mass functions")
		redshiftSlices = [0.2, 0.6, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]; xlim=(1e6,1e15); ylim=(1e-6,1e3); figsize=(8,10)
		#Assuming that the last ensemble is the one you want to plot
		haloMassFunct = [haloMass.calcHaloMassFunctions(ensembles[-1], redshiftSlices=redshiftSlices, n_mass=n_mass, n_sample=n_sample, logRange=logHostMassRange)]
		haloMass.plotMassFunctions(haloMassFunct, labels=labels[-1], colors=colors[-1], redshiftSlices=redshiftSlices, figsize=figsize, xlim=xlim, ylim=ylim, \
		output=outFolder+'haloMassFunction.pdf')

	#Black hole merger events
	if blackHoleMergers:
		print("Black hole mergers")
		gw_packs_z = []
		gw_packs_m = []
		for e_index in range(len(ensembles)):
			calculator = gwe.GravitationalWaveEventCalculator(ensembles[e_index], logRange=logHostMassRange, n_mass=n_mass, n_sample=n_sample)
			calculator.evaluateSignalToNoise()
			gw_packs_z.append(calculator.computeEventRateByRedshift(chirpMassLimits=(0,np.inf)))
			gw_packs_m.append(calculator.computeEventRateByMass())
		#gwe.plotEventRateByRedshift(gw_packs_z, colors, labels, yearsOfObservation=4, ylim=(1e-1,5e1), xlim=(0,20), output=outFolder+'gw_byz.pdf')
		#gwe.plotEventRateByMass(gw_packs_m, colors, labels, yearsOfObservation=4, ylim=(1e-1,5e1), xlim=(3e2,1e6), output=outFolder+'gw_bym.pdf')
		gwe.plotEventRateByRedshift(gw_packs_z, colors, labels, yearsOfObservation=4, ylim=(1e0,5e2), xlim=(0,20), output=outFolder+'gw_byz.pdf')
		gwe.plotEventRateByMass(gw_packs_m, colors, labels, yearsOfObservation=4, ylim=(1e0,5e2), xlim=(3e2,1e6), output=outFolder+'gw_bym.pdf')
		gwe.computeTotalEvents(gw_packs_m, yearsOfObservation=4)

	#Bolometric luminosity functions
	if spinDistributions:
		print("Spin")
		redshiftSlices = [0.0, 2.0, 6.0]; xlim=(0,1); ylim=(0,1); figsize=(8,4)
		psd.plotSpinDistributionGrid(ensembles, listOfRedshifts=redshiftSlices, listOfLabels=labels, listOfColors=colors, Mbh_range=[1e6,1e7], show=False, output=outputFolder+'spin_6to7.pdf', xlim=xlim, ylim=ylim, figsize=figsize)
		psd.plotSpinDistributionGrid(ensembles, listOfRedshifts=redshiftSlices, listOfLabels=labels, listOfColors=colors, Mbh_range=[1e7,1e8], show=False, output=outputFolder+'spin_7to8.pdf', xlim=xlim, ylim=ylim, figsize=figsize)
		psd.plotSpinDistributionGrid(ensembles, listOfRedshifts=redshiftSlices, listOfLabels=labels, listOfColors=colors, Mbh_range=[1e8,1e9], show=False, output=outputFolder+'spin_8to9.pdf', xlim=xlim, ylim=ylim, figsize=figsize)
		psd.plotSpinDistributionGrid(ensembles, listOfRedshifts=redshiftSlices, listOfLabels=labels, listOfColors=colors, Mbh_range=[1e8,np.inf], show=False, output=outputFolder+'spin_8toinf.pdf', xlim=xlim, ylim=ylim, figsize=figsize)

if __name__ == '__main__':

	#Input parameters
	outFolder = 'NSF/'
	#outFolder = 'BackgroundsPaper/'
	#outFolder = 'SeedingPaper2/'
	#outFolder = 'SeedingPaper2_optimistic/'

	#Original SAM Paper runs
	'''
	alphaBeta = [(8.45,5.0), (8.45,5.0), (8.45,5.0), (8.0,4.0)]
	ensembles = ['burstonly_dcbh_072817', 'powerlaw_dcbh_072717', 'agnms_dcbh_072717', 'iso_dcbh_080417']
	colors = ['b', 'purple',  'c', 'g']
	labels = ['Burst', 'PowerLaw', 'AGNMS', 'Iso']
	iso = [False, False, False, True]
	logHostMassRange = (11.0,15.0)
	n_mass = 20
	n_sample = 15
	'''

	#Seeding Paper Runs
	'''
	alphaBeta = [(8.45,5.0), (8.45,5.0), (8.45,5.0), (8.45,5.0)]
	ensembles = ['powerLaw_popIII_pmerge0.1_072018', 'agnms_popIII_pmerge0.1_072018', 'powerLaw_dcbh_pmerge0.1_072018', 'agnms_dcbh_pmerge0.1_072018']
	labels = ['Light-PL', 'Light-MS', 'Heavy-PL', 'Heavy-MS']
	ensembles = ['powerLaw_popIII_pmerge1.0_072018', 'agnms_popIII_pmerge1.0_072018', 'powerLaw_dcbh_pmerge1.0_072018', 'agnms_dcbh_pmerge1.0_072018']
	labels = ['Light-PL*', 'Light-MS*', 'Heavy-PL*', 'Heavy-MS*']
	colors = ['r', 'orange', 'purple', 'c']
	iso = [False, False, False, False]
	'''

	#Clustering of the Backgrounds
	'''
	ensembles = ['blq_dcbh_pmerge0.1_072018', 'blq_popIII_pmerge0.1_072018']
	colors = ['firebrick', 'royalblue']
	labels = ['Heavy', 'Light']
	iso = None
	'''

	#NSF
	ensembles = ['powerLaw_popIII_pmerge0.1_072018', 'powerLaw_dcbh_pmerge0.1_072018', 'blq_popIII_pmerge0.1_072018']
	colors = ['r', 'purple', 'blue']
	labels = ['Light', 'Heavy', 'Light-NoEdd']
	iso = None
	#Log-normal convolution kernel
	numberOfDexToConvolve = 0.3

	#Merger tree library properties
	logHostMassRange = (10.6,15.0)
	n_mass = 23
	n_sample = 20

	#Treated as booleans
	msigma0 = 0
	lum_funct = 0
	mass_funct = 0
	blq_mass_funct = 0
	mass_funct_local = 0
	occupation = 0
	mass_weighted_occupation = 0
	threshold_occupation = 0
	mergerFraction = 0
	ms_evolution = 0
	lum_past6 = 1
	mass_past6 = 0
	coevolution = 0
	illustrateConvolution = 0
	assemblyHistoryUnnormed = 0
	assemblyHistoryNormed = 0
	relativeMsigma = 0
	accretedMassDensity = 0
	halo_mass_funct = 0
	blackHoleMergers = 0

	occupationIndex = 3

	plotEverything(ensembles, colors, labels, outFolder, numberOfDexToConvolve=numberOfDexToConvolve, \
	logHostMassRange=logHostMassRange, n_mass=n_mass, n_sample=n_sample, \
	msigma0=msigma0, lum_funct=lum_funct, mass_funct=mass_funct, blq_mass_funct=blq_mass_funct, threshold_occupation=threshold_occupation, \
	mass_funct_local=mass_funct_local, occupation=occupation, mass_weighted_occupation=mass_weighted_occupation, \
	mergerFraction=mergerFraction, ms_evolution=ms_evolution, lum_past6=lum_past6, mass_past6=mass_past6, \
	coevolution=coevolution, illustrateConvolution=illustrateConvolution, assemblyHistoryUnnormed=assemblyHistoryUnnormed, \
	assemblyHistoryNormed=assemblyHistoryNormed, relativeMsigma=relativeMsigma, accretedMassDensity=accretedMassDensity, \
	halo_mass_funct=halo_mass_funct, blackHoleMergers=blackHoleMergers, occupationIndex=occupationIndex)
