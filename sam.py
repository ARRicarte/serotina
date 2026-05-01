"""
Semi-analytic model for the cosmic co-evolution of supermassive black holes.

The SAM class is assembled from focused mixin modules:
  _params.py     — parameter parsing and defaults
  _storage.py    — array initialization for progenitors and black holes
  _seeding.py    — black hole seeding prescriptions (DCBH, PopIII, NSC)
  _accretion.py  — accretion integration and BH-progenitor matching
  _mergers.py    — BH and halo merger mechanics
  _evolution.py  — main event-driven evolution loop
  _output.py     — snapshots, serialization, and diagnostics
"""

import numpy as np
import gzip
from . import cosmology_functions
from .helpers import smhm

from ._params import ParametersMixin
from ._storage import StorageMixin
from ._seeding import SeedingMixin
from ._accretion import AccretionMixin
from ._mergers import MergersMixin
from ._evolution import EvolutionMixin
from ._output import OutputMixin


class SAM(ParametersMixin, StorageMixin, SeedingMixin,
          AccretionMixin, MergersMixin, EvolutionMixin, OutputMixin):
	"""
	Semi-analytic model for black hole evolution across cosmic time.

	Loads a dark matter merger tree, paints black holes onto halos, and
	evolves them forward in time via accretion and mergers.

	Parameters
	----------
	treefile : str
	    Path to a gzip-compressed numpy merger tree file.
	parameters : dict, optional
	    SAM configuration.  See setParameters() for the full list of keys
	    and their defaults.
	silent : bool, optional
	    Suppress progress output.
	"""

	def __init__(self, treefile, parameters=None, silent=False):

		#Interpret the input parameters and assign them to the SAM.
		self.setParameters(parameters)

		#Open merger tree file.  These are zzip-compressed numpy arrays.
		with gzip.open(treefile, 'rb') as myfile:
			if not silent:
				print("Opening dark matter merger tree: " + treefile)
			mergerTreeTable = np.load(myfile)

		#Tree files have a particular structure that I imposed.
		self.m_halo = mergerTreeTable[:,0]
		self.redshift = mergerTreeTable[:,1]
		self.parent = mergerTreeTable[:,2].astype(int) - 1
		self.progenitor = mergerTreeTable[:,3].astype(int) - 1
		self.nchild = mergerTreeTable[:,4].astype(int)
		#self.m_smooth = mergerTreeTable[:,5]  #This is in the file, but it's not needed.
		self.n_node = mergerTreeTable.shape[0]
		self.nodeProperties = ['m_halo', 'redshift', 'parent', 'progenitor', 'nchild', 'time']

		#Get a time axis, representing age of the universe.
		if not silent:
			print("Calculating times and stellar masses.")
		self.uniqueRedshift = np.flipud(np.unique(self.redshift))
		self.uniqueTime = cosmology_functions.z2t(self.uniqueRedshift)		#Gyr
		self.time = np.zeros(self.n_node)
		for lev in range(len(self.uniqueRedshift)):
			timeAtLev = self.uniqueTime[len(self.uniqueTime)-1-lev]
			self.time[self.redshift==self.uniqueRedshift[len(self.uniqueTime)-1-lev]] = timeAtLev

		#Stellar masses
		self.m_star = smhm.Mstar(self.m_halo, self.redshift)
		self.nodeProperties.append('m_star')

		#Set up storage.  Black holes move among progenitors, which move along nodes.  If you ever want to start over, use self.initializeStorage()
		if not silent:
			print("Initializing storage arrays.")
		self.initializeStorage(keepSpins=False)

		print("SAM initialized.")


if __name__ == "__main__":
	from .helpers import sam_functions as sf
	#Obtain the appropriate merger tree
	m_halo = 1e13
	i_tree = 1
	path = None
	lorax = sf.treeNavigator(path=path)
	treefile = path + '/' + lorax.findClosestTree(m_halo, i_tree)
	parameters = None

	#Run the SAM
	outputNameBase = 'test'
	savedProperties = ['m_bh', 'm_halo', 'indices', 'L_bol', 'eddRatio', 'spin_bh', 'redshift', \
	'elliptical', 'time', 'lastMajorMerger', 'bhToProg', 'bh_id', 'satelliteToCentral', \
	'm_steady', 'm_burst', 'sigma']
	howToSave = 'mainProgenitor'
	savedRedshifts = None

	model = SAM(treefile, parameters=parameters)
	model.evolveUniverse(outputNameBase=outputNameBase, savedRedshifts=savedRedshifts, howToSave=howToSave, savedProperties=savedProperties)
