import numpy as np
from .. import constants
from .. import cosmology

import pyximport; pyximport.install(reload_support=True)

from .calcRemnantKick import *
from .calcRemnantSpin import *
