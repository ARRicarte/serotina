#Installation of Cython modules. An alternative to this is to use setup.
import numpy
import pyximport
pyximport.install(reload_support=True, setup_args={"include_dirs":numpy.get_include()})

from .crossmatch import *
from .findFirstDuplicate2 import *
from .findFirstDuplicate2_2d import *
from .findDuplicates import *
from .primariesAndSecondaries import *
from .folderToMovie import *
