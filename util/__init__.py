# Try pre-compiled Cython extensions; fall back to on-the-fly compilation for development.
try:
    from .crossmatch import crossmatch
    from .findFirstDuplicate2 import findFirstDuplicate2
    from .findFirstDuplicate2_2d import findFirstDuplicate2_2d
    from .findDuplicates import findDuplicates
    from .primariesAndSecondaries import primariesAndSecondaries
except ImportError:
    import numpy
    import pyximport
    pyximport.install(reload_support=True, setup_args={"include_dirs": numpy.get_include()})
    from .crossmatch import crossmatch
    from .findFirstDuplicate2 import findFirstDuplicate2
    from .findFirstDuplicate2_2d import findFirstDuplicate2_2d
    from .findDuplicates import findDuplicates
    from .primariesAndSecondaries import primariesAndSecondaries

from .folderToMovie import *
from .makeGaussianSmoothingKernel import *
from .convolve import *
