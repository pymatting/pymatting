# Import relevant submodules for ease-of-use
from pymatting.util import *
from pymatting.laplacian import *
from pymatting.solver import *
from pymatting.foreground import *
from pymatting.preconditioner import *
from pymatting.alpha import *
from pymatting.cutout import *

import importlib
__version__ = importlib.metadata.version(__name__)