from pymatting.foreground.estimate_foreground_cf import estimate_foreground_cf
from pymatting.foreground.estimate_foreground_ml import estimate_foreground_ml
from pymatting.foreground.estimate_foreground_ml import (
    estimate_foreground_ml as estimate_foreground,
)
try:
    from pymatting.foreground.estimate_foreground_ml_cupy import estimate_foreground_ml_cupy
    from pymatting.foreground.estimate_foreground_ml_pyopencl import estimate_foreground_ml_pyopencl
except ModuleNotFoundError:
    pass