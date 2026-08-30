"""Body-fitted conformal spectral airfoil flow solver."""

from .mapping import AirfoilMapping
from .solver import FlowSolver, SolverConfig

__all__ = ["AirfoilMapping", "FlowSolver", "SolverConfig"]
__version__ = "0.1.0"

