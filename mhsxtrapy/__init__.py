from importlib.metadata import version

from mhsxtrapy._boundary import (
    BoundaryData,
    alpha_HS04,
    is_flux_balanced,
    max_a_parameter,
)
from mhsxtrapy._field import ExtrapolationResult, extrapolate
from mhsxtrapy.types import FluxBalanceState, Instrument, WhichSolution

__version__ = version("mhsxtrapy")

__all__ = [
    "extrapolate",
    "ExtrapolationResult",
    "__version__",
    "BoundaryData",
    "is_flux_balanced",
    "alpha_HS04",
    "max_a_parameter",
    "WhichSolution",
    "FluxBalanceState",
    "Instrument",
]
