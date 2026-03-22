from __future__ import annotations

from functools import lru_cache

from mhsxtrapy.types import WhichSolution

from ._base import BaseSolution
from ._low import LowSolution
from ._nadol_neukirch import NadolNeukirchSolution
from ._neukirch_wiegelmann import NeukirchWiegelmannSolution

__all__ = ["get_solution"]


@lru_cache
def get_solution(
    which: WhichSolution,
    *,
    a: float | None = None,
    b: float | None = None,
    z0: float | None = None,
    deltaz: float | None = None,
    kappa: float | None = None,
) -> BaseSolution:
    if which == WhichSolution.LOW:
        return LowSolution(kappa=kappa, a=a)
    elif which == WhichSolution.NEUKIRCH_WIEGELMANN:
        return NeukirchWiegelmannSolution(z0=z0, deltaz=deltaz, a=a, b=b)
    elif which == WhichSolution.NADOL_NEUKIRCH:
        return NadolNeukirchSolution(z0=z0, deltaz=deltaz, a=a, b=b)
    else:
        raise ValueError(f"Unknown solution: {which}")
