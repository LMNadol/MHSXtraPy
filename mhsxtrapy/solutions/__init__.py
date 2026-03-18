from __future__ import annotations

from functools import lru_cache

from mhsxtrapy.types import WhichSolution

from ._low import LowSolution
from ._nadol_neukirch import NaNeuSolution
from ._neukirch_wiegelmann import NeuWieSolution

__all__ = []


@lru_cache
def get_solution(
    which,
    *,
    a: float | None = None,
    b: float | None = None,
    z0: float | None = None,
    deltaz: float | None = None,
    kappa: float | None = None,
):
    if which == WhichSolution.LOW:
        return LowSolution(kappa=kappa, a=a)
    elif which == WhichSolution.NEUKIRCH_WIEGELMANN:
        return NeuWieSolution(z0=z0, deltaz=deltaz, a=a, b=b)
    elif which == WhichSolution.NADOL_NEUKIRCH:
        return NaNeuSolution(z0=z0, deltaz=deltaz, a=a, b=b)
    else:
        raise ValueError(f"Unknown solution: {which}")
