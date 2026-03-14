from mhsxtrapy.types import WhichSolution

from .low import LowSolution
from .naneu import NaNeuSolution
from .neuwie import NeuWieSolution


def get_solution(which):
    if which == WhichSolution.LOW:
        return LowSolution()
    elif which == WhichSolution.NEUWIE:
        return NeuWieSolution()
    elif which == WhichSolution.NANEU:
        return NaNeuSolution()
    else:
        raise ValueError(f"Unknown solution: {which}")
