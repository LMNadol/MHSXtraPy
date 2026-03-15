from enum import Enum

__all__ = ["WhichSolution", "FluxBalanceState"]


class WhichSolution(Enum):
    LOW = "low"
    NEUWIE = "neuwie"
    NANEU = "naneu"


class FluxBalanceState(Enum):
    BALANCED = "Balanced"
    UNBALANCED = "Unbalanced"
