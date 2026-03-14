from enum import Enum


class WhichSolution(Enum):
    LOW = "low"
    NEUWIE = "neuwie"
    NANEU = "naneu"


class FluxBalanceState(Enum):
    BALANCED = "Balanced"
    UNBALANCED = "Unbalanced"
