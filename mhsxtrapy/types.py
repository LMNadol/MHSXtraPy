from enum import Enum


class WhichSolution(Enum):
    LOW = "Low"
    NEUWIE = "Neuwie"
    ASYMP = "Asymp"


class FluxBalanceState(Enum):
    BALANCED = "Balanced"
    UNBALANCED = "Unbalanced"
