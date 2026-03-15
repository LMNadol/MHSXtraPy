from enum import Enum

__all__ = ["WhichSolution", "FluxBalanceState"]


class WhichSolution(Enum):
    LOW = "low"
    NEUKIRCH_WIEGELMANN = "neuwie"
    NADOL_NEUKIRCH = "naneu"


class FluxBalanceState(Enum):
    BALANCED = "Balanced"
    UNBALANCED = "Unbalanced"
