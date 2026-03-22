from __future__ import annotations

from enum import Enum

__all__ = ["WhichSolution", "FluxBalanceState", "Instrument"]


class WhichSolution(Enum):
    LOW = "low"
    NEUKIRCH_WIEGELMANN = "neuwie"
    NADOL_NEUKIRCH = "naneu"


class FluxBalanceState(Enum):
    BALANCED = "Balanced"
    UNBALANCED = "Unbalanced"


class Instrument(Enum):
    SOLAR_ORBITER = "Solar Orbiter"
    SDO = "SDO"
