from __future__ import annotations

from ._3d import plot_magnetogram_3D
from ._pp import plot_ddensity_xy, plot_dpressure_xy
from .plot import (
    find_corners_SDO,
    find_corners_SolarOrbiter,
    plot_ddensity_z,
    plot_dpressure_z,
    plot_magnetogram_2D,
    show_footpoints,
    show_poles,
)

__all__: list[str] = [
    "plot_magnetogram_2D",
    "plot_dpressure_z",
    "plot_ddensity_z",
    "plot_magnetogram_3D",
    "plot_dpressure_xy",
    "plot_ddensity_xy",
    "show_poles",
    "show_footpoints",
    "find_corners_SolarOrbiter",
    "find_corners_SDO",
]
