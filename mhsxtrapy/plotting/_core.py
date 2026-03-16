from __future__ import annotations

from typing import Tuple

import numpy as np
from matplotlib import colormaps, colors, rc
from scipy.ndimage import find_objects, label, maximum_filter, minimum_filter

from mhsxtrapy.constants import LATEX_ON
from mhsxtrapy.field3d import Field3dData
from mhsxtrapy.types import FluxBalanceState

rc("text", usetex=LATEX_ON)

__all__ = [
    "cmap_magneto",
    "cmap_pressure",
    "cmap_density",
    "norm_aia",
    "cmap_aia",
    "norm_hmi",
    "detect_footpoints",
    "calculate_tick_count",
    "set_axis_labels",
]

cmap_magneto = colors.LinearSegmentedColormap.from_list(
    "cmap_magneto",
    (
        # Edit this gradient at https://eltos.github.io/gradient/#magnetogram=2D2D2D-D3D3D3
        (0.000, (0.176, 0.176, 0.176)),
        (1.000, (1.000, 1.000, 1.000)),
    ),
)

cmap_pressure = colors.LinearSegmentedColormap.from_list(
    "cmap_pressure",
    (
        # Edit this gradient at https://eltos.github.io/gradient/#cmap=FDF8ED-F1F6FC-D0CBF4-9F9FF9-8080F8-5556B1-3A369F-24216C-151920
        (0.000, (0.992, 0.973, 0.929)),
        (0.125, (0.945, 0.965, 0.988)),
        (0.250, (0.816, 0.796, 0.957)),
        (0.375, (0.624, 0.624, 0.976)),
        (0.500, (0.502, 0.502, 0.973)),
        (0.625, (0.333, 0.337, 0.694)),
        (0.750, (0.227, 0.212, 0.624)),
        (0.875, (0.141, 0.129, 0.424)),
        (1.000, (0.082, 0.098, 0.125)),
    ),
)

cmap_density = colors.LinearSegmentedColormap.from_list(
    "cmap_density",
    (
        # Edit this gradient at https://eltos.github.io/gradient/#cmap=FDF8ED-FCF1F8-F4CBE4-F99FCB-F880C0-D35E7E-BD2E49-871821-201515
        (0.000, (0.992, 0.973, 0.929)),
        (0.125, (0.988, 0.945, 0.973)),
        (0.250, (0.957, 0.796, 0.894)),
        (0.375, (0.976, 0.624, 0.796)),
        (0.500, (0.973, 0.502, 0.753)),
        (0.625, (0.827, 0.369, 0.494)),
        (0.750, (0.741, 0.180, 0.286)),
        (0.875, (0.529, 0.094, 0.129)),
        (1.000, (0.125, 0.082, 0.082)),
    ),
)

norm_aia = colors.SymLogNorm(50, vmin=6e1, vmax=100e02)
cmap_aia = colormaps["sdoaia171"]

norm_hmi = colors.SymLogNorm(50, vmin=-7.5e2, vmax=7.5e2)


def detect_footpoints(data: Field3dData) -> Tuple:
    """
    Detenct footpoints around centres of poles on photospheric magentogram.

    Args:
        data (Field3dData): magnetic field data

    Returns:
        Tuple: sink and source regions where footpoints will be plotted
    """

    sinks = data.bz.copy()
    sources = data.bz.copy()

    maxmask = sources < sources.max() * 0.4
    sources[maxmask != 0] = 0

    minmask = sinks < sinks.min() * 0.4
    sinks[minmask == 0] = 0

    return sinks, sources


def calculate_tick_count(min_val, max_val, relative_size):
    """
    Calculate optimal tick spacing considering the relative size of this axis
    compared to the largest axis in the plot.

    Args:
        min_val (_type_): minimum value of the axis
        max_val (_type_): maximum value of the axis
        relative_size (_type_): length of this axis divided by length of longest axis

    Returns:
        _type_: ticks
    """

    axis_length = abs(max_val - min_val)

    # Base step size on axis length
    if axis_length <= 5:
        base_step = 0.5
    elif axis_length <= 10:
        base_step = 1.0
    elif axis_length <= 20:
        base_step = 2.0
    elif axis_length <= 50:
        base_step = 5.0
    elif axis_length <= 100:
        base_step = 10.0
    else:  # <= 200
        base_step = 20.0

    # Adjust step size based on relative axis length
    # For shorter axes (relative to the longest), increase step size to prevent crowding
    if relative_size < 0.2:  # Very short axis
        step_size = base_step * 4
    elif relative_size < 0.5:  # Moderately short axis
        step_size = base_step * 2
    else:  # Normal or long axis
        step_size = base_step

    # Round min and max to step size
    min_tick = np.ceil(min_val / step_size) * step_size
    max_tick = np.floor(max_val / step_size) * step_size

    num_ticks = int((max_tick - min_tick) / step_size) + 1
    return np.linspace(min_tick, max_tick, num_ticks)


def set_axis_labels(ax, x_length, y_length, z_length):
    """
    Set axis labels with improved positioning for largely different axis lengths

    Args:
        ax (_type_): previous plotting environment
        x_length (_type_): length of x-direction array
        y_length (_type_): length of y-direction array
        z_length (_type_): length of z-direction array

    """
    # Calculate length ratios using log scale to handle large differences
    lengths = np.array([x_length, y_length, z_length])
    log_lengths = np.log10(lengths)
    max_log_length = np.max(log_lengths)

    # Calculate relative sizes on log scale
    x_relative = log_lengths[0] / max_log_length
    y_relative = log_lengths[1] / max_log_length
    z_relative = log_lengths[2] / max_log_length

    # Base padding value
    base_pad_x = x_length / 2.0
    base_pad_y = y_length / 2.0
    base_pad_z = z_length / 6.0

    # Calculate padding with exponential scaling for very different lengths
    def calculate_pad(relative_size, base_pad):
        if relative_size < 0.5:
            return base_pad * np.exp(2 * (1 - relative_size))
        return base_pad

    # Set labels with calculated padding
    ax.set_xlabel(r"$x$ [Mm]", labelpad=calculate_pad(x_relative, base_pad_x))
    ax.set_ylabel(r"$y$ [Mm]", labelpad=calculate_pad(y_relative, base_pad_y))
    ax.set_zlabel(r"$z$ [Mm]", labelpad=calculate_pad(z_relative, base_pad_z))

    # Adjust label rotations based on the relative sizes
    ax.xaxis.label.set_rotation(20)
    ax.yaxis.label.set_rotation(-20)
    ax.zaxis.label.set_rotation(0)


def _get_coordinates(data: Field3dData) -> Tuple:

    if data.flux_balance_state == FluxBalanceState.BALANCED:
        return data.x, data.y, data.z
    elif data.flux_balance_state == FluxBalanceState.UNBALANCED:
        xmax = data.x[-1]
        ymax = data.y[-1]
        x_big = np.arange(2.0 * data.nx) * 2.0 * xmax / (2.0 * data.nx - 1) - xmax
        y_big = np.arange(2.0 * data.ny) * 2.0 * ymax / (2.0 * data.ny - 1) - ymax
        return x_big, y_big, data.z


def _make_boxedges(data):

    # Limit fieldline plot to original data size (rather than Seehafer size)
    boxedges = np.zeros((2, 3))

    # # Y boundaries must come first, X second due to switched order explained above
    boxedges[0, 0] = data.y[0]
    boxedges[1, 0] = data.y[-1]
    boxedges[0, 1] = data.x[0]
    boxedges[1, 1] = data.x[-1]
    boxedges[0, 2] = data.z[0]
    boxedges[1, 2] = data.z[-1]

    return boxedges


def find_center(data: Field3dData) -> Tuple:
    """
    Find centres of poles on photospheric magentogram.
    """

    _, xmax, _, ymax, _, _ = (
        data.x[0],
        data.x[-1],
        data.y[0],
        data.y[-1],
        data.z[0],
        data.z[-1],
    )

    neighborhood_size = data.nx / 1.0
    threshold = 1.0

    data_max = maximum_filter(data.bz, neighborhood_size)  # mode ='reflect'
    maxima = data.bz == data_max
    data_min = minimum_filter(data.bz, neighborhood_size)
    minima = data.bz == data_min

    diff = (data_max - data_min) > threshold
    maxima[diff == 0] = 0
    minima[diff == 0] = 0

    labeled_sources, num_objects_sources = label(maxima)  # type: ignore
    slices_sources = find_objects(labeled_sources)
    x_sources, y_sources = [], []

    labeled_sinks, num_objects_sinks = label(minima)  # type: ignore
    slices_sinks = find_objects(labeled_sinks)
    x_sinks, y_sinks = [], []

    for dy, dx in slices_sources:
        x_center = (dx.start + dx.stop - 1) / 2
        x_sources.append(x_center / (data.nx / xmax))
        y_center = (dy.start + dy.stop - 1) / 2
        y_sources.append(y_center / (data.ny / ymax))

    for dy, dx in slices_sinks:
        x_center = (dx.start + dx.stop - 1) / 2
        x_sinks.append(x_center / (data.nx / xmax))
        y_center = (dy.start + dy.stop - 1) / 2
        y_sinks.append(y_center / (data.ny / ymax))

    return x_sources, y_sources, x_sinks, y_sinks
