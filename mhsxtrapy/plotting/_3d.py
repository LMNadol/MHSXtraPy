from __future__ import annotations

import math

import numpy as np
from matplotlib import rc

from mhsxtrapy._constants import LATEX_ON
from mhsxtrapy._field import ExtrapolationResult
from mhsxtrapy.plotting._fieldline3d import fieldline3d

from ._utils import _get_coordinates, _make_boxedges

rc("font", **{"family": "serif", "serif": ["Times"]})
rc("text", usetex=LATEX_ON)


def plot_fieldlines_grid(
    data: ExtrapolationResult, ax: object, n_lines_x: int, n_lines_y: int
) -> None:
    """
    Plot field lines on grid.

    Args:
        data (ExtrapolationResult): magnetic field data
        ax (_type_): previous plotting environment
    """

    _, xmax, _, ymax, _, _ = (
        data.x[0],
        data.x[-1],
        data.y[0],
        data.y[-1],
        data.z[0],
        data.z[-1],
    )

    x, y, z = _get_coordinates(data)

    x_0 = 1.0e-8
    y_0 = 1.0e-8
    dx = xmax / n_lines_x
    dy = ymax / n_lines_y

    nlinesmaxx = math.floor(xmax / dx)
    nlinesmaxy = math.floor(ymax / dy)

    # h1 = 1.0 / 100.0  # Initial step length for fieldline3D
    eps = 1.0e-8
    # Tolerance to which we require point on field line known for fieldline3D
    hmin = 0.0  # Minimum step length for fieldline3D
    hmax = 1.0  # Maximum step length for fieldline3D

    boxedges = _make_boxedges(data)

    for ilinesx in range(0, nlinesmaxx):
        for ilinesy in range(0, nlinesmaxy):
            x_start = x_0 + dx * ilinesx
            y_start = y_0 + dy * ilinesy

            # if data.bz[int(y_start), int(x_start)] < 0.0:
            # Convert physical coords to pixel indices for bz lookup
            ix_pixel = int(round(x_start / xmax * (data.nx - 1)))
            iy_pixel = int(round(y_start / ymax * (data.ny - 1)))
            ix_pixel = max(0, min(ix_pixel, data.nx - 1))
            iy_pixel = max(0, min(iy_pixel, data.ny - 1))

            h1 = 1.0 / 100.0  # Reset step length for each field line
            if data.bz[iy_pixel, ix_pixel] < 0.0:
                h1 = -h1

            ystart = [y_start, x_start, 0.0]
            # Fieldline3D expects startpt, BField, Row values, Column values so we need to give Y first, then X
            fieldline = fieldline3d(
                ystart,
                data.field,
                y,
                x,
                z,
                h1,
                hmin,
                hmax,
                eps,
                oneway=False,
                boxedge=boxedges,
                gridcoord=False,
                coordsystem="cartesian",
            )  # , periodicity='xy')

            if np.isclose(fieldline[:, 2][-1], 0.0) and np.isclose(
                fieldline[:, 2][0], 0.0
            ):
                # Need to give row direction first/ Y, then column direction/ X
                ax.plot(
                    fieldline[:, 1],
                    fieldline[:, 0],
                    fieldline[:, 2],
                    color="black",
                    linewidth=0.5,
                    zorder=4000,
                )
            else:
                ax.plot(
                    fieldline[:, 1],
                    fieldline[:, 0],
                    fieldline[:, 2],
                    color="black",
                    linewidth=0.5,
                    zorder=4000,
                )


def plot_fieldlines_AR(
    data: ExtrapolationResult,
    sinks: np.ndarray,
    sources: np.ndarray,
    ax: object,
    pixel_stride_x: int,
    pixel_stride_y: int,
) -> None:
    """
    Plot field lines starting at detected foot points around poles.

    Args:
        data (ExtrapolationResult): magnetic field data
        sinks (np.ndarray): regions of negative flux
        sources (np.ndarray): regions of positive flux
        ax (_type_): previous plotting environment
    """

    _, xmax, _, ymax, _, _ = (
        data.x[0],
        data.x[-1],
        data.y[0],
        data.y[-1],
        data.z[0],
        data.z[-1],
    )

    x, y, z = _get_coordinates(data)

    # h1 = 1.0 / 100.0  # Initial step length for fieldline3D
    eps = 1.0e-8
    # Tolerance to which we require point on field line known for fieldline3D
    hmin = 0.0  # Minimum step length for fieldline3D
    hmax = 1.0  # Maximum step length for fieldline3D

    boxedges = _make_boxedges(data)

    for ix in range(0, data.nx, pixel_stride_x):
        for iy in range(0, data.ny, pixel_stride_y):
            if sources[iy, ix] != 0 or sinks[iy, ix] != 0:

                x_start = ix / (data.nx / xmax)  # + 1.0e-8
                y_start = iy / (data.ny / ymax)  # + 1.0e-8

                # Use pixel indices (iy, ix) for bz lookup, not physical coords
                h1 = 1.0 / 100.0  # Reset step length for each field line
                if data.bz[iy, ix] < 0.0:
                    h1 = -h1

                ystart = [y_start, x_start, 0.0]

                fieldline = fieldline3d(
                    ystart,
                    data.field,
                    y,
                    x,
                    z,
                    h1,
                    hmin,
                    hmax,
                    eps,
                    oneway=False,
                    boxedge=boxedges,
                    gridcoord=False,
                    coordsystem="cartesian",
                )  # , periodicity='xy')

                if np.isclose(fieldline[:, 2][-1], 0.0) and np.isclose(
                    fieldline[:, 2][0], 0.0
                ):
                    # Need to give row direction first/ Y, then column direction/ X
                    ax.plot(
                        fieldline[:, 1],
                        fieldline[:, 0],
                        fieldline[:, 2],
                        color="black",
                        linewidth=0.5,
                        zorder=4000,
                    )

                else:
                    ax.plot(
                        fieldline[:, 1],
                        fieldline[:, 0],
                        fieldline[:, 2],
                        color="black",
                        linewidth=0.5,
                        zorder=4000,
                    )
