from __future__ import annotations

import math
import os
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc

from mhsxtrapy.constants import DEFAULT_N_LINES, DEFAULT_PIXEL_STRIDE, LATEX_ON
from mhsxtrapy.field3d import Field3dData
from mhsxtrapy.plotting.fieldline3d import fieldline3d
from mhsxtrapy.types import FluxBalanceState

from ._core import (
    _get_coordinates,
    _make_boxedges,
    cmap_aia,
    cmap_magneto,
    detect_footpoints,
    norm_aia,
    set_axis_labels,
)

rc("font", **{"family": "serif", "serif": ["Times"]})
rc("text", usetex=LATEX_ON)


def plot_magnetogram_3D(
    data: Field3dData,
    view: Literal["los", "side", "angular"],
    footpoints: Literal["all", "active-regions"],
    boundary: Literal["FeI-6173", "EUV"] = "FeI-6173",
    n_lines_x: int | None = DEFAULT_N_LINES,
    n_lines_y: int | None = DEFAULT_N_LINES,
    pixel_stride_x: int | None = DEFAULT_PIXEL_STRIDE,
    pixel_stride_y: int | None = DEFAULT_PIXEL_STRIDE,
):
    """
    Create figure of magnetic field line from Field3dData object. Specify angle of view and optional zoom
    for the side view onto the transition region, which footpoints are chosen for field lines.

    Args:
        data (Field3dData): magnetic field data
        view (Literal[&quot;los&quot;, &quot;side&quot;, &quot;angular&quot;]): which view should be displayed
        footpoints (Literal[&quot;all&quot;, &quot;active): which footpoints should be used

    Raises:
        ValueError: In case view value is wrong
        ValueError: In case footpoints value is wrong
    """

    if boundary == "EUV":
        if data.EUV is None:
            raise ValueError("EUV selected as boundary, but no EUV image provided.")

    xmin, xmax, ymin, ymax, zmin, zmax = (
        data.x[0],
        data.x[-1],
        data.y[0],
        data.y[-1],
        data.z[0],
        data.z[-1],
    )

    x, y, z = _get_coordinates(data)
    x_grid, y_grid = np.meshgrid(x, y)

    if data.flux_balance_state == FluxBalanceState.UNBALANCED:
        x_grid = x_grid[data.ny : 2 * data.ny, data.nx : 2 * data.nx]
        y_grid = y_grid[data.ny : 2 * data.ny, data.nx : 2 * data.nx]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    if boundary == "EUV":
        ax.contourf(
            x_grid,
            y_grid,
            data.EUV,
            1000,
            cmap=cmap_aia,
            norm=norm_aia,
            offset=0.0,
        )
    else:
        ax.contourf(
            x_grid,
            y_grid,
            data.bz,
            20,
            cmap=cmap_magneto,
            offset=0.0,
        )

    ax.set_xlabel(r"$x$ [Mm]", size=14)
    ax.set_ylabel(r"$y$ [Mm]", size=14)
    ax.set_zlabel(r"$z$ [Mm]", size=14)  # type: ignore
    ax.grid(False)
    ax.set_zlim(zmin, zmax)  # type: ignore
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_box_aspect((xmax, ymax, zmax))  # type: ignore

    x_length = abs(xmax - xmin)
    y_length = abs(ymax - ymin)
    z_length = abs(zmax - zmin)
    # max_length = max(x_length, y_length, z_length)

    # Set axis labels with dynamic positioning
    set_axis_labels(ax, x_length, y_length, z_length)

    if footpoints == "all":
        plot_fieldlines_grid(data, ax, n_lines_x, n_lines_y)
    elif footpoints == "active-regions":
        sinks, sources = detect_footpoints(data)
        plot_fieldlines_AR(data, sinks, sources, ax, pixel_stride_x, pixel_stride_y)
    else:
        raise ValueError(
            f"Invalid footpoints option: {footpoints}. Choose from 'all' or 'active-regions'."
        )

    if view == "los":
        ax.view_init(90, -90)  # type: ignore
        ax.set_zticklabels([])  # type: ignore
        ax.set_zlabel("")  # type: ignore

        [t.set_va("center") for t in ax.get_yticklabels()]  # type: ignore
        [t.set_ha("center") for t in ax.get_yticklabels()]  # type: ignore

        [t.set_va("center") for t in ax.get_xticklabels()]  # type: ignore
        [t.set_ha("center") for t in ax.get_xticklabels()]  # type: ignore
    elif view == "side":
        ax.set_zticks(np.linspace(zmin, zmax, 5))
        ax.view_init(0, -90)  # type: ignore
        ax.set_yticklabels([])  # type: ignore
        ax.set_ylabel("")

        [t.set_va("top") for t in ax.get_xticklabels()]  # type: ignore
        [t.set_ha("center") for t in ax.get_xticklabels()]  # type: ignore

        [t.set_va("top") for t in ax.get_zticklabels()]  # type: ignore
        [t.set_ha("center") for t in ax.get_zticklabels()]  # type: ignore

    elif view == "angular":
        ax.set_zticks(np.linspace(zmin, zmax, 5))
        ax.view_init(30, 240, 0)  # type: ignore

        [t.set_va("bottom") for t in ax.get_yticklabels()]  # type: ignore
        [t.set_ha("right") for t in ax.get_yticklabels()]  # type: ignore

        [t.set_va("bottom") for t in ax.get_xticklabels()]  # type: ignore
        [t.set_ha("left") for t in ax.get_xticklabels()]  # type: ignore

        [t.set_va("top") for t in ax.get_zticklabels()]  # type: ignore
        [t.set_ha("center") for t in ax.get_zticklabels()]  # type: ignore
    else:
        raise ValueError(
            f"Invalid view option: {view}. Choose from 'los', 'side', or 'angular'."
        )

    # Ensure the 'figures' directory exists
    if not os.path.exists("figures"):
        os.makedirs("figures")
    # plt.colorbar(C)
    # Construct the path using the view variable
    path = f"figures/magnetogram-3D_{view}.png"

    plt.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.4)

    plt.show()


def plot_fieldlines_grid(data: Field3dData, ax, n_lines_x, n_lines_y) -> None:
    """
    Plot field lines on grid.

    Args:
        data (Field3dData): magnetic field data
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
    data: Field3dData,
    sinks: np.ndarray,
    sources: np.ndarray,
    ax,
    pixel_stride_x,
    pixel_stride_y,
):
    """
    Plot field lines starting at detected foot points around poles.

    Args:
        data (Field3dData): magnetic field data
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
