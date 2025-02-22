from __future__ import annotations

from typing import Literal, Tuple

import numpy as np

import math

from scipy.ndimage import maximum_filter, minimum_filter, label, find_objects

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rc, colors

from mhsxtrapy.field3d import Field3dData
from mhsxtrapy.field2d import Field2dData

from msat.pyvis.fieldline3d import fieldline3d

import os

rc("font", **{"family": "serif", "serif": ["Times"]})
rc("text", usetex=True)

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

MU0 = 1.25663706 * 10**-6
L = 10**6
G_SOLAR = 272.2


def plot_magnetogram_3D(
    data: Field3dData,
    view: Literal["los", "side", "angular"],
    footpoints: Literal["all", "active-regions"],
):
    """
    Create figure of magnetic field line from Field3dData object. Specify angle of view and optional zoom
    for the side view onto the transition region, which footpoints are chosen for field lines,
    if and where the figure is supposed to be saved.
    """

    xmin, xmax, ymin, ymax, zmin, zmax = (
        data.x[0],
        data.x[-1],
        data.y[0],
        data.y[-1],
        data.z[0],
        data.z[-1],
    )

    x_grid, y_grid = np.meshgrid(data.x, data.y)
    fig = plt.figure()
    ax = fig.figure.add_subplot(111, projection="3d")
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

    if footpoints == "all":
        plot_fieldlines_grid(data, ax)
    elif footpoints == "active-regions":
        sinks, sources = detect_footpoints(data)
        plot_fieldlines_AR(data, sinks, sources, ax)
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
        ax.view_init(0, -90)  # type: ignore
        ax.set_yticklabels([])  # type: ignore
        ax.set_ylabel("")

        [t.set_va("top") for t in ax.get_xticklabels()]  # type: ignore
        [t.set_ha("center") for t in ax.get_xticklabels()]  # type: ignore

        [t.set_va("center") for t in ax.get_zticklabels()]  # type: ignore
        [t.set_ha("center") for t in ax.get_zticklabels()]  # type: ignore
    elif view == "angular":
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

    # Construct the path using the view variable
    path = f"figures/magnetogram-3D_{view}.png"

    plt.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.2)

    plt.show()


def plot_fieldlines_grid(data: Field3dData, ax) -> None:
    """
    Plot field lines on grid.
    """

    xmin, xmax, ymin, ymax, zmin, zmax = (
        data.x[0],
        data.x[-1],
        data.y[0],
        data.y[-1],
        data.z[0],
        data.z[-1],
    )

    x_0 = 1.0e-8
    y_0 = 1.0e-8
    dx = xmax / 15.0
    dy = ymax / 15.0

    nlinesmaxx = math.floor(xmax / dx)
    nlinesmaxy = math.floor(ymax / dy)

    h1 = 1.0 / 100.0  # Initial step length for fieldline3D
    eps = 1.0e-8
    # Tolerance to which we require point on field line known for fieldline3D
    hmin = 0.0  # Minimum step length for fieldline3D
    hmax = 1.0  # Maximum step length for fieldline3D

    # Limit fieldline plot to original data size (rather than Seehafer size)
    boxedges = np.zeros((2, 3))

    # # Y boundaries must come first, X second due to switched order explained above
    boxedges[0, 0] = ymin
    boxedges[1, 0] = ymax
    boxedges[0, 1] = xmin
    boxedges[1, 1] = xmax
    boxedges[0, 2] = zmin
    boxedges[1, 2] = zmax

    for ilinesx in range(0, nlinesmaxx):
        for ilinesy in range(0, nlinesmaxy):
            x_start = x_0 + dx * ilinesx
            y_start = y_0 + dy * ilinesy

            if data.bz[int(y_start), int(x_start)] < 0.0:
                h1 = -h1

            ystart = [y_start, x_start, 0.0]
            # Fieldline3D expects startpt, BField, Row values, Column values so we need to give Y first, then X
            fieldline = fieldline3d(
                ystart,
                data.field,
                data.y,
                data.x,
                data.z,
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


def detect_footpoints(data: Field3dData) -> Tuple:
    """
    Detenct footpoints around centres of poles on photospheric magentogram.
    """

    sinks = data.bz.copy()
    sources = data.bz.copy()

    maxmask = sources < sources.max() * 0.4
    sources[maxmask != 0] = 0

    minmask = sinks < sinks.min() * 0.4
    sinks[minmask == 0] = 0

    return sinks, sources


def plot_fieldlines_AR(data: Field3dData, sinks: np.ndarray, sources: np.ndarray, ax):
    """
    Plot field lines starting at detected foot points around poles.
    """

    xmin, xmax, ymin, ymax, zmin, zmax = (
        data.x[0],
        data.x[-1],
        data.y[0],
        data.y[-1],
        data.z[0],
        data.z[-1],
    )

    h1 = 1.0 / 100.0  # Initial step length for fieldline3D
    eps = 1.0e-8
    # Tolerance to which we require point on field line known for fieldline3D
    hmin = 0.0  # Minimum step length for fieldline3D
    hmax = 1.0  # Maximum step length for fieldline3D

    # Limit fieldline plot to original data size (rather than Seehafer size)
    boxedges = np.zeros((2, 3))

    # # Y boundaries must come first, X second due to switched order explained above
    boxedges[0, 0] = ymin
    boxedges[1, 0] = ymax
    boxedges[0, 1] = xmin
    boxedges[1, 1] = xmax
    boxedges[0, 2] = zmin
    boxedges[1, 2] = zmax  # 2 * data.z0  # FOR ZOOM

    for ix in range(0, data.nx, int(data.nx / 20)):
        for iy in range(0, data.ny, int(data.ny / 20)):
            if sources[iy, ix] != 0 or sinks[iy, ix] != 0:

                x_start = ix / (data.nx / xmax)  # + 1.0e-8
                y_start = iy / (data.ny / ymax)  # + 1.0e-8
                # print(x_start, y_start)
                if data.bz[int(y_start), int(x_start)] < 0.0:
                    h1 = -h1

                ystart = [y_start, x_start, 0.0]

                fieldline = fieldline3d(
                    ystart,
                    data.field,
                    data.y,
                    data.x,
                    data.z,
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


def plot_dpressure_xy(data: Field3dData, z: np.float64) -> None:

    B0 = data.field[:, :, 0, 2].max()
    xmin, xmax, ymin, ymax, zmin, zmax = (
        data.x[0],
        data.x[-1],
        data.y[0],
        data.y[-1],
        data.z[0],
        data.z[-1],
    )

    x_grid, y_grid = np.meshgrid(data.x, data.y)

    index = np.abs(data.z - z).argmin()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    C = ax.contourf(
        x_grid,
        y_grid,
        abs(data.dpressure[:, :, index]) * (B0 * 10**-4) ** 2.0 / MU0,
        12,
        cmap=cmap_pressure,
        # vmin = data3d.dpressure[:, :, iiz].min(),
        # vmax = data3d.dpressure[:, :, iiz].max(),
    )
    ax.set_xlabel(r"$x$ [Mm]", size=14)
    ax.set_ylabel(r"$y$ [Mm]", size=14)
    ax.set_box_aspect(data.y[-1] / data.x[-1])
    ax.tick_params(direction="out", length=2, width=0.5)
    ax_position = ax.get_position()  # (left, bottom, width, height)
    # Calculate the fraction based on the axis width (can also use height if desired)
    fraction = ax_position.height * 0.045

    # Create colorbar and adjust its size dynamically based on the plot size
    cbar = fig.colorbar(
        C, ax=ax, fraction=fraction, pad=0.04
    )  # Adjust pad as necessary
    cbar.set_label(r"kg m$^{-1}$ s$^{-2}$", fontsize=14)
    plt.title(
        r"$|\Delta p|$ at $z = $ " + str(round(data.z[index], 2)) + " [Mm]", size=14
    )
    # Ensure the 'figures' directory exists
    if not os.path.exists("figures"):
        os.makedirs("figures")

    # Construct the dynamic plot name
    plotname = f"figures/dpressure_xy_z={str(round(data.z[index], 2))}.png"
    plt.savefig(plotname, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.show()


def plot_ddensity_xy(data: Field3dData, z: np.float64) -> None:

    B0 = data.field[:, :, 0, 2].max()
    xmin, xmax, ymin, ymax, zmin, zmax = (
        data.x[0],
        data.x[-1],
        data.y[0],
        data.y[-1],
        data.z[0],
        data.z[-1],
    )

    x_grid, y_grid = np.meshgrid(data.x, data.y)

    index = np.abs(data.z - z).argmin()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    C = ax.contourf(
        x_grid,
        y_grid,
        abs(data.ddensity[:, :, index]) * (B0 * 10**-4) ** 2.0 / (MU0 * G_SOLAR * L),
        12,
        cmap=cmap_density,
        # vmin = data3d.dpressure[:, :, iiz].min(),
        # vmax = data3d.dpressure[:, :, iiz].max(),
    )
    ax.set_xlabel(r"$x$ [Mm]", size=14)
    ax.set_ylabel(r"$y$ [Mm]", size=14)
    ax.set_box_aspect(data.y[-1] / data.x[-1])
    ax.tick_params(direction="out", length=2, width=0.5)
    ax_position = ax.get_position()  # (left, bottom, width, height)
    # Calculate the fraction based on the axis width (can also use height if desired)
    fraction = ax_position.height * 0.045

    # Create colorbar and adjust its size dynamically based on the plot size
    cbar = fig.colorbar(
        C, ax=ax, fraction=fraction, pad=0.04
    )  # Adjust pad as necessary
    cbar.set_label(r"kg m$^{-3}$", fontsize=14)
    plt.title(
        r"$|\Delta \rho|$ at $z =$" + str(round(data.z[index], 2)) + " [Mm]", size=14
    )
    # Ensure the 'figures' directory exists
    if not os.path.exists("figures"):
        os.makedirs("figures")

    # Construct the dynamic plot name
    plotname = f"figures/ddensity_xy_z={str(round(data.z[index], 2))}.png"
    plt.savefig(plotname, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.show()
