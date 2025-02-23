from __future__ import annotations

import math
import os
from typing import Literal, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors, rc

from mhsxtrapy.field2d import Field2dData
from mhsxtrapy.field3d import Field3dData, FluxBalanceState
from mhsxtrapy.msat.pyvis.fieldline3d import fieldline3d
from mhsxtrapy.plotting.plot_unbalanced import (
    plot_ddensity_xy as plot_ddensity_xy_unbalanced,
)
from mhsxtrapy.plotting.plot_unbalanced import (
    plot_dpressure_xy as plot_dpressure_xy_unbalanced,
)
from mhsxtrapy.plotting.plot_unbalanced import (
    plot_magnetogram_3D as plot_magnetogram_3D_unbalanced,
)
from mhsxtrapy.plotting.plot_balanced import (
    plot_ddensity_xy as plot_ddensity_xy_balanced,
)
from mhsxtrapy.plotting.plot_balanced import (
    plot_dpressure_xy as plot_dpressure_xy_balanced,
)
from mhsxtrapy.plotting.plot_balanced import (
    plot_magnetogram_3D as plot_magnetogram_3D_balanced,
)

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


def plot_magnetogram_2D(data: Field2dData) -> None:
    """
    Plot photospheric boundary condition as basis for field line figures.
    """

    fig = plt.figure()
    ax = fig.figure.add_subplot(111)  # type: ignore
    x_grid, y_grid = np.meshgrid(data.x, data.y)
    C = ax.contourf(
        x_grid,
        y_grid,
        data.bz,
        20,
        cmap=cmap_magneto,
        # offset=0.0,
    )
    ax.contour(
        x_grid,
        y_grid,
        data.bz,
        levels=20,
        linewidths=0.1,
        linestyles="solid",
        colors="white",
    )
    ax.set_xlabel(r"$x$ [Mm]", size=14)
    ax.set_ylabel(r"$y$ [Mm]", size=14)
    plt.tick_params(direction="out", length=2, width=0.5)
    # cbar = fig.colorbar(C)
    ax.set_box_aspect(data.y[-1] / data.x[-1])
    # Get the axis position (in figure coordinates)
    ax_position = ax.get_position()  # (left, bottom, width, height)
    # Calculate the fraction based on the axis width (can also use height if desired)
    fraction = ax_position.height * 0.045

    # Create colorbar and adjust its size dynamically based on the plot size
    cbar = fig.colorbar(
        C, ax=ax, fraction=fraction, pad=0.04
    )  # Adjust pad as necessary
    cbar.set_label(r"Gauss", fontsize=14)

    # Ensure the 'figures' directory exists
    if not os.path.exists("figures"):
        os.makedirs("figures")

    plotname = "figures/magnetogram-2D.png"
    plt.savefig(plotname, dpi=300, bbox_inches="tight", pad_inches=0.5)
    plt.show()


def plot_dpressure_z(data: Field3dData) -> None:

    B0 = data.field[:, :, 0, 2].max()
    min_values = np.min(data.dpressure, axis=(0, 1))
    ix_max = np.unravel_index(data.bz.argmax(), data.bz.shape)[1]
    iy_max = np.unravel_index(data.bz.argmax(), data.bz.shape)[0]

    fig, ax = plt.subplots()

    ax.plot(
        data.z,
        data.dpressure[iy_max, ix_max, :] * (B0 * 10**-4) ** 2.0 / MU0,
        linewidth=1.5,
        color="black",
        label=r"at maximal $B_z(z=0)$",
    )

    ax.plot(
        data.z,
        min_values * (B0 * 10**-4) ** 2.0 / MU0,
        linewidth=1.5,
        color=(0.498, 0.502, 0.973),
        label=r"minimum per $z$",
    )

    ax.set_xlabel(r"$z$ [Mm]", size=14)
    plt.xlim([0, data.z0])
    plt.legend(frameon=False)
    ax.tick_params(direction="out", length=2, width=0.5)  # type: ignore

    ax.set_ylabel(r"$\Delta p$ [kg m$^{-1}$ s$^{-2}$]", size=14)

    # Ensure the 'figures' directory exists
    if not os.path.exists("figures"):
        os.makedirs("figures")

    plotname = "figures/dpressure_z.png"
    plt.savefig(plotname, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.show()


def plot_ddensity_z(data: Field3dData) -> None:

    B0 = data.field[:, :, 0, 2].max()
    min_values = np.min(data.ddensity, axis=(0, 1))
    ix_max = np.unravel_index(data.bz.argmax(), data.bz.shape)[1]
    iy_max = np.unravel_index(data.bz.argmax(), data.bz.shape)[0]

    fig, ax = plt.subplots()

    ax.plot(
        data.z,
        data.ddensity[iy_max, ix_max, :] * (B0 * 10**-4) ** 2.0 / (MU0 * G_SOLAR * L),
        linewidth=1.5,
        color="black",
        label=r"at maximal $B_z(z=0)$",
    )

    ax.plot(
        data.z,
        min_values * (B0 * 10**-4) ** 2.0 / (MU0 * G_SOLAR * L),
        linewidth=1.5,
        color=(0.827, 0.369, 0.494),
        label=r"minimum per $z$",
    )

    ax.set_xlabel(r"$z$ [Mm]", size=14)
    plt.xlim([0, data.z0])
    plt.legend(frameon=False)
    ax.tick_params(direction="out", length=2, width=0.5)

    ax.set_ylabel(r"$\Delta \rho$ [kg m$^{-3}$]", size=14)

    # Ensure the 'figures' directory exists
    if not os.path.exists("figures"):
        os.makedirs("figures")

    plotname = "figures/ddensity_z.png"
    plt.savefig(plotname, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.show()


def plot_magnetogram_3D(
    data: Field3dData,
    view: Literal["los", "side", "angular"],
    footpoints: Literal["all", "active-regions"],
):
    if data.flux_balance_state == FluxBalanceState.BALANCED:

        plot_magnetogram_3D_balanced(data, view, footpoints)

    elif data.flux_balance_state == FluxBalanceState.UNBALANCED:

        plot_magnetogram_3D_unbalanced(data, view, footpoints)


def plot_dpressure_xy(data: Field3dData, z: np.float64) -> None:
    if data.flux_balance_state == FluxBalanceState.BALANCED:

        plot_dpressure_xy_balanced(data, z)

    elif data.flux_balance_state == FluxBalanceState.UNBALANCED:

        plot_dpressure_xy_unbalanced(data, z)


def plot_ddensity_xy(data: Field3dData, z: np.float64) -> None:
    if data.flux_balance_state == FluxBalanceState.BALANCED:

        plot_ddensity_xy_balanced(data, z)

    elif data.flux_balance_state == FluxBalanceState.UNBALANCED:

        plot_ddensity_xy_unbalanced(data, z)
