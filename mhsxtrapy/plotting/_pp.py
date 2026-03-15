from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from mhsxtrapy.constants import G_SOLAR, MU0, L
from mhsxtrapy.field3d import Field3dData
from mhsxtrapy.types import FluxBalanceState

plt.rcParams["text.usetex"] = False
from ._core import _get_coordinates, cmap_density, cmap_pressure


def plot_dpressure_xy(data: Field3dData, z: np.float64) -> None:
    """
    Plots 2D pressure variation array at height z.

    Args:
        data (Field3dData): magnetic fiel data
        z (np.float64): vertical height
    """

    B0 = data.field[:, :, 0, 2].max()
    xmin, xmax, ymin, ymax, zmin, zmax = (
        data.x[0],
        data.x[-1],
        data.y[0],
        data.y[-1],
        data.z[0],
        data.z[-1],
    )
    # x_big = np.arange(2.0 * data.nx) * 2.0 * xmax / (2.0 * data.nx - 1) - xmax
    # y_big = np.arange(2.0 * data.ny) * 2.0 * ymax / (2.0 * data.ny - 1) - ymax

    x, y, z = _get_coordinates(data)
    x_grid, y_grid = np.meshgrid(x, y)

    if data.flux_balance_state == FluxBalanceState.UNBALANCED:
        x_grid = x_grid[data.ny : 2 * data.ny, data.nx : 2 * data.nx]
        y_grid = y_grid[data.ny : 2 * data.ny, data.nx : 2 * data.nx]

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
    """
    Plots 2D density variation array at height z.

    Args:
        data (Field3dData): magnetic fiel data
        z (np.float64): vertical height
    """

    B0 = data.field[:, :, 0, 2].max()
    xmin, xmax, ymin, ymax, zmin, zmax = (
        data.x[0],
        data.x[-1],
        data.y[0],
        data.y[-1],
        data.z[0],
        data.z[-1],
    )
    # x_big = np.arange(2.0 * data.nx) * 2.0 * xmax / (2.0 * data.nx - 1) - xmax
    # y_big = np.arange(2.0 * data.ny) * 2.0 * ymax / (2.0 * data.ny - 1) - ymax

    x, y, z = _get_coordinates(data)
    x_grid, y_grid = np.meshgrid(x, y)

    if data.flux_balance_state == FluxBalanceState.UNBALANCED:
        x_grid = x_grid[data.ny : 2 * data.ny, data.nx : 2 * data.nx]
        y_grid = y_grid[data.ny : 2 * data.ny, data.nx : 2 * data.nx]

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
