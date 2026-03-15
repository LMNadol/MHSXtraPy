from __future__ import annotations

import os
from typing import Literal, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
from scipy.interpolate import interp1d
from scipy.ndimage import find_objects, label, maximum_filter, minimum_filter

from mhsxtrapy.constants import G_SOLAR, MU0, L
from mhsxtrapy.field2d import Field2dData
from mhsxtrapy.field3d import Field3dData

from ._3d import plot_magnetogram_3D as plot_3D
from ._core import cmap_magneto, detect_footpoints
from ._pp import plot_ddensity_xy as plot_dd
from ._pp import plot_dpressure_xy as plot_dp

rc("font", **{"family": "serif", "serif": ["Times"]})
rc("text", usetex=True)
plt.rcParams["text.usetex"] = False

__all__ = [
    "plot_magnetogram_2D",
    "plot_dpressure_z",
    "plot_ddensity_z",
    "plot_magnetogram_3D",
    "plot_dpressure_xy",
    "plot_ddensity_xy",
    "find_center",
    "show_poles",
    "detect_footpoints",
    "show_footpoints",
]


def plot_magnetogram_2D(data: Field2dData) -> None:
    """
    Plot photospheric boundary condition as basis for field line figures.

    Args:
        data (Field2dData): boundary condition data
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
    """
    Plots vertical variation in pressure at x and y where Bz is maximal on the photosphere.

    Args:
        data (Field3dData): magnetic field data
    """

    B0 = data.field[:, :, 0, 2].max()
    min_values = np.min(data.dpressure, axis=(0, 1))
    ix_max = np.unravel_index(data.bz.argmax(), data.bz.shape)[1]
    iy_max = np.unravel_index(data.bz.argmax(), data.bz.shape)[0]

    z_fine = np.linspace(data.z[0], data.z[-1], 1500)
    interp_func = interp1d(
        data.z,
        data.dpressure[iy_max, ix_max, :],
        kind="cubic",
        fill_value="extrapolate",
    )
    data_fine = interp_func(z_fine)

    interp_func2 = interp1d(data.z, min_values, kind="cubic", fill_value="extrapolate")
    data_fine2 = interp_func2(z_fine)

    fig, ax = plt.subplots()

    ax.plot(
        z_fine,
        data_fine * (B0 * 10**-4) ** 2.0 / MU0,
        linewidth=1.5,
        color="black",
        label=r"at maximal $B_z(z=0)$",
    )

    ax.plot(
        z_fine,
        data_fine2 * (B0 * 10**-4) ** 2.0 / MU0,
        linewidth=1.5,
        color=(0.498, 0.502, 0.973),
        label=r"minimum per $z$",
    )

    ax.set_xlabel(r"$z$ [Mm]", size=14)
    plt.xlim([0, 2.0 * data.z0])  # type: ignore
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
    """
    Plots vertical variation in density at x and y where Bz is maximal on the photosphere.

    Args:
        data (Field3dData): magnetic field data
    """

    B0 = data.field[:, :, 0, 2].max()
    min_values = np.min(data.ddensity, axis=(0, 1))
    ix_max = np.unravel_index(data.bz.argmax(), data.bz.shape)[1]
    iy_max = np.unravel_index(data.bz.argmax(), data.bz.shape)[0]

    z_fine = np.linspace(data.z[0], data.z[-1], 1500)
    interp_func = interp1d(
        data.z, data.ddensity[iy_max, ix_max, :], kind="cubic", fill_value="extrapolate"
    )
    data_fine = interp_func(z_fine)

    interp_func2 = interp1d(data.z, min_values, kind="cubic", fill_value="extrapolate")
    data_fine2 = interp_func2(z_fine)

    fig, ax = plt.subplots()

    ax.plot(
        z_fine,
        data_fine * (B0 * 10**-4) ** 2.0 / (MU0 * G_SOLAR * L),
        linewidth=1.5,
        color="black",
        label=r"at maximal $B_z(z=0)$",
    )

    ax.plot(
        z_fine,
        data_fine2 * (B0 * 10**-4) ** 2.0 / (MU0 * G_SOLAR * L),
        linewidth=1.5,
        color=(0.827, 0.369, 0.494),
        label=r"minimum per $z$",
    )

    ax.set_xlabel(r"$z$ [Mm]", size=14)
    plt.xlim([0, 2.0 * data.z0])  # type: ignore
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
    boundary: Literal["FeI-6173", "EUV"] = "FeI-6173",
):
    """
    Wrapper function for 3D magentic field plotting.

        Args:
        data (Field3dData): magnetic field data
        view (Literal[&quot;los&quot;, &quot;side&quot;, &quot;angular&quot;]): which view should be displayed
        footpoints (Literal[&quot;all&quot;, &quot;active): which footpoints should be used
    """

    plot_3D(data, view, footpoints, boundary)


def plot_dpressure_xy(data: Field3dData, z: np.float64) -> None:
    """
    Wrapper function for 2D pressure variation plotting.

    Args:
        data (Field3dData): magnetic field data
        z (np.float64): height z at which variation is plotted
    """

    plot_dp(data, z)


def plot_ddensity_xy(data: Field3dData, z: np.float64) -> None:
    """
    Wrapper function for 2D density variation plotting.

    Args:
        data (Field3dData): magnetic field data
        z (np.float64): height z at which variation is plotted
    """

    plot_dd(data, z)


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


def show_poles(data: Field3dData):
    """
    Show centres of poles on photospheric magentogram.
    """

    x_plot = np.outer(data.y, np.ones(data.nx))
    y_plot = np.outer(data.x, np.ones(data.ny)).T

    _, xmax, _, ymax, _, _ = (
        data.x[0],
        data.x[-1],
        data.y[0],
        data.y[-1],
        data.z[0],
        data.z[-1],
    )

    x_sources, y_sources, x_sinks, y_sinks = find_center(data)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.grid(color="white", linestyle="dotted", linewidth=0.5)
    ax.contourf(
        y_plot, x_plot, data.bz, 1000, cmap=cmap_magneto
    )  # , norm=norm) # type: ignore
    ax.set_xlabel(r"$x/L$")
    ax.set_ylabel(r"$y/L$")
    plt.tick_params(direction="in", length=2, width=0.5)
    ax.set_box_aspect(ymax / xmax)

    for i in range(0, len(x_sinks)):

        xx = x_sinks[i]
        yy = y_sinks[i]
        ax.scatter(xx, yy, marker="x", color="yellow")

    for i in range(0, len(x_sources)):

        xx = x_sources[i]
        yy = y_sources[i]
        ax.scatter(xx, yy, marker="x", color="blue")

    sinks_label = mpatches.Patch(
        color="yellow",
        label="Sinks",
    )
    sources_label = mpatches.Patch(color="blue", label="Sources")

    plt.legend(handles=[sinks_label, sources_label], frameon=False, fontsize=12)
    plt.show()


def show_footpoints(data: Field3dData) -> None:
    """
    Show footpoints around centres of poles on photospheric magentogram.
    """

    sinks, sources = detect_footpoints(data)

    _, xmax, _, ymax, _, _ = (
        data.x[0],
        data.x[-1],
        data.y[0],
        data.y[-1],
        data.z[0],
        data.z[-1],
    )

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.contourf(
        np.outer(data.x, np.ones(data.ny)).T,
        np.outer(data.y, np.ones(data.nx)),
        data.bz,
        1000,
        cmap=cmap_magneto,
        # norm=norm,
    )
    ax.set_xlabel(r"$x/L$")
    ax.set_ylabel(r"$y/L$")
    plt.tick_params(direction="in", length=2, width=0.5)
    ax.set_box_aspect(ymax / xmax)

    for ix in range(0, data.nx, int(data.nx / 25)):
        for iy in range(0, data.ny, int(data.ny / 25)):
            if sources[iy, ix] != 0:
                ax.scatter(
                    ix / (data.nx / xmax),
                    iy / (data.ny / ymax),
                    color="blue",
                    s=2.0,
                )

            if sinks[iy, ix] != 0:
                ax.scatter(
                    ix / (data.nx / xmax),
                    iy / (data.ny / ymax),
                    color="yellow",
                    s=2.0,
                )

    sinks_label = mpatches.Patch(color="yellow", label="Sinks")
    sources_label = mpatches.Patch(color="blue", label="Sources")

    plt.legend(handles=[sinks_label, sources_label], frameon=False, fontsize=12)
    plt.show()
