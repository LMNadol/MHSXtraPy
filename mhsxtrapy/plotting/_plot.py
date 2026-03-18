from __future__ import annotations

import os
from typing import Literal

import astropy.units as u
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import sunpy.map
from astropy.coordinates import SkyCoord
from astropy.io.fits import getdata
from matplotlib import rc, ticker
from matplotlib.patches import ConnectionPatch, Rectangle
from scipy.interpolate import interp1d

from mhsxtrapy._boundary import Field2dData
from mhsxtrapy._constants import (
    DEFAULT_N_LINES,
    DEFAULT_PIXEL_STRIDE,
    G_SOLAR,
    LATEX_ON,
    MU0,
    L,
)
from mhsxtrapy._field import Field3dData

from ._3d import plot_magnetogram_3D as plot_3D
from ._pp import plot_ddensity_xy as plot_dd
from ._pp import plot_dpressure_xy as plot_dp
from ._utils import cmap_magneto, detect_footpoints, find_center, norm_hmi

rc("font", **{"family": "serif", "serif": ["Times"]})
rc("text", usetex=LATEX_ON)


def plot_magnetogram_2D(data: Field2dData) -> None:
    """
    Plot photospheric boundary condition as basis for field line figures.

    Args:
        data (Field2dData): boundary condition data
    """

    fig = plt.figure()
    ax = fig.add_subplot(111)  # type: ignore
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
    n_lines_x: int | None = DEFAULT_N_LINES,
    n_lines_y: int | None = DEFAULT_N_LINES,
    pixel_stride_x: int | None = DEFAULT_PIXEL_STRIDE,
    pixel_stride_y: int | None = DEFAULT_PIXEL_STRIDE,
):
    """
    Wrapper function for 3D magentic field plotting.

        Args:
        data (Field3dData): magnetic field data
        view (Literal[&quot;los&quot;, &quot;side&quot;, &quot;angular&quot;]): which view should be displayed
        footpoints (Literal[&quot;all&quot;, &quot;active): which footpoints should be used
    """

    plot_3D(
        data,
        view,
        footpoints,
        boundary,
        n_lines_x,
        n_lines_y,
        pixel_stride_x,
        pixel_stride_y,
    )


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


def find_corners_SDO(
    path: str,
    ulon: float,
    llon: float,
    ulat: float,
    llat: float,
    cmap=cmap_magneto,
    norm=norm_hmi,
) -> None:
    """
    Plots SDO magnetogram and cut-out region defined through ulon, llon, ulat and llat.

    Args:
        path (str): path to .fits file
        ulon (float): upper longitude
        llon (float): lower longitude
        ulat (float): upper latitude
        llat (float): lower latitude
    """

    hmi_image = sunpy.map.Map(path).rotate()  # type: ignore
    # hdr = hmi_image.fits_header

    left_corner = SkyCoord(
        Tx=llon * u.arcsec, Ty=llat * u.arcsec, frame=hmi_image.coordinate_frame  # type: ignore
    )
    right_corner = SkyCoord(
        Tx=ulon * u.arcsec, Ty=ulat * u.arcsec, frame=hmi_image.coordinate_frame  # type: ignore
    )

    hpc_coords = sunpy.map.all_coordinates_from_map(hmi_image)
    mask = ~sunpy.map.coordinate_is_on_solar_disk(hpc_coords)
    magnetogram_big = sunpy.map.Map(hmi_image.data, hmi_image.meta, mask=mask)

    fig = plt.figure(figsize=(7.2, 4.8))

    ax1 = fig.add_subplot(121, projection=magnetogram_big)
    magnetogram_big.plot(  # type: ignore
        axes=ax1,
        cmap=cmap,
        norm=norm_hmi,
        annotate=False,
    )
    magnetogram_big.draw_grid(axes=ax1, color="white", alpha=0.25, lw=0.5)  # type: ignore
    ax1.grid(alpha=0)
    for coord in ax1.coords:  # type: ignore
        coord.frame.set_linewidth(0)
        coord.set_ticks_visible(False)
        coord.set_ticklabel_visible(False)

    magnetogram_big.draw_quadrangle(  # type: ignore
        left_corner, top_right=right_corner, edgecolor="black", lw=0.5
    )
    magnetogram_small = hmi_image.submap(left_corner, top_right=right_corner)
    ax2 = fig.add_subplot(122, projection=magnetogram_small)
    im = magnetogram_small.plot(
        axes=ax2,
        norm=norm_hmi,
        cmap=cmap,
        annotate=False,
    )
    ax2.grid(alpha=0)
    lon, lat = ax2.coords[0], ax2.coords[1]  # type: ignore
    lon.frame.set_linewidth(0.5)
    lat.frame.set_linewidth(0.5)
    lon.set_axislabel(
        "Helioprojective Longitude",
    )
    lon.set_ticks_position("b")
    lat.set_axislabel(
        "Helioprojective Latitude",
    )
    lat.set_axislabel_position("r")
    lat.set_ticks_position("r")
    lat.set_ticklabel_position("r")
    xpix, ypix = magnetogram_big.wcs.world_to_pixel(right_corner)  # type: ignore
    con1 = ConnectionPatch(
        (0, 1),
        (xpix, ypix),
        "axes fraction",
        "data",
        axesA=ax2,
        axesB=ax1,
        arrowstyle="-",
        color="black",
        lw=0.5,
    )
    xpix, ypix = magnetogram_big.wcs.world_to_pixel(  # type: ignore
        SkyCoord(
            right_corner.Tx, left_corner.Ty, frame=magnetogram_big.coordinate_frame  # type: ignore
        )
    )
    con2 = ConnectionPatch(
        (0, 0),
        (xpix, ypix),
        "axes fraction",
        "data",
        axesA=ax2,
        axesB=ax1,
        arrowstyle="-",
        color="black",
        lw=0.5,
    )
    ax2.add_artist(con1)
    ax2.add_artist(con2)
    ax2.tick_params(direction="in", length=2, width=0.5)

    pos = ax2.get_position().get_points()
    cax = fig.add_axes([pos[0, 0], pos[1, 1] + 0.01, pos[1, 0] - pos[0, 0], 0.025])  # type: ignore
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.locator = ticker.FixedLocator([-1e2, 0, 1e2])
    cbar.set_label("LOS Magnetic Field [gauss]", labelpad=-40, rotation=0)
    cbar.update_ticks()
    cbar.ax.xaxis.set_ticks_position("top")

    plotname = "figures/magnetogram-cutout.png"
    plt.savefig(plotname, dpi=300, bbox_inches="tight", pad_inches=0.5)

    plt.show()


def find_corners_SolarOrbiter(
    path: str, stx: float, lstx: float, sty: float, lsty: float
) -> None:
    """
    Plots Solar Orbiter magnetogram and cut-out region defined through ulon, llon, ulat and llat.

    Args:
        path (str): path to .fits file
        stx (float): start index x-direction
        lstx (float): last index x-direction
        sty (float): start index y-direction
        lsty (float): last index y-direction
    """

    bz = getdata(path, ext=False)

    x = np.arange(bz.shape[1])  # type: ignore
    y = np.arange(bz.shape[0])  # type: ignore

    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.contourf(x, y, bz, 1000, cmap=cmap_magneto, norm=norm_hmi)
    ax.set_xlabel(r"$x_i$")
    ax.set_ylabel(r"$y_i$")
    ax.set_box_aspect(bz.shape[0] / bz.shape[1])  # type: ignore

    currentAxis = plt.gca()
    currentAxis.add_patch(
        Rectangle(
            (stx, sty),
            lstx - stx,
            lsty - sty,
            fill=None,
            alpha=1,
            zorder=10,
            color="black",
            lw=1.0,
        )
    )

    plt.tick_params(direction="in", length=2, width=0.5)

    plt.show()
