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

from mhsxtrapy._boundary import BoundaryData
from mhsxtrapy._field import ExtrapolationResult
from mhsxtrapy.constants import (
    DEFAULT_N_LINES,
    DEFAULT_PIXEL_STRIDE,
    G_SOLAR,
    LATEX_ON,
    MU0,
    L,
)
from mhsxtrapy.types import FluxBalanceState

from ._3d import plot_fieldlines_AR, plot_fieldlines_grid
from ._utils import (
    _detect_footpoints,
    _find_center,
    _get_coordinates,
    _set_axis_labels,
    cmap_aia,
    cmap_density,
    cmap_magneto,
    cmap_pressure,
    norm_aia,
    norm_hmi,
)

rc("font", **{"family": "serif", "serif": ["Times"]})
rc("text", usetex=LATEX_ON)

__all__ = []


def plot_magnetogram(data: BoundaryData) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot photospheric boundary condition as basis for field line figures.

    Args:
        data (BoundaryData): boundary condition data
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

    return fig, ax


def plot_dpressure_z(data: ExtrapolationResult) -> tuple[plt.Figure, plt.Axes]:
    """
    Plots vertical variation in pressure at x and y where Bz is maximal on the photosphere.

    Args:
        data (ExtrapolationResult): magnetic field data
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

    return fig, ax


def plot_ddensity_z(data: ExtrapolationResult) -> tuple[plt.Figure, plt.Axes]:
    """
    Plots vertical variation in density at x and y where Bz is maximal on the photosphere.

    Args:
        data (ExtrapolationResult): magnetic field data
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

    return fig, ax


def plot_field_3d(
    data: ExtrapolationResult,
    view: Literal["los", "side", "angular"],
    footpoints: Literal["all", "active-regions"],
    boundary: Literal["FeI-6173", "EUV"] = "FeI-6173",
    n_lines_x: int | None = DEFAULT_N_LINES,
    n_lines_y: int | None = DEFAULT_N_LINES,
    pixel_stride_x: int | None = DEFAULT_PIXEL_STRIDE,
    pixel_stride_y: int | None = DEFAULT_PIXEL_STRIDE,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Create figure of magnetic field line from ExtrapolationResult object. Specify angle of view and optional zoom
    for the side view onto the transition region, which footpoints are chosen for field lines.

    Args:
        data (ExtrapolationResult): magnetic field data
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
    _set_axis_labels(ax, x_length, y_length, z_length)

    if footpoints == "all":
        plot_fieldlines_grid(data, ax, n_lines_x, n_lines_y)
    elif footpoints == "active-regions":
        sinks, sources = _detect_footpoints(data)
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

    return fig, ax


def show_poles(data: ExtrapolationResult) -> None:
    """
    Show centres of poles on photospheric magnetogram.
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

    x_sources, y_sources, x_sinks, y_sinks = _find_center(data)

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


def show_footpoints(data: ExtrapolationResult) -> None:
    """
    Show footpoints around centres of poles on photospheric magentogram.
    """

    sinks, sources = _detect_footpoints(data)

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
    cmap: object = cmap_magneto,
    norm: object = norm_hmi,
) -> tuple[plt.Figure, plt.Axes, plt.Axes]:
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

    return fig, ax1, ax2


def find_corners_SolarOrbiter(
    path: str, stx: float, lstx: float, sty: float, lsty: float
) -> tuple[plt.Figure, plt.Axes]:
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

    return fig, ax


def plot_dpressure_xy(
    data: ExtrapolationResult, z: np.float64
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plots 2D pressure variation array at height z.

    Args:
        data (ExtrapolationResult): magnetic fiel data
        z (np.float64): vertical height
    """

    B0 = data.field[:, :, 0, 2].max()

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

    return fig, ax


def plot_ddensity_xy(
    data: ExtrapolationResult, z: np.float64
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plots 2D density variation array at height z.

    Args:
        data (ExtrapolationResult): magnetic fiel data
        z (np.float64): vertical height
    """

    B0 = data.field[:, :, 0, 2].max()

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

    return fig, ax
