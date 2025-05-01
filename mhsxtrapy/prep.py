from __future__ import annotations

import numpy as np


from sunpy.net import Fido, attrs as a
import sunpy.map

import astropy.units as u
from astropy.io import fits
from astropy.io.fits import getdata
from astropy.coordinates import SkyCoord

from matplotlib import rc, colors, ticker
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from matplotlib import cbook
from matplotlib.patches import Rectangle

rc("font", **{"family": "serif", "serif": ["Times"]})
rc("text", usetex=True)

c1 = (1.000, 0.224, 0.376)
c2 = (0.420, 0.502, 1.000)
c4 = (1.000, 0.224, 0.376)

cmap_magneto = colors.LinearSegmentedColormap.from_list(
    "cmap_magneto",
    (
        # Edit this gradient at https://eltos.github.io/gradient/#magnetogram=2D2D2D-D3D3D3
        (0.000, (0.176, 0.176, 0.176)),
        (1.000, (1.000, 1.000, 1.000)),
    ),
)
norm = colors.SymLogNorm(50, vmin=-7.5e2, vmax=7.5e2)


def find_corners_SDO(
    path: str, ulon: float, llon: float, ulat: float, llat: float
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
    hdr = hmi_image.fits_header

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
        cmap=cmap_magneto,
        norm=norm,
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
        norm=norm,
        cmap=cmap_magneto,
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
    sub_bz = bz[sty:lsty, stx:lstx]  # type: ignore

    x = np.arange(bz.shape[1])  # type: ignore
    y = np.arange(bz.shape[0])  # type: ignore

    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.contourf(x, y, bz, 1000, cmap=cmap_magneto, norm=norm)
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
