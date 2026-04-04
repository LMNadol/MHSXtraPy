from __future__ import annotations

import os

import numpy as np
from scipy.interpolate import griddata
from sunpy import config
from sunpy.map import Map

from mhsxtrapy._boundary import BoundaryData, FluxBalanceState, max_a_parameter
from mhsxtrapy._field import extrapolate
from mhsxtrapy.plotting import (
    plot_ddensity_xy,
    plot_ddensity_z,
    plot_dpressure_xy,
    plot_dpressure_z,
    plot_field_3d,
)
from mhsxtrapy.types import WhichSolution

config.set("downloads", "timeout", "300")  # 300 seconds

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def load_data():

    sharp_map = Map(
        os.path.join(_DATA_DIR, "hmi.sharp_cea_720s.377.20110215_000000_TAI.Br.fits")
    )
    ar_number = sharp_map.meta.get("HARPNUM")
    print(f"Active Region Number (HARPNUM): {ar_number}")
    print(f"NOAA Active Region Number (NOAA_AR): {sharp_map.meta.get('NOAA_AR')}")

    path_aia171 = os.path.join(
        _DATA_DIR, "aia.lev1.171A_2011_02_15T00_00_00.34Z.image_lev1.fits"
    )

    aia_image = Map(path_aia171).rotate()

    return sharp_map, aia_image


def preprocess(sharp_map, aia_image):

    left_corner = sharp_map.bottom_left_coord
    right_corner = sharp_map.top_right_coord
    aia_small = aia_image.submap(left_corner, top_right=right_corner)

    px_arcsec = (
        np.arctan(
            (
                sharp_map.fits_header["CDELT1"]
                * np.pi
                / 180
                * sharp_map.fits_header["RSUN_REF"]
            )
            / sharp_map.fits_header["DSUN_OBS"]
        )
        * 180
        / np.pi
        * 3600
    )
    py_arcsec = (
        np.arctan(
            (
                sharp_map.fits_header["CDELT2"]
                * np.pi
                / 180
                * sharp_map.fits_header["RSUN_REF"]
            )
            / sharp_map.fits_header["DSUN_OBS"]
        )
        * 180
        / np.pi
        * 3600
    )

    px_radians = px_arcsec / ((3600 * 360) / (2 * np.pi))
    py_radians = py_arcsec / ((3600 * 360) / (2 * np.pi))

    dist_Mm = sharp_map.fits_header["DSUN_OBS"] * 10**-6
    px = px_radians * dist_Mm
    py = py_radians * dist_Mm

    ny, nx = sharp_map.data.shape
    bz = sharp_map.data

    pz = max(px, py)
    nz = 120

    nf = int(min(nx, ny))
    x = np.arange(nx, dtype=np.float64) * px
    y = np.arange(ny, dtype=np.float64) * py
    z = np.arange(nz, dtype=np.float64) * pz

    sharp_data = BoundaryData(
        nx,
        ny,
        nz,
        nf,
        px,
        py,
        pz,
        x,
        y,
        z,
        bz,
        flux_balance_state=FluxBalanceState.BALANCED,
    )

    nx = aia_small.data.shape[1]
    ny = aia_small.data.shape[0]

    x = (
        np.arange(nx) * (sharp_data.x[-1] - sharp_data.x[0]) / (nx - 1)
        - sharp_data.x[0]
    )
    y = (
        np.arange(ny) * (sharp_data.y[-1] - sharp_data.y[0]) / (ny - 1)
        - sharp_data.y[0]
    )

    xv_fine, yv_fine = np.meshgrid(sharp_data.x, sharp_data.y)
    xv, yv = np.meshgrid(x, y)

    AIA_higherres = griddata(
        np.column_stack((yv.flatten(), xv.flatten())),
        aia_small.data.flatten(),
        np.column_stack((yv_fine.flatten(), xv_fine.flatten())),
        method="cubic",
    ).reshape(sharp_data.bz.shape)

    return sharp_data, AIA_higherres


if __name__ == "__main__":

    sharp_map, aia_image = load_data()
    sharp_data, AIA_higherres = preprocess(sharp_map, aia_image)

    max_a = max_a_parameter(sharp_data, alpha=0.01, b=1.0)

    sharp_extra = extrapolate(
        sharp_data,
        alpha=0.01,
        a=0.9 * max_a,
        which_solution=WhichSolution.NADOL_NEUKIRCH,
        b=1.0,
        z0=2.0,
        deltaz=0.2,
    )

    fig, ax = plot_field_3d(
        sharp_extra,
        view="side",
        footpoints="active-regions",
        pixel_stride_x=10,
        pixel_stride_y=10,
    )
    fig.savefig(
        os.path.join(os.path.dirname(__file__), "..", "figures", "field_3d.png"),
        dpi=300,
    )

    fig, ax = plot_dpressure_xy(sharp_extra, z=4.0)
    fig.savefig(
        os.path.join(os.path.dirname(__file__), "..", "figures", "dpressure_xy.png"),
        dpi=300,
    )
    fig, ax = plot_ddensity_xy(sharp_extra, z=4.0)
    fig.savefig(
        os.path.join(os.path.dirname(__file__), "..", "figures", "ddensity_xy.png"),
        dpi=300,
    )

    fig, ax = plot_dpressure_z(sharp_extra)
    fig.savefig(
        os.path.join(os.path.dirname(__file__), "..", "figures", "dpressure_z.png"),
        dpi=300,
    )
    fig, ax = plot_ddensity_z(sharp_extra)
    fig.savefig(
        os.path.join(os.path.dirname(__file__), "..", "figures", "ddensity_z.png"),
        dpi=300,
    )
