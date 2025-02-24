from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import sunpy.map
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io.fits import getdata
from astropy.io.fits import open as astroopen


class FluxBalanceState(Enum):
    BALANCED = "Balanced"
    UNBALANCED = "Unbalanced"


@dataclass
class Field2dData:
    """
    Dataclass of type Field2dData with the following attributes:
    nx, ny, nz  :   Dimensions of 3D magnetic field, usually nx and ny determined by magnetogram size,
                    while nz defined by user through height to which extrapolation is carried out.
    nf          :   Number of Fourier modes used in calculation of magnetic field vector, usually
                    nf = min(nx, ny) is taken. To do: split into nfx, nfy, sucht that all possible modes
                    in both directions can be used.
    px, py, pz  :   Pixel sizes in x-, y-, z-direction, in normal length scale (Mm).
    x, y, z     :   1D arrays of grid points on which magnetic field is given with shapes (nx,), (ny,)
                    and (nz,) respectively.
    bz          :   Bottom boundary magentogram of size (ny, nx,). Indexing of vectors done in this order,
                    such that, following intuition, x-direction corresponds to latitudinal extension and
                    y-direction to longitudinal extension of the magnetic field.

    Returns:
        _type_: Field2dData object
    """

    nx: int
    ny: int
    nz: int
    nf: int
    px: float
    py: float
    pz: float
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    bz: np.ndarray

    flux_balance_state: FluxBalanceState

    @classmethod
    def from_fits_SolarOrbiter(
        cls, path: str, stx: int, lstx: int, sty: int, lsty: int
    ):
        """
        Creates dataclass of type Field2dData from SolarOrbiter Archive data in .fits format.
        Only needs to be handed path to file and the indices one wants to use to cut the region to size,
        then creates Field2dData for extrapolation to 20 Mm.

        Steps:
        (1)     From the file given at path read in the image and header data refarding the distance to
                the sun, pixel unit and pixel size in arcsec.
        (2)     Cut image to specific size [sty:lsty, stx:lstx] around feature under investigation.
        (3)     Determine nx, ny, nf, px, py, xmax, ymax from data.
        (4)     Choose nz, pz and zmax.
        (5)     Determine x, y, z.
        (6)     Write all into Field2dData object.

        Args:
            path (str): path to .fits file
            stx (int): start index x-direction
            lstx (int): last index x-direction
            sty (int): start index y-direction
            lsty (int): last index y-direction

        Returns:
            _type_: Field2dData object
        """

        with astroopen(path) as data:  # type: ignore

            image = getdata(path, ext=False)

            hdr = data[0].header  # type: ignore
            dist = hdr["DSUN_OBS"]
            px_unit = hdr["CUNIT1"]
            py_unit = hdr["CUNIT2"]
            px_arcsec = hdr["CDELT1"]
            py_arcsec = hdr["CDELT2"]

        image_cut = image[sty:lsty, stx:lstx]  # type: ignore

        nx = image_cut.shape[1]
        ny = image_cut.shape[0]

        nf = min(nx, ny)

        px_radians = px_arcsec / 206265.0
        py_radians = py_arcsec / 206265.0

        dist_Mm = dist * 10**-6
        px = px_radians * dist_Mm
        py = py_radians * dist_Mm

        xmin = 0.0
        ymin = 0.0
        zmin = 0.0

        xmax = nx * px
        ymax = ny * py
        zmax = 20.0

        # pz = np.float64(90.0 * 10**-3)
        pz = max(px, py)

        nz = int(np.floor(zmax / pz))

        x = np.arange(nx) * (xmax - xmin) / (nx - 1) - xmin
        y = np.arange(ny) * (ymax - ymin) / (ny - 1) - ymin
        z = np.arange(nz) * (zmax - zmin) / (nz - 1) - zmin

        if np.fabs(check_fluxbalance(image_cut)) < 0.01:
            return Field2dData(
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
                image_cut,
                flux_balance_state=FluxBalanceState.BALANCED,
            )
        else:
            return Field2dData(
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
                image_cut,
                flux_balance_state=FluxBalanceState.UNBALANCED,
            )

    @classmethod
    def from_fits_SDO(
        cls, path: str, ulon: float, llon: float, ulat: float, llat: float
    ):
        """
        Creates dataclass of type Field2dData from SDO HMI data in .fits format.
        Only needs to be handed path to file and longitudes and latitutes of desired cut out,
        then creates Field2dData for extrapolation to 20 Mm.

        Steps:
        (1)     From the file given at path read in the image and header data refarding the distance to
                the sun, pixel unit and pixel size in arcsec.
        (2)     Cut image to specific size round feature under investigation.
        (3)     Determine nx, ny, nf, px, py, xmax, ymax from data.
        (4)     Choose nz, pz and zmax.
        (5)     Determine x, y, z.
        (6)     Write all into Field2dData object.

        Args:
            path (str): path to .fits file
            ulon (float): upper longitude
            llon (float): lower longitude
            ulat (float): upper latitute
            llat (float): lower latitute

        Returns:
            _type_: _description_
        """

        hmi_image = sunpy.map.Map(path).rotate()  # type: ignore

        hdr = hmi_image.fits_header

        left_corner = SkyCoord(
            Tx=llon * u.arcsec,  # type: ignore
            Ty=llat * u.arcsec,  # type: ignore
            frame=hmi_image.coordinate_frame,
        )
        right_corner = SkyCoord(
            Tx=ulon * u.arcsec,  # type: ignore
            Ty=ulat * u.arcsec,  # type: ignore
            frame=hmi_image.coordinate_frame,
        )

        image = hmi_image.submap(left_corner, top_right=right_corner)

        dist = hdr["DSUN_OBS"]
        px_unit = hdr["CUNIT1"]
        py_unit = hdr["CUNIT2"]
        px_arcsec = hdr["CDELT1"]
        py_arcsec = hdr["CDELT2"]

        nx = image.data.shape[1]
        ny = image.data.shape[0]

        nf = min(nx, ny)

        px_radians = px_arcsec / 206265.0
        py_radians = py_arcsec / 206265.0

        dist_Mm = dist * 10**-6
        px = px_radians * dist_Mm
        py = py_radians * dist_Mm

        xmin = 0.0
        ymin = 0.0
        zmin = 0.0

        xmax = nx * px
        ymax = ny * py
        zmax = 20.0

        pz = 90.0 * 10**-3
        # pz = max(px, py)

        nz = int(np.floor(zmax / pz))

        x = np.arange(nx) * (xmax - xmin) / (nx - 1) - xmin
        y = np.arange(ny) * (ymax - ymin) / (ny - 1) - ymin
        z = np.arange(nz) * (zmax - zmin) / (nz - 1) - zmin

        if np.fabs(check_fluxbalance(image.data)) < 0.01:
            return Field2dData(
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
                image.data,
                flux_balance_state=FluxBalanceState.BALANCED,
            )
        else:
            return Field2dData(
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
                image.data,
                flux_balance_state=FluxBalanceState.UNBALANCED,
            )


"""
    Summation of flux through the bottom boundary (photospheric Bz) normalised
    by the sum of absolute values. Value between -1 and 1, corresponding to entirely
    outward and inward flux, respectively. Can (probably) consider values between
    -0.01 and 0.01 as flux-balanced, such that the application of Seehafer is not
    necessary.
"""


def check_fluxbalance(bz: np.ndarray) -> float:

    return np.sum(bz) / np.sum(np.fabs(bz))


"""
    "Optimal" alpha calculated according to Hagino and Sakurai (2004).
    Alpha is calculated from the vertical electric current in the photosphere
    (from horizontal photospheric field) and the photospheric vertical magnetic field.
"""


def alpha_HS04(bx: np.ndarray, by: np.ndarray, bz: np.ndarray) -> float:

    Jz = np.gradient(by, axis=1) - np.gradient(bx, axis=0)
    return np.sum(Jz * np.sign(bz)) / np.sum(np.fabs(bz))
