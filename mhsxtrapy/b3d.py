from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from mhsxtrapy.field2d import Field2dData, FluxBalanceState
from mhsxtrapy.solutions import get_solution
from mhsxtrapy.types import WhichSolution

__all__ = [
    "FourierCoefficients",
    "TrigBasis",
    "FourierMeta",
    "FourierBasis",
    "MagneticField",
    "seehafer",
    "xnm",
    "get_phi_dphi",
    "compute_wavenumbers",
    "b3d",
]


@dataclass
class FourierCoefficients:
    anm: np.ndarray
    bnm: np.ndarray
    cnm: np.ndarray
    dnm: np.ndarray

    def __truediv__(self, other):
        return FourierCoefficients(
            anm=self.anm / other,
            bnm=self.bnm / other,
            cnm=self.cnm / other,
            dnm=self.dnm / other,
        )


@dataclass
class TrigBasis:
    sin_x: np.ndarray
    cos_x: np.ndarray
    sin_y: np.ndarray
    cos_y: np.ndarray


@dataclass
class FourierMeta:
    nx: int
    ny: int
    bfield: np.ndarray
    dbz: np.ndarray


@dataclass
class FourierBasis:
    coeffs: FourierCoefficients
    trigfunc: TrigBasis
    meta: FourierMeta


@dataclass
class MagneticField:
    bfield: np.ndarray
    dbz: np.ndarray


def seehafer(
    bz: np.ndarray,
) -> np.ndarray:
    """
    Given the photospheric z-component of the magnetic field returns Seehafer-mirrored
    field vector four times the size of original vector, (ny, nx,). Creates odd Seehafer-extension
    according to
        Bz(-x,y,0) = -Bz(x,y,0)
        Bz(x,-y,0) = -Bz(x,y,0)
        Bz(-x,-y,0) = Bz(x,y,0)

    Args:
        bz (np.ndarray): photospheric z-component of the magnetic field of size (ny, nx,)

    Returns:
        np.ndarray: Seehafer-mirrored photospheric magnetic field vector of size (2ny, 2nx,)
    """

    ny, nx = bz.shape

    bz_big = np.zeros((2 * ny, 2 * nx))

    bz_big[ny:, nx:] = bz  # Flip both axes of field and assign to bottom-right
    bz_big[ny:, :nx] = -bz[:, ::-1]  # Flip horizontally and assign to bottom-left
    bz_big[:ny, nx:] = -bz[::-1, :]  # Flip vertically and assign to top-right
    bz_big[:ny, :nx] = bz[::-1, ::-1]  # Flip both axes of field and assign to top-left

    return bz_big


def xnm(
    bz: np.ndarray,
    nf: int,
) -> FourierCoefficients:
    """
    Given the Seehafer-mirrored photospheric vertical component of the magnetic field returns
    coefficients anm, bnm, cnm, dnm for series expansion of the 3D magnetic field (for definition of anm
    see PhD thesis of L Nadol) using nf-many Fourier modes in each direction.

    Calculates FFT coefficients using np.fft and shifts zeroth frequency into the centre of the
    array (exact position dependent on if nx and ny are even or odd). Then the coefficients anm
    are calcualted from a combination of real parts of FFT coefficients according to L Nadol PhD
    thesis. Returns arrays of size (nf, nf).

    Args:
        bz (np.ndarray): Seehafer-mirrored photospheric vertical component of the magnetic field
        nf (int): number of Fourier modes in each horizontal direction

    Returns:
        Tuple: coefficient arrays anm, bnm, cnm, dnm, each of size (nf, nf,)
    """

    anm = np.zeros((nf, nf))
    bnm = np.zeros((nf, nf))
    cnm = np.zeros((nf, nf))
    dnm = np.zeros((nf, nf))

    ny, nx = bz.shape

    signal = np.fft.fftshift(np.fft.fft2(bz) / nx / ny)

    signal[1::2, ::2] *= -1
    signal[::2, 1::2] *= -1

    if nx % 2 == 0:
        centre_x = int(nx / 2)
    else:
        centre_x = int((nx - 1) / 2)
    if ny % 2 == 0:
        centre_y = int(ny / 2)
    else:
        centre_y = int((ny - 1) / 2)

    centre_x, centre_y = nx // 2, ny // 2

    for ix in range(nf):
        for iy in range(nf):
            anm[iy, ix] = (
                -signal[centre_y + iy, centre_x + ix]
                + signal[centre_y + iy, centre_x - ix]
                + signal[centre_y - iy, centre_x + ix]
                - signal[centre_y - iy, centre_x - ix]
            ).real
            bnm[iy, ix] = (
                -signal[centre_y + iy, centre_x + ix]
                + signal[centre_y + iy, centre_x - ix]
                - signal[centre_y - iy, centre_x + ix]
                + signal[centre_y - iy, centre_x - ix]
            ).imag
            cnm[iy, ix] = (
                -signal[centre_y + iy, centre_x + ix]
                + signal[centre_y - iy, centre_x + ix]
                - signal[centre_y + iy, centre_x - ix]
                + signal[centre_y - iy, centre_x - ix]
            ).imag
            dnm[iy, ix] = (
                signal[centre_y + iy, centre_x + ix]
                + signal[centre_y + iy, centre_x - ix]
                + signal[centre_y - iy, centre_x + ix]
                + signal[centre_y - iy, centre_x - ix]
            ).real

    for iy in range(1, nf):
        dnm[iy, 0] = (
            signal[centre_y + iy, centre_x + 0] + signal[centre_y - iy, centre_x + 0]
        ).real
        cnm[iy, 0] = (
            -signal[centre_y + iy, centre_x + 0] + signal[centre_y - iy, centre_x + 0]
        ).imag

    for ix in range(1, nf):
        dnm[0, ix] = (
            signal[centre_y + 0, centre_x + ix] + signal[centre_y + 0, centre_x - ix]
        ).real
        bnm[0, ix] = (
            -signal[centre_y + 0, centre_x + ix] + signal[centre_y + 0, centre_x - ix]
        ).imag

    return FourierCoefficients(anm, bnm, cnm, dnm)


def get_phi_dphi(
    z_arr: np.ndarray,
    q_arr: np.ndarray,
    p_arr: np.ndarray,
    nf: int,
    nz: int,
    solution: WhichSolution,
    z0: float | None = None,
    deltaz: float | None = None,
    kappa: float | None = None,
):
    """
    Returns two arrays of size (nf, nf, nz,) of the values of the functions bar-Phi and
    its z-derivative for either Low, N+W or N+W-A, depending on the value of the Enum solution.

    Args:
        z_arr (np.ndarray): grid points in vertical direction of computational box, in lengthscale L
        q_arr (np.ndarray): array containing the parameter q, i.e. delta in L Nadol PhD thesis
        p_arr (np.ndarray): array containing the parameter p, i.e. gamma in L Nadol PhD thesis
        nf (int): number of Fourier modes in both horizontal directions
        nz (int): number of grid points in vertical direction
        solution (WhichSolution): Enum for solution selection, either "Low", "Neuwie" or "Naneu"
        z0 (float | None, optional): Centre of region over which transition from non-force-free to force-free
            takes place in "Neuwie" and "Naneu" solution. Defaults to None.
        deltaz (float | None, optional): Width of region over which transition from non-force-free to force-free
            takes place in "Neuwie" and "Naneu". Defaults to None.
        kappa (float | None, optional): Parameter for "Low" solution setting the speed with which the
            exponential function declines. Defaults to None.

    Raises:
        ValueError: In case the parameter choices for z0, deltza and kappa do not fit with the choice of solution.

    Returns:
        _type_: Arrays containig bar-Phi and its first derivative w.r.t. z.
    """
    phi_arr = np.zeros((nf, nf, nz))
    dphidz_arr = np.zeros((nf, nf, nz))

    sol = get_solution(solution, z0=z0, deltaz=deltaz, kappa=kappa)

    if (
        solution == WhichSolution.NADOL_NEUKIRCH
        or solution == WhichSolution.NEUKIRCH_WIEGELMANN
    ):

        for iz, z in enumerate(z_arr):
            phi_arr[:, :, iz] = sol.phi(z, p_arr, q_arr)
            dphidz_arr[:, :, iz] = sol.dphidz(z, p_arr, q_arr)

    elif solution == WhichSolution.LOW:

        for iy in range(0, int(nf)):
            for ix in range(0, int(nf)):
                q = q_arr[iy, ix]
                p = p_arr[iy, ix]
                for iz, z in enumerate(z_arr):
                    phi_arr[iy, ix, iz] = sol.phi(z, p, q)
                    dphidz_arr[iy, ix, iz] = sol.dphidz(z, p, q)

    return phi_arr, dphidz_arr


def compute_wavenumbers(nx, ny, px, py, nf, flux_balance_state, lxn, lyn) -> Tuple:

    if flux_balance_state == FluxBalanceState.BALANCED:
        kx = np.arange(nf) * 2.0 * np.pi / lxn
        ky = np.arange(nf) * 2.0 * np.pi / lyn
    elif flux_balance_state == FluxBalanceState.UNBALANCED:
        kx = np.arange(nf) * np.pi / lxn
        ky = np.arange(nf) * np.pi / lyn
    else:
        raise ValueError(
            f"Invalid flux_balance_state: {flux_balance_state}. Expected 'BALANCED' or 'UNBALANCED'."
        )

    # ones = 0.0 * np.arange(nf) + 1.0
    # k2 = np.outer(ky**2, ones) + np.outer(ones, kx**2)

    # ones = np.ones(nf)
    k2 = ky[:, None] ** 2 + kx[None, :] ** 2

    return kx, ky, k2


def _compute_fourier_coefficients(nf, field, k2, lxn, lyn, kx, ky) -> FourierBasis:
    # Calls seehafer and xnm

    if field.flux_balance_state == FluxBalanceState.BALANCED:
        k2[0, 0] = (2.0 * np.pi / lxn) ** 2 + (2.0 * np.pi / lyn) ** 2

        coeffs = xnm(field.bz, nf) / k2

        nx = field.nx
        ny = field.ny
        bfield = np.zeros((ny, nx, field.nz, 3))
        dbz = np.zeros((ny, nx, field.nz, 3))

        # Use FFT-consistent grid for reconstruction: the FFT in xnm
        # assumes sample spacing px = L/N, so positions must be j*px
        # (not linspace which gives j*L/(N-1) and causes phase errors).
        x_recon = np.arange(field.nx, dtype=np.float64) * field.px  # instead of field.x
        y_recon = np.arange(field.ny, dtype=np.float64) * field.py  # instead of field.y
        sin_x = np.sin(np.outer(kx, x_recon - lxn / 2.0))
        sin_y = np.sin(np.outer(ky, y_recon - lyn / 2.0))
        cos_x = np.cos(np.outer(kx, x_recon - lxn / 2.0))
        cos_y = np.cos(np.outer(ky, y_recon - lyn / 2.0))
    elif field.flux_balance_state == FluxBalanceState.UNBALANCED:
        k2[0, 0] = (np.pi / lxn) ** 2 + (np.pi / lyn) ** 2
        k2[1, 0] = (np.pi / lxn) ** 2 + (np.pi / lyn) ** 2
        k2[0, 1] = (np.pi / lxn) ** 2 + (np.pi / lyn) ** 2

        seehafer_bz = seehafer(field.bz)

        coeffs = xnm(seehafer_bz, nf) / k2
        coeffs.bnm = np.zeros_like(coeffs.bnm)
        coeffs.cnm = np.zeros_like(coeffs.cnm)
        coeffs.dnm = np.zeros_like(coeffs.dnm)

        nx = 2 * field.nx
        ny = 2 * field.ny

        bfield = np.zeros((ny, nx, field.nz, 3))
        dbz = np.zeros((ny, nx, field.nz, 3))

        # Use FFT-consistent grid for Seehafer domain reconstruction
        x_big = (
            np.arange(nx, dtype=np.float64) * field.px - field.nx * field.px
        )  # instead of x_big = np.arange(2.0 * field.nx) * 2.0 * xmax / (2.0 * field.nx - 1) - xmax
        y_big = (
            np.arange(ny, dtype=np.float64) * field.py - field.ny * field.py
        )  # instead of y_big = np.arange(2.0 * field.ny) * 2.0 * ymax / (2.0 * field.ny - 1) - ymax

        sin_x = np.sin(np.outer(kx, x_big))
        sin_y = np.sin(np.outer(ky, y_big))
        cos_x = np.cos(np.outer(kx, x_big))
        cos_y = np.cos(np.outer(ky, y_big))
    else:
        raise ValueError(
            f"Invalid flux_balance_state: {field.flux_balance_state}. Expected 'BALANCED' or 'UNBALANCED'."
        )

    trigfunc = TrigBasis(sin_x, cos_x, sin_y, cos_y)
    meta = FourierMeta(nx, ny, bfield, dbz)

    return FourierBasis(coeffs, trigfunc, meta)


def _compute_field_at_height(
    iz, nx, ny, nf, sin_x, cos_x, sin_y, cos_y, phi, dphi, coeffs, kx, ky, k2, alpha
) -> MagneticField:
    # Single z-level field computation

    bfield_at_z = np.zeros((ny, nx, 3))
    dbz_at_z = np.zeros((ny, nx, 3))

    # ones = 0.0 * np.arange(nf) + 1.0
    ones = np.ones(nf)
    ky_grid = np.outer(ky, ones)
    kx_grid = np.outer(ones, kx)

    coeffs1 = np.multiply(np.multiply(k2, phi[:, :, iz]), coeffs.anm)
    coeffs2 = np.multiply(np.multiply(k2, phi[:, :, iz]), coeffs.bnm)
    coeffs3 = np.multiply(np.multiply(k2, phi[:, :, iz]), coeffs.cnm)
    coeffs4 = np.multiply(np.multiply(k2, phi[:, :, iz]), coeffs.dnm)

    bfield_at_z[:, :, 2] = (
        np.matmul(sin_y.T, np.matmul(coeffs1, sin_x))
        + np.matmul(cos_y.T, np.matmul(coeffs2, sin_x))
        + np.matmul(sin_y.T, np.matmul(coeffs3, cos_x))
        + np.matmul(cos_y.T, np.matmul(coeffs4, cos_x))
    )

    coeffs1 = np.multiply(
        np.multiply(coeffs.anm, dphi[:, :, iz]), ky_grid
    ) + alpha * np.multiply(np.multiply(coeffs.dnm, phi[:, :, iz]), kx_grid)
    coeffs2 = -np.multiply(
        np.multiply(coeffs.dnm, dphi[:, :, iz]), ky_grid
    ) - alpha * np.multiply(np.multiply(coeffs.anm, phi[:, :, iz]), kx_grid)
    coeffs3 = -np.multiply(
        np.multiply(coeffs.bnm, dphi[:, :, iz]), ky_grid
    ) + alpha * np.multiply(np.multiply(coeffs.cnm, phi[:, :, iz]), kx_grid)
    coeffs4 = np.multiply(
        np.multiply(coeffs.cnm, dphi[:, :, iz]), ky_grid
    ) - alpha * np.multiply(np.multiply(coeffs.bnm, phi[:, :, iz]), kx_grid)

    bfield_at_z[:, :, 0] = (
        np.matmul(cos_y.T, np.matmul(coeffs4, cos_x))
        + np.matmul(sin_y.T, np.matmul(coeffs3, sin_x))
        + np.matmul(cos_y.T, np.matmul(coeffs1, sin_x))
        + np.matmul(sin_y.T, np.matmul(coeffs2, cos_x))
    )

    coeffs1 = -np.multiply(
        np.multiply(coeffs.cnm, dphi[:, :, iz]), kx_grid
    ) - alpha * np.multiply(np.multiply(coeffs.bnm, phi[:, :, iz]), ky_grid)
    coeffs2 = np.multiply(
        np.multiply(coeffs.bnm, dphi[:, :, iz]), kx_grid
    ) + alpha * np.multiply(np.multiply(coeffs.cnm, phi[:, :, iz]), ky_grid)
    coeffs3 = np.multiply(
        np.multiply(coeffs.anm, dphi[:, :, iz]), kx_grid
    ) - alpha * np.multiply(np.multiply(coeffs.dnm, phi[:, :, iz]), ky_grid)
    coeffs4 = -np.multiply(
        np.multiply(coeffs.dnm, dphi[:, :, iz]), kx_grid
    ) + alpha * np.multiply(np.multiply(coeffs.anm, phi[:, :, iz]), ky_grid)

    bfield_at_z[:, :, 1] = (
        np.matmul(cos_y.T, np.matmul(coeffs2, cos_x))
        + np.matmul(sin_y.T, np.matmul(coeffs1, sin_x))
        + np.matmul(sin_y.T, np.matmul(coeffs3, cos_x))
        + np.matmul(cos_y.T, np.matmul(coeffs4, sin_x))
    )

    coeffs1 = np.multiply(np.multiply(k2, dphi[:, :, iz]), coeffs.anm)
    coeffs2 = np.multiply(np.multiply(k2, dphi[:, :, iz]), coeffs.bnm)
    coeffs3 = np.multiply(np.multiply(k2, dphi[:, :, iz]), coeffs.cnm)
    coeffs4 = np.multiply(np.multiply(k2, dphi[:, :, iz]), coeffs.dnm)

    dbz_at_z[:, :, 2] = (
        np.matmul(sin_y.T, np.matmul(coeffs1, sin_x))
        + np.matmul(cos_y.T, np.matmul(coeffs2, sin_x))
        + np.matmul(sin_y.T, np.matmul(coeffs3, cos_x))
        + np.matmul(cos_y.T, np.matmul(coeffs4, cos_x))
    )

    coeffs1 = np.multiply(
        np.multiply(np.multiply(k2, phi[:, :, iz]), coeffs.anm), kx_grid
    )
    coeffs2 = np.multiply(
        np.multiply(np.multiply(k2, phi[:, :, iz]), coeffs.bnm), kx_grid
    )
    coeffs3 = -np.multiply(
        np.multiply(np.multiply(k2, phi[:, :, iz]), coeffs.cnm), kx_grid
    )
    coeffs4 = -np.multiply(
        np.multiply(np.multiply(k2, phi[:, :, iz]), coeffs.dnm), kx_grid
    )

    dbz_at_z[:, :, 0] = (
        np.matmul(sin_y.T, np.matmul(coeffs1, cos_x))
        + np.matmul(cos_y.T, np.matmul(coeffs2, cos_x))
        + np.matmul(sin_y.T, np.matmul(coeffs3, sin_x))
        + np.matmul(cos_y.T, np.matmul(coeffs4, sin_x))
    )

    coeffs1 = np.multiply(
        np.multiply(np.multiply(k2, phi[:, :, iz]), coeffs.anm), ky_grid
    )
    coeffs2 = -np.multiply(
        np.multiply(np.multiply(k2, phi[:, :, iz]), coeffs.bnm), ky_grid
    )
    coeffs3 = +np.multiply(
        np.multiply(np.multiply(k2, phi[:, :, iz]), coeffs.cnm), ky_grid
    )
    coeffs4 = -np.multiply(
        np.multiply(np.multiply(k2, phi[:, :, iz]), coeffs.dnm), ky_grid
    )
    dbz_at_z[:, :, 1] = (
        np.matmul(cos_y.T, np.matmul(coeffs1, sin_x))
        + np.matmul(sin_y.T, np.matmul(coeffs2, sin_x))
        + np.matmul(cos_y.T, np.matmul(coeffs3, cos_x))
        + np.matmul(sin_y.T, np.matmul(coeffs4, cos_x))
    )

    return MagneticField(bfield_at_z, dbz_at_z)


def b3d(
    field: Field2dData,
    alpha: float,
    a: float,
    solution: WhichSolution,
    b: float | None = None,
    z0: float | None = None,
    deltaz: float | None = None,
    kappa: float | None = None,
) -> Tuple:
    """
    Calculate 3D magnetic field from Field2dData and given paramters a, b, alpha, z0 and delta z
    (for definitions see PhD thesis L Nadol). Extrapolation based on either Low, N+W or N+W-A
    solution depending on the value of solution.

    Returns a tuple of two arrays of size (ny, nx, nz, 3,), of which the first one contains By, Bx
    and Bz and the second one contains partial derivatives of Bz (dBzdy, dBzdx and dBzdz).

    Args:
        field (Field2dData): Field2dData object containing all the boundary condition information
        alpha (float): force-free parameter
        a (float): amplitude parameter
        solution (WhichSolution): Enum for solution selection, either "Low", "Neuwie" or "Naneu"
        b (float | None, optional): "switch-off" parameter
        z0 (float | None, optional): Centre of region over which transition from non-force-free to force-free
            takes place in "Neuwie" and "Naneu" solution. Defaults to None.
        deltaz (float | None, optional): Width of region over which transition from non-force-free to force-free
            takes place in "Neuwie" and "Naneu". Defaults to None.
        kappa (float | None, optional): Parameter for "Low" solution setting the speed with which the
            exponential function declines. Defaults to None.

    Raises:
        ValueError: In case the Enum fluxbalance is neither BALANCED nor UNBALANCED.
        ValueError: In case the parameter choices for z0, deltza and kappa do not fit with the choice of solution.

    Returns:
        Tuple: a tuple of two arrays of size (ny, nx, nz, 3,), of which the first one contains By, Bx
            and Bz and the second one contains partial derivatives of Bz (dBzdy, dBzdx and dBzdz).
    """

    if field.flux_balance_state == FluxBalanceState.BALANCED:
        nf = int(np.floor(field.nf / 2))
        lengthscale = 1.0
    elif field.flux_balance_state == FluxBalanceState.UNBALANCED:
        nf = field.nf
        lengthscale = 2.0
    else:
        raise ValueError(
            f"Invalid flux_balance_state: {field.flux_balance_state}. Expected 'BALANCED' or 'UNBALANCED'."
        )

    # xmax, ymax = field.x[-1], field.y[-1]

    lx = field.nx * field.px * lengthscale
    ly = field.ny * field.py * lengthscale
    lxn = lx / lengthscale
    lyn = ly / lengthscale

    kx, ky, k2 = compute_wavenumbers(
        field.nx, field.ny, field.px, field.py, nf, field.flux_balance_state, lxn, lyn
    )

    # ones = 0.0 * np.arange(nf) + 1.0

    basis = _compute_fourier_coefficients(nf, field, k2, lxn, lyn, kx, ky)

    coeffs = basis.coeffs
    trig_func = basis.trigfunc
    meta = basis.meta

    if (
        solution == WhichSolution.NEUKIRCH_WIEGELMANN
        or solution == WhichSolution.NADOL_NEUKIRCH
    ):
        assert z0 is not None and deltaz is not None and b is not None
        p = 0.5 * deltaz * np.sqrt(k2 * (1.0 - a - a * b) - alpha**2)
        q = 0.5 * deltaz * np.sqrt(k2 * (1.0 - a + a * b) - alpha**2)
    elif solution == WhichSolution.LOW:
        assert kappa is not None
        p = 2.0 / kappa * np.sqrt(k2 - alpha**2)
        q = 2.0 / kappa * np.sqrt(k2 * a)
    else:
        logging.warning(f"Unknown solution type: {solution}.")
        raise ValueError(
            f"Unknown solution type: {solution}. Expected 'LOW' or 'NEUWIE' or 'NANEU'."
        )

    phi, dphi = get_phi_dphi(
        field.z,
        q,
        p,
        nf,
        field.nz,
        solution=solution,
        z0=z0,
        deltaz=deltaz,
        kappa=kappa,
    )

    bfield = meta.bfield
    dbz = meta.dbz

    for iz in range(0, field.nz):

        mfield = _compute_field_at_height(
            iz,
            meta.nx,
            meta.ny,
            nf,
            trig_func.sin_x,
            trig_func.cos_x,
            trig_func.sin_y,
            trig_func.cos_y,
            phi,
            dphi,
            coeffs,
            kx,
            ky,
            k2,
            alpha,
        )

        bfield[:, :, iz, 2] = mfield.bfield[:, :, 2]
        bfield[:, :, iz, 0] = mfield.bfield[:, :, 0]
        bfield[:, :, iz, 1] = mfield.bfield[:, :, 1]

        dbz[:, :, iz, 2] = mfield.dbz[:, :, 2]
        dbz[:, :, iz, 0] = mfield.dbz[:, :, 0]
        dbz[:, :, iz, 1] = mfield.dbz[:, :, 1]

    return bfield, dbz
