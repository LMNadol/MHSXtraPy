from __future__ import annotations

import numpy as np

import logging

from typing import Tuple

from enum import Enum

from mhsxtrapy.phibar import (
    phi,
    dphidz,
    phi_nw,
    phi_low,
    dphidz_nw,
    dphidz_low,
)
from mhsxtrapy.field2d import Field2dData, FluxBalanceState


class WhichSolution(Enum):
    LOW = "Low"
    NEUWIE = "Neuwie"
    ASYMP = "Asymp"


def seehafer(
    bz: np.ndarray,
) -> np.ndarray:
    """
    Given the photospheric z-component of the magnetic field (field) returns Seehafer-mirrored
    field vector four times the size of original vector, (ny, nx,). Creates odd Seehafer-extension
    according to
        Bz(-x,y,0) = -Bz(x,y,0)
        Bz(x,-y,0) = -Bz(x,y,0)
        Bz(-x,-y,0) = Bz(x,y,0)
    Returns array of size (2ny, 2nx,)
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
) -> Tuple:
    """
    Given the Seehafer-mirrored photospheric vertical component of the magnetic field (data_bz) returns
    coefficients anm for series expansion of the 3D magnetic field (for definition of anm
    see PhD thesis of L Nadol) using nf_max-many Fourier modes in each direction.

    Calculates FFT coefficients using np.fft and shifts zeroth frequency into the centre of the
    array (exact position dependent on if nx and ny are even or odd). Then the coefficients anm
    are calcualted from a combination of real parts of FFT coefficients according to L Nadol PhD
    thesis. Returns array of size (nf_max, nf_max,).
    """

    anm = np.zeros((nf, nf))
    bnm = np.zeros((nf, nf))
    cnm = np.zeros((nf, nf))
    dnm = np.zeros((nf, nf))

    ny, nx = bz.shape

    signal = np.fft.fftshift(np.fft.fft2(bz) / nx / ny)

    for ix in range(0, nx, 2):
        for iy in range(1, ny, 2):
            temp = signal[iy, ix]
            signal[iy, ix] = -temp

    for ix in range(1, nx, 2):
        for iy in range(0, ny, 2):
            temp = signal[iy, ix]
            signal[iy, ix] = -temp

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

    return anm, bnm, cnm, dnm


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
    Returns two arrays of size (nf, nf, nz,) of the values of the functions Phi and
    its z-derivative for either Low, N+W or N+W-A, depending on the values of
    asymptotic and tanh: (True, True) = N+W-A, (False, True) = N+W, (False, False) = Low,
    (True, False) does not exist.
    """
    # if use_asymptotic and not use_tanh:
    #     return ValueError("Cannot use asymptotic solution of Low (1991, 1992).")

    phi_arr = np.zeros((nf, nf, nz))
    dphidz_arr = np.zeros((nf, nf, nz))

    if solution == WhichSolution.ASYMP:

        assert z0 is not None and deltaz is not None

        for iz, z in enumerate(z_arr):
            phi_arr[:, :, iz] = phi(z, p_arr, q_arr, z0, deltaz)
            dphidz_arr[:, :, iz] = dphidz(z, p_arr, q_arr, z0, deltaz)

    elif solution == WhichSolution.NEUWIE:

        assert z0 is not None and deltaz is not None

        for iz, z in enumerate(z_arr):
            phi_arr[:, :, iz] = phi_nw(z, p_arr, q_arr, z0, deltaz)
            dphidz_arr[:, :, iz] = dphidz_nw(z, p_arr, q_arr, z0, deltaz)

    elif solution == WhichSolution.LOW:

        assert kappa is not None

        for iy in range(0, int(nf)):
            for ix in range(0, int(nf)):
                q = q_arr[iy, ix]
                p = p_arr[iy, ix]
                for iz, z in enumerate(z_arr):
                    phi_arr[iy, ix, iz] = phi_low(z, p, q, kappa)
                    dphidz_arr[iy, ix, iz] = dphidz_low(z, p, q, kappa)

    else:
        raise ValueError(
            f"Invalid solution: {self.solution}. Expected 'LOW' or 'NEUWIE' or 'ASYMP'."
        )

    return phi_arr, dphidz_arr


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
    solution depending on the values of asymptotic and tanh: (True, True) = N+W-A,
    (False, True) = N+W, (False, False) = Low, (True, False) does not exist.

    Returns a tuple of two arrays of size (ny, nx, nz, 3,), of which the first one contains By, Bx
    and Bz and the second one contains partial derivatives of Bz (dBzdy, dBzdx and dBzdz).
    """

    if field.flux_balance_state == FluxBalanceState.BALANCED:
        nf = int(np.floor(field.nf / 2))
        l = 1.0
    elif field.flux_balance_state == FluxBalanceState.UNBALANCED:
        nf = field.nf
        l = 2.0
    else:
        raise ValueError(
            f"Invalid flux_balance_state: {field.flux_balance_state}. Expected 'BALANCED' or 'UNBALANCED'."
        )

    xmax, ymax = field.x[-1], field.y[-1]

    lx = field.nx * field.px * l
    ly = field.ny * field.py * l
    lxn = lx / l
    lyn = ly / l

    if field.flux_balance_state == FluxBalanceState.BALANCED:
        kx = np.arange(nf) * 2.0 * np.pi / lxn
        ky = np.arange(nf) * 2.0 * np.pi / lyn
    elif field.flux_balance_state == FluxBalanceState.UNBALANCED:
        kx = np.arange(nf) * np.pi / lxn
        ky = np.arange(nf) * np.pi / lyn
    else:
        raise ValueError(
            f"Invalid flux_balance_state: {field.flux_balance_state}. Expected 'BALANCED' or 'UNBALANCED'."
        )

    ones = 0.0 * np.arange(nf) + 1.0

    ky_grid = np.outer(ky, ones)
    kx_grid = np.outer(ones, kx)

    k2 = np.outer(ky**2, ones) + np.outer(ones, kx**2)

    if field.flux_balance_state == FluxBalanceState.BALANCED:
        k2[0, 0] = (2.0 * np.pi / lxn) ** 2 + (2.0 * np.pi / lyn) ** 2

        anm, bnm, cnm, dnm = np.divide(xnm(field.bz, nf), k2)

        bfield = np.zeros((field.ny, field.nx, field.nz, 3))
        dbz = np.zeros((field.ny, field.nx, field.nz, 3))

        sin_x = np.sin(np.outer(kx, field.x - lxn / 2.0))
        sin_y = np.sin(np.outer(ky, field.y - lyn / 2.0))
        cos_x = np.cos(np.outer(kx, field.x - lxn / 2.0))
        cos_y = np.cos(np.outer(ky, field.y - lyn / 2.0))
    elif field.flux_balance_state == FluxBalanceState.UNBALANCED:
        k2[0, 0] = (np.pi / lxn) ** 2 + (np.pi / lyn) ** 2
        k2[1, 0] = (np.pi / lxn) ** 2 + (np.pi / lyn) ** 2
        k2[0, 1] = (np.pi / lxn) ** 2 + (np.pi / lyn) ** 2

        seehafer_bz = seehafer(field.bz)

        anm, bnm, cnm, dnm = np.divide(xnm(seehafer_bz, nf), k2)
        bnm = np.zeros_like(bnm)
        cnm = np.zeros_like(cnm)
        dnm = np.zeros_like(dnm)

        bfield = np.zeros((2 * field.ny, 2 * field.nx, field.nz, 3))
        dbz = np.zeros((2 * field.ny, 2 * field.nx, field.nz, 3))

        x_big = np.arange(2.0 * field.nx) * 2.0 * xmax / (2.0 * field.nx - 1) - xmax
        y_big = np.arange(2.0 * field.ny) * 2.0 * ymax / (2.0 * field.ny - 1) - ymax

        sin_x = np.sin(np.outer(kx, x_big))
        sin_y = np.sin(np.outer(ky, y_big))
        cos_x = np.cos(np.outer(kx, x_big))
        cos_y = np.cos(np.outer(ky, y_big))
    else:
        raise ValueError(
            f"Invalid flux_balance_state: {field.flux_balance_state}. Expected 'BALANCED' or 'UNBALANCED'."
        )

    if solution == WhichSolution.NEUWIE or solution == WhichSolution.ASYMP:
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
            f"Unknown solution type: {solution}. Expected 'LOW' or 'NEUWIE' or 'ASYMP'."
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

    for iz in range(0, field.nz):

        coeffs1 = np.multiply(np.multiply(k2, phi[:, :, iz]), anm)
        coeffs2 = np.multiply(np.multiply(k2, phi[:, :, iz]), bnm)
        coeffs3 = np.multiply(np.multiply(k2, phi[:, :, iz]), cnm)
        coeffs4 = np.multiply(np.multiply(k2, phi[:, :, iz]), dnm)

        bfield[:, :, iz, 2] = (
            np.matmul(sin_y.T, np.matmul(coeffs1, sin_x))
            + np.matmul(cos_y.T, np.matmul(coeffs2, sin_x))
            + np.matmul(sin_y.T, np.matmul(coeffs3, cos_x))
            + np.matmul(cos_y.T, np.matmul(coeffs4, cos_x))
        )

        coeffs1 = np.multiply(
            np.multiply(anm, dphi[:, :, iz]), ky_grid
        ) + alpha * np.multiply(np.multiply(dnm, phi[:, :, iz]), kx_grid)
        coeffs2 = -np.multiply(
            np.multiply(dnm, dphi[:, :, iz]), ky_grid
        ) - alpha * np.multiply(np.multiply(anm, phi[:, :, iz]), kx_grid)
        coeffs3 = -np.multiply(
            np.multiply(bnm, dphi[:, :, iz]), ky_grid
        ) + alpha * np.multiply(np.multiply(cnm, phi[:, :, iz]), kx_grid)
        coeffs4 = np.multiply(
            np.multiply(cnm, dphi[:, :, iz]), ky_grid
        ) - alpha * np.multiply(np.multiply(bnm, phi[:, :, iz]), kx_grid)

        bfield[:, :, iz, 0] = (
            np.matmul(cos_y.T, np.matmul(coeffs4, cos_x))
            + np.matmul(sin_y.T, np.matmul(coeffs3, sin_x))
            + np.matmul(cos_y.T, np.matmul(coeffs1, sin_x))
            + np.matmul(sin_y.T, np.matmul(coeffs2, cos_x))
        )

        coeffs1 = -np.multiply(
            np.multiply(cnm, dphi[:, :, iz]), kx_grid
        ) - alpha * np.multiply(np.multiply(bnm, phi[:, :, iz]), ky_grid)
        coeffs2 = np.multiply(
            np.multiply(bnm, dphi[:, :, iz]), kx_grid
        ) + alpha * np.multiply(np.multiply(cnm, phi[:, :, iz]), ky_grid)
        coeffs3 = np.multiply(
            np.multiply(anm, dphi[:, :, iz]), kx_grid
        ) - alpha * np.multiply(np.multiply(dnm, phi[:, :, iz]), ky_grid)
        coeffs4 = -np.multiply(
            np.multiply(dnm, dphi[:, :, iz]), kx_grid
        ) + alpha * np.multiply(np.multiply(anm, phi[:, :, iz]), ky_grid)

        bfield[:, :, iz, 1] = (
            np.matmul(cos_y.T, np.matmul(coeffs2, cos_x))
            + np.matmul(sin_y.T, np.matmul(coeffs1, sin_x))
            + np.matmul(sin_y.T, np.matmul(coeffs3, cos_x))
            + np.matmul(cos_y.T, np.matmul(coeffs4, sin_x))
        )

        coeffs1 = np.multiply(np.multiply(k2, dphi[:, :, iz]), anm)
        coeffs2 = np.multiply(np.multiply(k2, dphi[:, :, iz]), bnm)
        coeffs3 = np.multiply(np.multiply(k2, dphi[:, :, iz]), cnm)
        coeffs4 = np.multiply(np.multiply(k2, dphi[:, :, iz]), dnm)
        dbz[:, :, iz, 2] = (
            np.matmul(sin_y.T, np.matmul(coeffs1, sin_x))
            + np.matmul(cos_y.T, np.matmul(coeffs2, sin_x))
            + np.matmul(sin_y.T, np.matmul(coeffs3, cos_x))
            + np.matmul(cos_y.T, np.matmul(coeffs4, cos_x))
        )

        coeffs1 = np.multiply(np.multiply(np.multiply(k2, phi[:, :, iz]), anm), kx_grid)
        coeffs2 = np.multiply(np.multiply(np.multiply(k2, phi[:, :, iz]), bnm), kx_grid)
        coeffs3 = -np.multiply(
            np.multiply(np.multiply(k2, phi[:, :, iz]), cnm), kx_grid
        )
        coeffs4 = -np.multiply(
            np.multiply(np.multiply(k2, phi[:, :, iz]), dnm), kx_grid
        )

        dbz[:, :, iz, 0] = (
            np.matmul(sin_y.T, np.matmul(coeffs1, cos_x))
            + np.matmul(cos_y.T, np.matmul(coeffs2, cos_x))
            + np.matmul(sin_y.T, np.matmul(coeffs3, sin_x))
            + np.matmul(cos_y.T, np.matmul(coeffs4, sin_x))
        )

        coeffs1 = np.multiply(np.multiply(np.multiply(k2, phi[:, :, iz]), anm), ky_grid)
        coeffs2 = -np.multiply(
            np.multiply(np.multiply(k2, phi[:, :, iz]), bnm), ky_grid
        )
        coeffs3 = +np.multiply(
            np.multiply(np.multiply(k2, phi[:, :, iz]), cnm), ky_grid
        )
        coeffs4 = -np.multiply(
            np.multiply(np.multiply(k2, phi[:, :, iz]), dnm), ky_grid
        )
        dbz[:, :, iz, 1] = (
            np.matmul(cos_y.T, np.matmul(coeffs1, sin_x))
            + np.matmul(sin_y.T, np.matmul(coeffs2, sin_x))
            + np.matmul(cos_y.T, np.matmul(coeffs3, cos_x))
            + np.matmul(sin_y.T, np.matmul(coeffs4, cos_x))
        )

    return bfield, dbz
