from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mhsxtrapy._boundary import FluxBalanceState

__all__ = []


@dataclass
class FourierCoefficients:
    anm: np.ndarray
    bnm: np.ndarray
    cnm: np.ndarray
    dnm: np.ndarray

    def __truediv__(self, other: float | np.ndarray) -> FourierCoefficients:
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


def _extract_fourier_modes(
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

    # TO DO can maybe be replaced with array slicing on `signal[centre_y ± iy_arr, centre_x ± ix_arr]` using `np.ix_`.

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


def _compute_wavenumbers(
    nx: int,
    ny: int,
    px: float,
    py: float,
    nf: int,
    flux_balance_state: FluxBalanceState,
    lxn: float,
    lyn: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

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


def _seehafer(
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


def _compute_fourier_coefficients(nf, field, k2, lxn, lyn, kx, ky) -> FourierBasis:
    # Calls seehafer and xnm

    if field.flux_balance_state == FluxBalanceState.BALANCED:
        k2[0, 0] = (2.0 * np.pi / lxn) ** 2 + (2.0 * np.pi / lyn) ** 2

        coeffs = _extract_fourier_modes(field.bz, nf) / k2

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

        seehafer_bz = _seehafer(field.bz)

        coeffs = _extract_fourier_modes(seehafer_bz, nf) / k2
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
