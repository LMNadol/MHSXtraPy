from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from mhsxtrapy._boundary import BoundaryData, FluxBalanceState
from mhsxtrapy._fourier import (
    FourierCoefficients,
    _compute_fourier_coefficients,
    _compute_wavenumbers,
)
from mhsxtrapy.solutions import get_solution
from mhsxtrapy.types import WhichSolution

logger = logging.getLogger(__name__)

__all__ = []


@dataclass
class MagneticField:
    bfield: np.ndarray
    dbz: np.ndarray


def _get_phi_dphi(
    z_arr: np.ndarray,
    q_arr: np.ndarray,
    p_arr: np.ndarray,
    nf: int,
    nz: int,
    solution: WhichSolution,
    z0: float | None = None,
    deltaz: float | None = None,
    kappa: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
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

    for iz, z in enumerate(z_arr):
        phi_arr[:, :, iz] = sol.phi(z, p_arr, q_arr)
        dphidz_arr[:, :, iz] = sol.dphidz(z, p_arr, q_arr)

    return phi_arr, dphidz_arr


def _compute_field_at_height(
    iz: int,
    nx: int,
    ny: int,
    nf: int,
    sin_x: np.ndarray,
    cos_x: np.ndarray,
    sin_y: np.ndarray,
    cos_y: np.ndarray,
    phi: np.ndarray,
    dphi: np.ndarray,
    coeffs: FourierCoefficients,
    kx: np.ndarray,
    ky: np.ndarray,
    k2: np.ndarray,
    alpha: float,
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


def _extrapolate_3d(
    field: BoundaryData,
    alpha: float,
    a: float,
    solution: WhichSolution,
    b: float | None = None,
    z0: float | None = None,
    deltaz: float | None = None,
    kappa: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate 3D magnetic field from BoundaryData and given paramters a, b, alpha, z0 and delta z
    (for definitions see PhD thesis L Nadol). Extrapolation based on either Low, N+W or N+W-A
    solution depending on the value of solution.

    Returns a tuple of two arrays of size (ny, nx, nz, 3,), of which the first one contains By, Bx
    and Bz and the second one contains partial derivatives of Bz (dBzdy, dBzdx and dBzdz).

    Args:
        field (BoundaryData): BoundaryData object containing all the boundary condition information
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

    kx, ky, k2 = _compute_wavenumbers(
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
        if z0 is None or deltaz is None or b is None:
            raise ValueError(
                "z0, deltaz, and b must not be None for NEUKIRCH_WIEGELMANN / NADOL_NEUKIRCH solution."
            )
        p = 0.5 * deltaz * np.sqrt(k2 * (1.0 - a - a * b) - alpha**2)
        q = 0.5 * deltaz * np.sqrt(k2 * (1.0 - a + a * b) - alpha**2)
    elif solution == WhichSolution.LOW:
        if kappa is None:
            raise ValueError("kappa must not be None for LOW solution.")
        p = 2.0 / kappa * np.sqrt(k2 - alpha**2)
        q = 2.0 / kappa * np.sqrt(k2 * a)
    else:
        logger.warning("Unknown solution type: %s.", solution)
        raise ValueError(
            f"Unknown solution type: {solution}. Expected 'LOW' or 'NEUWIE' or 'NANEU'."
        )

    phi, dphi = _get_phi_dphi(
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
