from __future__ import annotations

import numpy as np
from numba import njit

from .base import Solution
from .neuwie import dfdz_nw, f_nw

__all__ = ["NaNeuSolution", "phi", "dphidz"]


class NaNeuSolution(Solution):

    def phi(self, z, p, q, z0, deltaz):
        return phi(z, p, q, z0, deltaz)

    def dphidz(self, z, p, q, z0, deltaz):
        return dphidz(z, p, q, z0, deltaz)

    def f(self, z, z0, deltaz, a, b):
        return f_nw(z, z0, deltaz, a, b)

    def dfdz(self, z, z0, deltaz, a, b):
        return dfdz_nw(z, z0, deltaz, a, b)


@njit
def phi(
    z: np.float64,
    p: np.ndarray,
    q: np.ndarray,
    z0: float,
    deltaz: float,
) -> np.ndarray:
    """
    Returns solution of asymptotic approximated version of ODE (22)
    in Neukirch and Wiegelmann (2019) which defines the poloidal component
    of the magnetic field vector, in the case C-, C+ > 0 (for definitions
    see L Nadol PhD thesis).

    Vectorisation possible for p and q, which have to be passed as arrays of the
    same size. Vectorisation for z not possible due to differentiation between
    z < z0 and z > z0. Returns array of size p.shape = q.shape which is (nf, nf,).

    Args:
        z (np.float64): grid points in vertical direction
        p (np.ndarray): array containing parameter p
        q (np.ndarray): array containing parameter q
        z0 (float): centre of region over which transition from non-force-free to force-free takes place
        deltaz (float): width of region over which transition from non-force-free to force-free takes place

    Returns:
        np.ndarray: bar-phi according to Nadol and Neukirch (2025)
    """

    rplus = p / deltaz
    rminus = q / deltaz

    r = rminus / rplus

    d = np.cosh(2.0 * rplus * z0) + np.multiply(r, np.sinh(2.0 * rplus * z0))

    if z - z0 < 0.0:
        return (
            np.cosh(2.0 * rplus * (z0 - z))
            + np.multiply(r, np.sinh(2.0 * rplus * (z0 - z)))
        ) / d

    else:
        return np.exp(-2.0 * rminus * (z - z0)) / d


@njit
def dphidz(
    z: np.float64,
    p: np.ndarray,
    q: np.ndarray,
    z0: float,
    deltaz: float,
) -> np.ndarray:
    """
    Returns z-derivative of solution of asymptotic approximated version of ODE (22)
    in Neukirch and Wiegelmann (2019) which defines the poloidal component
    of the magnetic field vector, in the case C-, C+ > 0 (for definitions
    see L Nadol PhD thesis).

    Vectorisation possible for p and q, which have to be passed as arrays of the
    same size. Vectorisation for z not possible due to differentiation between
    z < z0 and z > z0. Returns array of size (nf, nf, nz,) whereas p.shape = q.shape = (nf, nf,).

    Args:
        z (np.float64): grid points in vertical direction
        p (np.ndarray): array containing parameter p
        q (np.ndarray): array containing parameter q
        z0 (float): centre of region over which transition from non-force-free to force-free takes place
        deltaz (float): width of region over which transition from non-force-free to force-free takes place

    Returns:
        np.ndarray: z-derivative of bar-phi according to Nadol and Neukirch (2025)
    """

    rplus = p / deltaz
    rminus = q / deltaz

    r = rminus / rplus
    d = np.cosh(2.0 * rplus * z0) + np.multiply(r, np.sinh(2.0 * rplus * z0))

    if z - z0 < 0.0:
        return (
            -2.0
            * np.multiply(
                rplus,
                (
                    np.sinh(2.0 * rplus * (z0 - z))
                    + np.multiply(r, np.cosh(2.0 * rplus * (z0 - z)))
                ),
            )
            / d
        )

    else:
        return -2.0 * np.multiply(rminus, np.exp(-2.0 * rminus * (z - z0))) / d
