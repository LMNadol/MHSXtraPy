from __future__ import annotations

import numpy as np
from scipy.special import hyp2f1

from .base import Solution


class NeuWieSolution(Solution):

    def phi(self, z, p, q, z0, deltaz):
        return phi_nw(z, p, q, z0, deltaz)

    def dphidz(self, z, p, q, z0, deltaz):
        return dphidz_nw(z, p, q, z0, deltaz)

    def f(self, z, z0, deltaz, a, b):
        return f_nw(z, z0, deltaz, a, b)

    def dfdz(self, z, z0, deltaz, a, b):
        return dfdz_nw(z, z0, deltaz, a, b)


# @njit
def phi_nw(
    z: np.float64,
    p: np.ndarray,
    q: np.ndarray,
    z0: float,
    deltaz: float,
) -> np.float64:
    """
    Args:
        z (np.float64): grid points in vertical direction
        p (np.ndarray): array containing parameter p
        q (np.ndarray): array containing parameter q
        z0 (float): centre of region over which transition from non-force-free to force-free takes place
        deltaz (float): width of region over which transition from non-force-free to force-free takes place

    Returns:
        np.float64: bar-phi according to Neukirch and Wiegelmann (2019)
    """
    w = (z - z0) / deltaz
    eta_d = 1.0 / (1.0 + np.exp(2.0 * w))
    phi = eta_d**q * (1 - eta_d) ** p * hyp2f1(p + q + 1, p + q, 2 * q + 1, eta_d)

    w0 = -z0 / deltaz
    eta0 = 1.0 / (1.0 + np.exp(2.0 * w0))
    phi0 = eta0**q * (1 - eta0) ** p * hyp2f1(p + q + 1, p + q, 2 * q + 1, eta0)

    return phi / phi0


# @njit
def dphidz_nw(
    z: np.float64,
    p: np.ndarray[np.float64, np.dtype[np.float64]],
    q: np.ndarray[np.float64, np.dtype[np.float64]],
    z0: float,
    deltaz: float,
) -> np.float64:
    """
    Args:
        z (np.float64): grid points in vertical direction
        p (np.ndarray): array containing parameter p
        q (np.ndarray): array containing parameter q
        z0 (float): centre of region over which transition from non-force-free to force-free takes place
        deltaz (float): width of region over which transition from non-force-free to force-free takes place

    Returns:
        np.float64: z-derivative of bar-phi according to Neukirch and Wiegelmann (2019)
    """

    w = (z - z0) / deltaz
    eta_d = 1.0 / (1.0 + np.exp(2.0 * w))

    dphi = (
        q * eta_d**q * (1 - eta_d) ** (p + 1) - p * eta_d ** (q + 1) * (1 - eta_d) ** p
    ) * hyp2f1(p + q + 1, p + q, 2 * q + 1, eta_d) + (p + q + 1) * (p + q) / (
        2 * q + 1
    ) * eta_d ** (
        q + 1
    ) * (
        1 - eta_d
    ) ** (
        p + 1
    ) * hyp2f1(
        p + q + 2, p + q + 1, 2 * q + 2, eta_d
    )

    w0 = -z0 / deltaz
    eta0 = 1.0 / (1.0 + np.exp(2.0 * w0))
    phi0 = eta0**q * (1 - eta0) ** p * hyp2f1(p + q + 1, p + q, 2 * q + 1, eta0)

    return -2.0 / deltaz * dphi / phi0


def f_nw(z: np.ndarray, z0: float, deltaz: float, a: float, b: float) -> np.ndarray:
    """
    Height profile of transition non-force-free to force-free
    according to Neukirch and Wiegelmann (2019). Vectorisation with z possible,
    returns array of size z.shape.

    Args:
        z (np.ndarray): grid points vertical direction
        z0 (float): centre of region over which transition from non-force-free to force-free takes place
        deltaz (float): width of region over which transition from non-force-free to force-free takes place
        a (float): amplitude parameter
        b (float): switch-off parameter

    Returns:
        np.ndarray: Height profile of transition non-force-free to force-free with z
    """

    return a * (1.0 - b * np.tanh((z - z0) / deltaz))


def dfdz_nw(z: np.ndarray, z0: float, deltaz: float, a: float, b: float) -> np.ndarray:
    """
    Z-derivative of height profile of transition non-force-free to
    force-free according to Neukirch and Wiegelmann (2019). Vectorisation with z possible,
    returns array of size z.shape.

    Args:
        z (np.ndarray): grid points vertical direction
        z0 (float): centre of region over which transition from non-force-free to force-free takes place
        deltaz (float): width of region over which transition from non-force-free to force-free takes place
        a (float): amplitude parameter
        b (float): switch-off parameter

    Returns:
        np.ndarray: Z-derivative of height profile of transition non-force-free to force-free with z
    """

    return -a * b / (deltaz * np.cosh((z - z0) / deltaz) ** 2)
