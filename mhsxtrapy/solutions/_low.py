from __future__ import annotations

import numpy as np
from scipy.special import jv

from ._base import BaseSolution


class LowSolution(BaseSolution):

    def __init__(self, kappa: float, a: float):
        self.kappa = kappa
        self.a = a

    def phi(self, z, p, q):
        return phi_low(z, p, q, self.kappa)

    def dphidz(self, z, p, q):
        return dphidz_low(z, p, q, self.kappa)

    def f(self, z):
        return f_low(z, self.a, self.kappa)

    def dfdz(self, z):
        return dfdz_low(z, self.a, self.kappa)


# @njit
def phi_low(z: np.float64, p: np.float64, q: np.float64, kappa: float) -> np.float64:
    """
    Returns solution of ODE (18) in Neukirch and Wiegelmann (2019)
    using the exponential switch function f(z) = a exp(-kappa z)
    introduced in Low (1991).

    Vectorisation should be possible for z, p and q, of which p and q have to be
    passed as arrays of the same size. Returns array of size (nf, nf, nz,) whereas
    p.shape = q.shape = (nf, nf,).

    Args:
        z (np.float64): grid points in vertical direction
        p (np.ndarray): array containing parameter p
        q (np.ndarray): array containing parameter q
        kappa (float): drop-off parameter

    Returns:
        np.float64: bar-phi according to Low (1991, 1992)
    """

    return jv(p, q * np.exp(-z * kappa / 2.0)) / jv(p, q)


# @njit
def dphidz_low(z: np.float64, p: np.float64, q: np.float64, kappa: float) -> np.float64:
    """
    Returns z-derivative of solution of ODE (18) in Neukirch and Wiegelmann (2019)
    using the exponential switch function f(z) = a exp(-kappa z)
    introduced in Low (1991).

    Vectorisation should be possible for z, p and q, of which p and q have to be
    passed as arrays of the same size. Returns array of size (nf, nf, nz,) whereas
    p.shape = q.shape = (nf, nf,).

    Args:
        z (np.float64): grid points in vertical direction
        p (np.ndarray): array containing parameter p
        q (np.ndarray): array containing parameter q
        kappa (float): drop-off parameter

    Returns:
        np.float64: z-derivative of bar-phi according to Low (1991, 1992)
    """

    return (
        (
            q * np.exp(-z * kappa / 2.0) * jv(p + 1.0, q * np.exp(-z * kappa / 2.0))
            - p * jv(p, q * np.exp(-z * kappa / 2.0))
        )
        * kappa
        / (2.0 * jv(p, q))
    )


def f_low(z: np.ndarray, a: float, kappa: float) -> np.ndarray:
    """
    Height profile of transition non-force-free to force-free
    according to Low (1991, 1992). Vectorisation with z possible,
    returns array of size z.shape.

    Args:
        z (np.ndarray): grid points vertical direction
        a (float): amplitude parameter
        kappa (float): drop-off parameter

    Returns:
        np.ndarray: Height profile of transition non-force-free to force-free with z
    """
    return a * np.exp(-kappa * z)


def dfdz_low(z: np.ndarray, a: float, kappa: float) -> np.ndarray:
    """
    Z-derivative of height profile of transition non-force-free to
    force-free according to Low (1991, 1992). Vectorisation with z possible,
    returns array of size z.shape.

    Args:
        z (np.ndarray): grid points vertical direction
        a (float): amplitude parameter
        kappa (float): drop-off parameter

    Returns:
        np.ndarray: Z-derivative of height profile of transition non-force-free to force-free with z
    """
    return -kappa * a * np.exp(-kappa * z)
