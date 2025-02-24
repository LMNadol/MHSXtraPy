from __future__ import annotations

import numpy as np
from numba import njit
from scipy.special import hyp2f1, jv


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


# @njit
def phi_nw(
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
