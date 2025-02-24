from __future__ import annotations

import numpy as np


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
