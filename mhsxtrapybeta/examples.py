import numpy as np


def multipole(x: np.float64, y: np.float64) -> np.float64:
    """
    Returns value of Multipole-VonMises distribution at given x and y inspired by Neukirch and Wiegelmann (2019) using a Von Mises distribution.
    More details can be found in Nadol and Neukirch (2025).

    Locations of sinks and sources as well as their radii can be adjusted through mu_x, mu_y and kappa_x, kappa_y, respectively.
    """

    xx = np.pi * (x - 1.0)
    yy = np.pi * (y - 1.0)

    mu_x1 = 1.0
    mu_y1 = -1.0

    mu_x2 = -1.2
    mu_y2 = -1.2

    mu_x3 = -2.4
    mu_y3 = 1.9

    mu_x4 = 2.1
    mu_y4 = -1.6

    mu_x5 = -1.5
    mu_y5 = 1.2

    mu_x6 = 2.5
    mu_y6 = 0.0

    mu_x7 = 0.0
    mu_y7 = -2.0

    mu_x8 = -1.0
    mu_y8 = -2.4

    mu_x9 = -1.0
    mu_y9 = 2.4

    kappa_x1 = 10.0
    kappa_y1 = 10.0

    kappa_x2 = 10.0
    kappa_y2 = 10.0

    kappa_x3 = 10.0
    kappa_y3 = 10.0

    kappa_x4 = 10.0
    kappa_y4 = 10.0

    kappa_x5 = 10.0
    kappa_y5 = 10.0

    kappa_x6 = 10.0
    kappa_y6 = 10.0

    kappa_x7 = 10.0
    kappa_y7 = 10.0

    kappa_x8 = 10.0
    kappa_y8 = 10.0

    kappa_x9 = 10.0
    kappa_y9 = 10.0

    return (
        np.exp(kappa_x1 * np.cos(xx - mu_x1))
        / (2.0 * np.pi * np.i0(kappa_x1))
        * np.exp(kappa_y1 * np.cos(yy - mu_y1))
        / (2.0 * np.pi * np.i0(kappa_y1))
        - np.exp(kappa_x2 * np.cos(xx - mu_x2))
        / (2.0 * np.pi * np.i0(kappa_x2))
        * np.exp(kappa_y2 * np.cos(yy - mu_y2))
        / (2.0 * np.pi * np.i0(kappa_y2))
        + np.exp(kappa_x3 * np.cos(xx - mu_x3))
        / (2.0 * np.pi * np.i0(kappa_x3))
        * np.exp(kappa_y3 * np.cos(yy - mu_y3))
        / (2.0 * np.pi * np.i0(kappa_y3))
        + np.exp(kappa_x4 * np.cos(xx - mu_x4))
        / (2.0 * np.pi * np.i0(kappa_x4))
        * np.exp(kappa_y4 * np.cos(yy - mu_y4))
        / (2.0 * np.pi * np.i0(kappa_y4))
        - np.exp(kappa_x5 * np.cos(xx - mu_x5))
        / (2.0 * np.pi * np.i0(kappa_x5))
        * np.exp(kappa_y5 * np.cos(yy - mu_y5))
        / (2.0 * np.pi * np.i0(kappa_y5))
        - np.exp(kappa_x6 * np.cos(xx - mu_x6))
        / (2.0 * np.pi * np.i0(kappa_x6))
        * np.exp(kappa_y6 * np.cos(yy - mu_y6))
        / (2.0 * np.pi * np.i0(kappa_y6))
        - np.exp(kappa_x7 * np.cos(xx - mu_x7))
        / (2.0 * np.pi * np.i0(kappa_x7))
        * np.exp(kappa_y7 * np.cos(yy - mu_y7))
        / (2.0 * np.pi * np.i0(kappa_y7))
        + np.exp(kappa_x8 * np.cos(xx - mu_x8))
        / (2.0 * np.pi * np.i0(kappa_x8))
        * np.exp(kappa_y8 * np.cos(yy - mu_y8))
        / (2.0 * np.pi * np.i0(kappa_y8))
        - np.exp(kappa_x9 * np.cos(xx - mu_x9))
        / (2.0 * np.pi * np.i0(kappa_x9))
        * np.exp(kappa_y9 * np.cos(yy - mu_y9))
        / (2.0 * np.pi * np.i0(kappa_y9))
    )
