from __future__ import annotations

import numpy as np

from mhsxtrapy._constants import G_SOLAR, KB, MBAR, MU0, T_CORONA, T_PHOTOSPHERE, L
from mhsxtrapy._field import ExtrapolationResult


def btemp_linear(
    field3d: ExtrapolationResult, heights: np.ndarray, temps: np.ndarray
) -> np.ndarray:
    """
    Return background temperature by height in Kelvin using linear interpolation between
    given temperatures (temps) at given heights (heights). temps and heights must be arrays
    of the same length.

    Args:
        field3d (ExtrapolationResult): 3D magnetic field data
        heights (np.ndarray): array of given heights at which interpolation takes place
        temps (np.ndarray): array of given temperature between which is interpolated, same length as heights

    Raises:
        ValueError: In case lengths of heights and temps do not match

    Returns:
        np.ndarray: 1D vertical background temperature profile
    """

    temp = np.zeros_like(field3d.z)

    if len(heights) != len(temps):
        raise ValueError("Number of heights and temperatures do not match")

    h_indices = np.searchsorted(heights, field3d.z, side="right") - 1
    h_indices = np.clip(h_indices, 0, len(heights) - 2)

    frac = (field3d.z - np.array(heights)[h_indices]) / (
        np.array(heights)[h_indices + 1] - np.array(heights)[h_indices]
    )

    temp = (
        np.array(temps)[h_indices]
        + (np.array(temps)[h_indices + 1] - np.array(temps)[h_indices]) * frac
    )
    return temp


def bpressure_linear(
    field3d: ExtrapolationResult, heights: np.ndarray, temps: np.ndarray
) -> np.ndarray:
    """
    Returns background pressure resulting from background temperature btemp_linear.
    Gives background pressure height profile normalised to 1 on the photosphere.
    Need to multiply by BETA0 / 2.0 to normalise to same scale as dpressure. Need to
    then multiply additionally by (B0 * 10**-4) ** 2.0 / MU0 to get into kg/(s^2m).

    Args:
        field3d (ExtrapolationResult): 3D magnetic field data
        heights (np.ndarray): array of given heights at which interpolation takes place
        temps (np.ndarray): array of given temperature between which is interpolated, same length as heights

    Raises:
        ValueError: In case lengths of heights and temps do not match

    Returns:
        np.ndarray: 1D vertical background pressure profile
    """

    if len(heights) != len(temps):
        raise ValueError("Number of heights and temperatures do not match")

    temp = np.zeros_like(field3d.z)

    T0 = temps[np.argmin(np.abs(np.array(heights) - field3d.z0))]

    H = KB * T0 / (MBAR * G_SOLAR) / L

    h_indices = np.searchsorted(heights, field3d.z, side="right") - 1
    h_indices = np.clip(h_indices, 0, len(heights) - 2)

    qj = (np.array(temps)[1:] - np.array(temps)[:-1]) / (
        np.array(heights)[1:] - np.array(heights)[:-1]
    )
    expj = -T0 / (H * qj)
    tempj = (np.abs(np.array(temps)[1:]) / np.array(temps)[:-1]) ** expj
    sum_tempj = np.concatenate(([1.0], np.cumprod(tempj)))

    pro_array = sum_tempj[h_indices]

    q = (np.array(temps)[h_indices + 1] - np.array(temps)[h_indices]) / (
        np.array(heights)[h_indices + 1] - np.array(heights)[h_indices]
    )
    tempz = (
        abs(np.array(temps)[h_indices] + q * (field3d.z - np.array(heights)[h_indices]))
        / np.array(temps)[h_indices]
    ) ** (-T0 / (H * q))
    temp = pro_array * tempz

    return temp


def bdensity_linear(
    field3d: ExtrapolationResult, heights: np.ndarray, temps: np.ndarray
) -> np.ndarray:
    """
    Returns background density resulting from background temperature btemp_linear. Gives background
    density height profile normalised to 1 on the photosphere. Need to multiply by
    BETA0 / (2.0 * H) * T0 / T_PHOTOSPHERE to normalise to same scale as ddensity. Need to then
    multiply additionally by (B0 * 10**-4) ** 2.0 / (MU0 * G_SOLAR * L) to get into kg/(m^3).

    Args:
        field3d (ExtrapolationResult): 3D magnetic field data
        heights (np.ndarray): array of given heights at which interpolation takes place
        temps (np.ndarray): array of given temperature between which is interpolated, same length as heights

    Raises:
        ValueError: In case lengths of heights and temps do not match

    Returns:
        np.ndarray: 1D vertical background density profile
    """

    if len(heights) != len(temps):
        raise ValueError("Number of heights and temperatures do not match")

    temp0 = temps[0]
    dummypres = bpressure_linear(field3d, heights, temps)
    dummytemp = btemp_linear(field3d, heights, temps)

    return dummypres / dummytemp * temp0


def fpressure_linear(
    field3d: ExtrapolationResult, heights: np.ndarray, temps: np.ndarray
) -> np.ndarray:
    """
    Returns full pressure in kg/(s^2m) described by equation (14) in Neukirch and Wiegelmann (2019)
    from linear interpolated background temperature. Multiply by (B0 * 10**-4) ** 2.0 / MU0 to get
    in normalised scale as dpressure.

    Args:
        field3d (ExtrapolationResult): 3D magnetic field data
        heights (np.ndarray): array of given heights at which interpolation takes place
        temps (np.ndarray): array of given temperature between which is interpolated, same length as heights

    Returns:
        np.ndarray: 3D full pressure, sum of background and variation
    """

    bp_matrix = np.zeros_like(field3d.dpressure)
    bp_matrix[:, :, :] = bpressure_linear(field3d, heights, temps)

    return (
        (field3d.BETA0 / 2.0 * bp_matrix + field3d.dpressure)
        * (field3d.B0 * 10**-4) ** 2.0
        / MU0
    )


def fdensity_linear(
    field3d: ExtrapolationResult, heights: np.ndarray, temps: np.ndarray
) -> np.ndarray:
    """
    Returns full density in kg/(m^3) described by equation (15) in Neukirch and Wiegelmann (2019)
    resulting from linearly interpolated background temperature. Divide by
    (B0 * 10**-4) ** 2.0 / (MU0 * G_SOLAR * L) to get in normalised scale as ddensity.

    Args:
        field3d (ExtrapolationResult): 3D magnetic field data
        heights (np.ndarray): array of given heights at which interpolation takes place
        temps (np.ndarray): array of given temperature between which is interpolated, same length as heights

    Returns:
        np.ndarray: 3D full density, sum of background and variation
    """

    T0 = temps[np.argmin(np.abs(np.array(heights) - field3d.z0))]

    H = KB * T0 / (MBAR * G_SOLAR) / L

    bd_matrix = np.zeros_like(field3d.ddensity)
    bd_matrix[:, :, :] = bdensity_linear(field3d, heights, temps)

    return (
        (field3d.BETA0 / (2.0 * H) * T0 / T_PHOTOSPHERE * bd_matrix + field3d.ddensity)
        * (field3d.B0 * 10**-4) ** 2.0
        / (MU0 * G_SOLAR * L)
    )


def btemp_tanh(field: ExtrapolationResult) -> np.ndarray:
    """
    Returns background temperature in Kelvin by height following a hyperbolic tangent height
    profile. From given photospheric and coronal temperatures T0 (temp at z0) and T1 are
    calculated as coefficients for the hyperbolic tangent temperature profile. Current values:
    T_PHOTOSPHERE = 5600.0 Kelvin
    T_CORONA = 2.0 * 10.0**6 Kelvin

    Returns:
        np.ndarray: vertical background temperature profile
    """

    if field.z0 is None or field.deltaz is None:
        raise ValueError(
            "z0 and deltaz must not be None to compute background temperature."
        )

    T0 = (T_PHOTOSPHERE + T_CORONA * np.tanh(field.z0 / field.deltaz)) / (
        1.0 + np.tanh(field.z0 / field.deltaz)
    )
    T1 = (T_CORONA - T_PHOTOSPHERE) / (1.0 + np.tanh(field.z0 / field.deltaz))

    return T0 + T1 * np.tanh((field.z - field.z0) / field.deltaz)


def bpressure_tanh(field: ExtrapolationResult) -> np.ndarray:
    """
    Returns background pressure resulting from background temperature btemp.
    Gives background pressure height profile normalised to 1 on the photosphere.
    Need to multiply by BETA0 / 2.0 to normalise to same scale as dpressure. Need to
    then multiply additionally by (B0 * 10**-4) ** 2.0 / MU0 to get into kg/(s^2m).

    Returns:
        np.ndarray: vertical background pressure profile
    """

    if field.z0 is None or field.deltaz is None:
        raise ValueError(
            "z0 and deltaz must not be None to compute background pressure."
        )

    T0 = (T_PHOTOSPHERE + T_CORONA * np.tanh(field.z0 / field.deltaz)) / (
        1.0 + np.tanh(field.z0 / field.deltaz)
    )  # in Kelvin
    T1 = (T_CORONA - T_PHOTOSPHERE) / (
        1.0 + np.tanh(field.z0 / field.deltaz)
    )  # in Kelvin
    H = KB * T0 / (MBAR * G_SOLAR) / L  # in m

    q1 = field.deltaz / (2.0 * H * (1.0 + T1 / T0))
    q2 = field.deltaz / (2.0 * H * (1.0 - T1 / T0))
    q3 = field.deltaz * (T1 / T0) / (H * (1.0 - (T1 / T0) ** 2))

    p1 = (
        2.0
        * np.exp(-2.0 * (field.z - field.z0) / field.deltaz)
        / (1.0 + np.exp(-2.0 * (field.z - field.z0) / field.deltaz))
        / (1.0 + np.tanh(field.z0 / field.deltaz))
    )
    p2 = (1.0 - np.tanh(field.z0 / field.deltaz)) / (
        1.0 + np.tanh((field.z - field.z0) / field.deltaz)
    )
    p3 = (1.0 + T1 / T0 * np.tanh((field.z - field.z0) / field.deltaz)) / (
        1.0 - T1 / T0 * np.tanh(field.z0 / field.deltaz)
    )

    return (p1**q1) * (p2**q2) * (p3**q3)


def bdensity_tanh(field: ExtrapolationResult) -> np.ndarray:
    """
    Returns background density resulting from background temperature btemp. Gives background
    density height profile normalised to 1 on the photosphere. Need to multiply by
    BETA0 / (2.0 * H) * T0 / T_PHOTOSPHERE to normalise to same scale as ddensity. Need to then
    multiply additionally by (B0 * 10**-4) ** 2.0 / (MU0 * G_SOLAR * L) to get into kg/(m^3).

    Returns:
        np.ndarray: vertical background density profile
    """

    if field.z0 is None or field.deltaz is None:
        raise ValueError(
            "z0 and deltaz must not be None to compute background density."
        )

    T0 = (T_PHOTOSPHERE + T_CORONA * np.tanh(field.z0 / field.deltaz)) / (
        1.0 + np.tanh(field.z0 / field.deltaz)
    )  # in Kelvin
    T1 = (T_CORONA - T_PHOTOSPHERE) / (
        1.0 + np.tanh(field.z0 / field.deltaz)
    )  # in Kelvin

    temp0 = T0 - T1 * np.tanh(field.z0 / field.deltaz)  # in Kelvin
    dummypres = bpressure_tanh(field)
    dummytemp = btemp_tanh(field) / temp0  # normalised

    return dummypres / dummytemp


def fpressure_tanh(field: ExtrapolationResult) -> np.ndarray:
    """
    Returns full pressure described by equation (14) in Neukirch and Wiegelmann (2019) in
    normalised scale. Multiply by (B0 * 10**-4) ** 2.0 / MU0 to get into kg/(s^2m).

    Returns:
        np.ndarray: full 3D pressure, sum of background and variation
    """

    bp_matrix = np.zeros_like(field.dpressure)
    bp_matrix[:, :, :] = bpressure_tanh(field)

    return (
        field.BETA0 / 2.0 * bp_matrix + field.dpressure
    )  # * (B0 * 10**-4) ** 2.0 / MU0


def fdensity(field: ExtrapolationResult) -> np.ndarray:
    """
    Returns full density described by equation (15) in Neukirch and Wiegelmann (2019) in
    normalised scale. Multiply by (B0 * 10**-4) ** 2.0 / (MU0 * G_SOLAR * L) to get
    into kg/(m^3).

    Returns:
        np.ndarray: full 3D density, sum of background and variation
    """

    if field.z0 is None or field.deltaz is None or field.b is None:
        raise ValueError("z0, deltaz, and b must not be None to compute full density.")

    T0 = (T_PHOTOSPHERE + T_CORONA * np.tanh(field.z0 / field.deltaz)) / (
        1.0 + np.tanh(field.z0 / field.deltaz)
    )
    H = KB * T0 / (MBAR * G_SOLAR) / L

    bd_matrix = np.zeros_like(field.ddensity)
    bd_matrix[:, :, :] = bdensity_tanh(field)
    return (
        field.BETA0 / (2.0 * H) * T0 / T_PHOTOSPHERE * bd_matrix + field.ddensity
    )  #  *(B0 * 10**-4) ** 2.0 / (MU0 * G_SOLAR * L)
