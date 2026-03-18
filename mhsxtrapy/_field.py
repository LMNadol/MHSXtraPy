from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from functools import cached_property

import h5py
import numpy as np

from mhsxtrapy._boundary import Field2dData, FluxBalanceState
from mhsxtrapy._constants import G_SOLAR, KB, MBAR, MU0, P0, T_CORONA, T_PHOTOSPHERE, L
from mhsxtrapy._extrapolation import WhichSolution, b3d
from mhsxtrapy.solutions import get_solution

__all__ = [
    "Field3dData",
]

# TO DO The `field` array stores components as `[By, Bx, Bz]` (indices 0, 1, 2). This is non-standard and error-prone. Either:
# - **(a)** Add named constants: `BY_IDX, BX_IDX, BZ_IDX = 0, 1, 2` and use throughout, or
# - **(b)** Add an accessor: `field3d.bx`, `field3d.by`, `field3d.bz_3d` properties that slice the array.


@dataclass
class Field3dData:
    """
    Dataclass of type Field3dData with the following attributes:
    ------------------------------------------------------------------------------------------------------
    Taken from Field2dData object which Field3dData object is based on:
    nx, ny, nz  :   Dimensions of 3D magnetic field, usually nx and ny determined by magnetogram size,
                    while nz defined by user through height to which extrapolation is carried out.
    nf          :   Number of Fourier modes used in calculation of magnetic field vector, usually
                    nf = min(nx, ny) is taken. To do: split into nfx, nfy, sucht that all possible modes
                    in both directions can be used.
    px, py, pz  :   Pixel sizes in x-, y-, z-direction, in normal length scale (Mm).
    x, y, z     :   1D arrays of grid points on which magnetic field is given with shapes (nx,), (ny,)
                    and (nz,) respectively.
    bz          :   Bottom boundary magentogram of size (ny, nx,). Indexing of vectors done in this order,
                    such that, following intuition, x-direction corresponds to latitudinal extension and
                    y-direction to longitudinal extension of the magnetic field.
    ------------------------------------------------------------------------------------------------------
    New attributes:
    field       :   3D magnetic field vector of size (ny, nx, nz, 3,) which contains magnetic field data in
                    Gauss in the shapes By = field(:, :, :, 0), Bx = field(:, :, :, 1) and
                    Bz = field(:, :, :, 2).
    grad_bz      :   3D vector of size (ny, nx, nz, 3,) containing partial derivatives of Bz in Gauss in the
                    shape Bzdy = field(:, :, :, 0), Bzdx = field(:, :, :, 1) and Bzdz = field(:, :, :, 2).
    a           :   Amplitude parameter of function f(z).
    b           :   Switch parameter of function f(z).
    alpha       :   Poloidal/toroidal ratio parameter in equation (10) of Neukirch and Wiegelmann (2019).
    z0          :   Height around which transtion from non-force-free to force-free takes place.
    deltaz      :   Width of region over which transition from non-force-free to force-free takes place.
    tanh        :   Boolean paramter determining if Low or N+W/N+W-A height profile is used for calculation
                    of plasma pressure, plasma density, current density and Lorentz force.
    ------------------------------------------------------------------------------------------------------

    Raises:
        ValueError: In case flux_balance has wrong value
        ValueError: In case solution and parameter choices do not agree

    Returns:
        _type_: Field3dData object
    """

    nx: int
    ny: int
    nz: int
    nf: int
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    bz: np.ndarray
    field: np.ndarray
    grad_bz: np.ndarray

    flux_balance_state: FluxBalanceState

    alpha: float
    a: float

    solution: WhichSolution

    b: float | None = None

    z0: float | None = None
    deltaz: float | None = None

    kappa: float | None = None

    EUV: np.ndarray | None = None

    def __post_init__(self):
        # Workaround for Python 3.14: @dataclass may not call __set_name__
        # on cached_property descriptors, leaving attrname as None.
        for name, value in type(self).__dict__.items():
            if isinstance(value, cached_property) and value.attrname is None:
                value.__set_name__(type(self), name)

    def __repr__(self) -> str:
        extras = []
        if self.b is not None:
            extras.append(f"b={self.b}")
        if self.z0 is not None:
            extras.append(f"z0={self.z0}")
        if self.deltaz is not None:
            extras.append(f"deltaz={self.deltaz}")
        if self.kappa is not None:
            extras.append(f"kappa={self.kappa}")
        if self.EUV is not None:
            extras.append(f"EUV={self.EUV.shape}")
        extra_str = ", ".join(extras)
        if extra_str:
            extra_str = ", " + extra_str
        return (
            f"Field3dData(nx={self.nx}, ny={self.ny}, nz={self.nz}, nf={self.nf}, "
            f"field={self.field.shape}, solution={self.solution.value}, "
            f"alpha={self.alpha}, a={self.a}, "
            f"flux_balance={self.flux_balance_state.value}{extra_str})"
        )

    def save(self, path):
        """
        Save Field3dData object as HDF5 file.
        """
        if os.path.isdir(path):
            path = os.path.join(path, "field3d.h5")
        with h5py.File(path, "w") as f:
            for name, attribute in self.__dict__.items():
                if isinstance(attribute, np.ndarray):
                    f.create_dataset(name, data=attribute)
                elif isinstance(attribute, Enum):
                    f.attrs[name] = attribute.value
                elif attribute is None:
                    f.attrs[name] = "___NONE___"
                else:
                    f.attrs[name] = attribute

    @classmethod
    def load(cls, path):
        """
        Load Field3dData object from HDF5 file.
        """
        _enum_fields = {
            "solution": WhichSolution,
            "flux_balance_state": FluxBalanceState,
        }
        if os.path.isdir(path):
            path = os.path.join(path, "field3d.h5")
        data = {}
        with h5py.File(path, "r") as f:
            for name in cls.__annotations__:
                if name in f:
                    data[name] = f[name][()]
                elif name in f.attrs:
                    val = f.attrs[name]
                    if isinstance(val, str) and val == "___NONE___":
                        val = None
                    elif name in _enum_fields:
                        val = _enum_fields[name](val)
                    data[name] = val
        return cls(**data)

    @cached_property
    def B0(self):
        return self.field[
            :, :, 0, 2
        ].max()  # Gauss background magnetic field strength in 10^-4 kg/(s^2A) = 10^-4 T

    @cached_property
    def PB0(self):
        return (self.B0 * 10**-4) ** 2 / (
            2 * MU0
        )  # magnetic pressure b0**2 / 2mu0 in kg/(s^2m)

    @cached_property
    def BETA0(self):
        return P0 / self.PB0  # Plasma Beta, ration plasma to magnetic pressure

    @cached_property
    def btemp(self) -> np.ndarray:
        """
        Returns background temperature in Kelvin by height following a hyperbolic tangent height
        profile. From given photospheric and coronal temperatures T0 (temp at z0) and T1 are
        calculated as coefficients for the hyperbolic tangent temperature profile. Current values:
        T_PHOTOSPHERE = 5600.0 Kelvin
        T_CORONA = 2.0 * 10.0**6 Kelvin

        Returns:
            np.ndarray: vertical background temperature profile
        """

        if self.z0 is None or self.deltaz is None:
            raise ValueError(
                "z0 and deltaz must not be None to compute background temperature."
            )

        T0 = (T_PHOTOSPHERE + T_CORONA * np.tanh(self.z0 / self.deltaz)) / (
            1.0 + np.tanh(self.z0 / self.deltaz)
        )
        T1 = (T_CORONA - T_PHOTOSPHERE) / (1.0 + np.tanh(self.z0 / self.deltaz))

        return T0 + T1 * np.tanh((self.z - self.z0) / self.deltaz)

    @cached_property
    def bpressure(self) -> np.ndarray:
        """
        Returns background pressure resulting from background temperature btemp.
        Gives background pressure height profile normalised to 1 on the photosphere.
        Need to multiply by BETA0 / 2.0 to normalise to same scale as dpressure. Need to
        then multiply additionally by (B0 * 10**-4) ** 2.0 / MU0 to get into kg/(s^2m).

        Returns:
            np.ndarray: vertical background pressure profile
        """

        if self.z0 is None or self.deltaz is None:
            raise ValueError(
                "z0 and deltaz must not be None to compute background pressure."
            )

        T0 = (T_PHOTOSPHERE + T_CORONA * np.tanh(self.z0 / self.deltaz)) / (
            1.0 + np.tanh(self.z0 / self.deltaz)
        )  # in Kelvin
        T1 = (T_CORONA - T_PHOTOSPHERE) / (
            1.0 + np.tanh(self.z0 / self.deltaz)
        )  # in Kelvin
        H = KB * T0 / (MBAR * G_SOLAR) / L  # in m

        q1 = self.deltaz / (2.0 * H * (1.0 + T1 / T0))
        q2 = self.deltaz / (2.0 * H * (1.0 - T1 / T0))
        q3 = self.deltaz * (T1 / T0) / (H * (1.0 - (T1 / T0) ** 2))

        p1 = (
            2.0
            * np.exp(-2.0 * (self.z - self.z0) / self.deltaz)
            / (1.0 + np.exp(-2.0 * (self.z - self.z0) / self.deltaz))
            / (1.0 + np.tanh(self.z0 / self.deltaz))
        )
        p2 = (1.0 - np.tanh(self.z0 / self.deltaz)) / (
            1.0 + np.tanh((self.z - self.z0) / self.deltaz)
        )
        p3 = (1.0 + T1 / T0 * np.tanh((self.z - self.z0) / self.deltaz)) / (
            1.0 - T1 / T0 * np.tanh(self.z0 / self.deltaz)
        )

        return (p1**q1) * (p2**q2) * (p3**q3)

    @cached_property
    def bdensity(self) -> np.ndarray:
        """
        Returns background density resulting from background temperature btemp. Gives background
        density height profile normalised to 1 on the photosphere. Need to multiply by
        BETA0 / (2.0 * H) * T0 / T_PHOTOSPHERE to normalise to same scale as ddensity. Need to then
        multiply additionally by (B0 * 10**-4) ** 2.0 / (MU0 * G_SOLAR * L) to get into kg/(m^3).

        Returns:
            np.ndarray: vertical background density profile
        """

        if self.z0 is None or self.deltaz is None:
            raise ValueError(
                "z0 and deltaz must not be None to compute background density."
            )

        T0 = (T_PHOTOSPHERE + T_CORONA * np.tanh(self.z0 / self.deltaz)) / (
            1.0 + np.tanh(self.z0 / self.deltaz)
        )  # in Kelvin
        T1 = (T_CORONA - T_PHOTOSPHERE) / (
            1.0 + np.tanh(self.z0 / self.deltaz)
        )  # in Kelvin

        temp0 = T0 - T1 * np.tanh(self.z0 / self.deltaz)  # in Kelvin
        dummypres = self.bpressure  # normalised
        dummytemp = self.btemp / temp0  # normalised

        return dummypres / dummytemp

    @cached_property
    def dpressure(self) -> np.ndarray:
        """
        Calculate variation in pressure described by equation (30) in Neukirch and Wiegelmann (2019).
        Normalised to same scale as BETA0 / 2.0 * bpressure (see L Nadol PhD thesis for details).

        Raises:
            ValueError: In case flux_balance has wrong value
            ValueError: In case solution and parameter choices do not agree

        Returns:
            np.ndarray: 3D variation in pressure
        """

        sol = get_solution(
            self.solution,
            a=self.a,
            b=self.b,
            z0=self.z0,
            deltaz=self.deltaz,
            kappa=self.kappa,
        )

        if self.flux_balance_state == FluxBalanceState.BALANCED:
            bz_matrix = self.field[:, :, :, 2]
        elif self.flux_balance_state == FluxBalanceState.UNBALANCED:
            bz_matrix = self.field[
                self.ny : 2 * self.ny, self.nx : 2 * self.nx, :, 2
            ]  # in Gauss
        else:
            raise ValueError(
                f"Invalid flux_balance_state: {self.flux_balance_state}. Expected 'BALANCED' or 'UNBALANCED'."
            )

        z_matrix = np.zeros_like(bz_matrix)
        z_matrix[:, :, :] = self.z

        return -sol.f(z_matrix) / 2.0 * bz_matrix**2.0 / self.B0**2.0

    @cached_property
    def ddensity(self) -> np.ndarray:
        """
        Calculate variation in pressure described by equation (31) in Neukirch and Wiegelmann (2019).
        Normalised to same scale as BETA0 / (2.0 * H) * T0 / T_PHOTOSPHERE * bdensity (see L Nadol
        PhD thesis for details).

        Raises:
            ValueError: In case flux_balance has wrong value
            ValueError: In case solution and parameter choices do not agree

        Returns:
            np.ndarray: 3D variation in density
        """
        sol = get_solution(
            self.solution,
            a=self.a,
            b=self.b,
            z0=self.z0,
            deltaz=self.deltaz,
            kappa=self.kappa,
        )

        if self.flux_balance_state == FluxBalanceState.BALANCED:
            bz_matrix = self.field[:, :, :, 2]
            bdotbz_matrix = (
                self.field[:, :, :, 0] * self.grad_bz[:, :, :, 0]
                + self.field[:, :, :, 1] * self.grad_bz[:, :, :, 1]
                + self.field[:, :, :, 2] * self.grad_bz[:, :, :, 2]
            )  # in Gauss**2
        elif self.flux_balance_state == FluxBalanceState.UNBALANCED:
            bz_matrix = self.field[self.ny : 2 * self.ny, self.nx : 2 * self.nx, :, 2]
            bdotbz_matrix = (
                self.field[self.ny : 2 * self.ny, self.nx : 2 * self.nx, :, 0]
                * self.grad_bz[self.ny : 2 * self.ny, self.nx : 2 * self.nx, :, 0]
                + self.field[self.ny : 2 * self.ny, self.nx : 2 * self.nx, :, 1]
                * self.grad_bz[self.ny : 2 * self.ny, self.nx : 2 * self.nx, :, 1]
                + self.field[self.ny : 2 * self.ny, self.nx : 2 * self.nx, :, 2]
                * self.grad_bz[self.ny : 2 * self.ny, self.nx : 2 * self.nx, :, 2]
            )  # in Gauss**2
        else:
            raise ValueError(
                f"Invalid flux_balance_state: {self.flux_balance_state}. Expected 'BALANCED' or 'UNBALANCED'."
            )

        z_matrix = np.zeros_like(bz_matrix)
        z_matrix[:, :, :] = self.z

        return (
            sol.dfdz(z_matrix) / 2.0 * bz_matrix**2 / self.B0**2
            + sol.f(z_matrix) * bdotbz_matrix / self.B0**2
        )

    @cached_property
    def fpressure(self) -> np.ndarray:
        """
        Returns full pressure described by equation (14) in Neukirch and Wiegelmann (2019) in
        normalised scale. Multiply by (B0 * 10**-4) ** 2.0 / MU0 to get into kg/(s^2m).

        Returns:
            np.ndarray: full 3D pressure, sum of background and variation
        """

        bp_matrix = np.zeros_like(self.dpressure)
        bp_matrix[:, :, :] = self.bpressure

        return (
            self.BETA0 / 2.0 * bp_matrix + self.dpressure
        )  # * (B0 * 10**-4) ** 2.0 / MU0

    @cached_property
    def fdensity(self) -> np.ndarray:
        """
        Returns full density described by equation (15) in Neukirch and Wiegelmann (2019) in
        normalised scale. Multiply by (B0 * 10**-4) ** 2.0 / (MU0 * G_SOLAR * L) to get
        into kg/(m^3).

        Returns:
            np.ndarray: full 3D density, sum of background and variation
        """

        if self.z0 is None or self.deltaz is None or self.b is None:
            raise ValueError(
                "z0, deltaz, and b must not be None to compute full density."
            )

        T0 = (T_PHOTOSPHERE + T_CORONA * np.tanh(self.z0 / self.deltaz)) / (
            1.0 + np.tanh(self.z0 / self.deltaz)
        )
        H = KB * T0 / (MBAR * G_SOLAR) / L

        bd_matrix = np.zeros_like(self.ddensity)
        bd_matrix[:, :, :] = self.bdensity

        return (
            self.BETA0 / (2.0 * H) * T0 / T_PHOTOSPHERE * bd_matrix + self.ddensity
        )  #  *(B0 * 10**-4) ** 2.0 / (MU0 * G_SOLAR * L)

    @cached_property
    def j3d(self) -> np.ndarray:
        """
        Calculate current density at all grid points. For details see function j3d below.

        Returns:
            np.ndarray: 3D current density
        """

        sol = get_solution(
            self.solution,
            a=self.a,
            b=self.b,
            z0=self.z0,
            deltaz=self.deltaz,
            kappa=self.kappa,
        )

        j = np.zeros_like(self.field)

        j[:, :, :, 2] = self.alpha * self.field[:, :, :, 2] * 10**-4

        f_matrix = np.zeros_like(self.grad_bz[:, :, :, 0])
        f_matrix[:, :, :] = sol.f(self.z)

        j[:, :, :, 1] = (
            self.alpha * self.field[:, :, :, 1] * 10**-4
            + f_matrix * self.grad_bz[:, :, :, 0] * 10**-4
        )

        j[:, :, :, 0] = (
            self.alpha * self.field[:, :, :, 0] * 10**-4
            - f_matrix * self.grad_bz[:, :, :, 1] * 10**-4
        )
        return j / MU0 * L

    @cached_property
    def lf3d(self) -> np.ndarray:
        """
        Calculate Lorentz force at all grid points. For details see function lf3d below.

        Returns:
            np.ndarray: 3D Lorentz force
        """

        j = self.j3d

        lf = np.zeros_like(self.field)

        lf[:, :, :, 0] = (
            j[:, :, :, 2] * self.field[:, :, :, 1] * 10**-4
            - j[:, :, :, 1] * self.field[:, :, :, 2] * 10**-4
        )
        lf[:, :, :, 1] = (
            j[:, :, :, 0] * self.field[:, :, :, 2] * 10**-4
            - j[:, :, :, 2] * self.field[:, :, :, 0] * 10**-4
        )
        lf[:, :, :, 2] = (
            j[:, :, :, 1] * self.field[:, :, :, 0] * 10**-4
            - j[:, :, :, 0] * self.field[:, :, :, 1] * 10**-4
        )

        return lf


def calculate_magfield(
    field2d: Field2dData,
    alpha: float,
    a: float,
    which_solution: WhichSolution,
    b: float | None = None,
    z0: float | None = None,
    deltaz: float | None = None,
    kappa: float | None = None,
) -> Field3dData:
    """
    Create Field3dData object from Field2dData object and choosen paramters using mhsxtrapy.b3d.b3d.

    Args:
        field2d (Field2dData): boundary condition
        alpha (float): force-free parameter
        a (float): amplitude parameter
        which_solution (WhichSolution): Enum to decide which solution to use, Low (1992), Neukirch and Wiegelmann (2019) or
            Nadol and Neukirch (2025)
        b (float | None, optional): Switch-off parameter for NW and NN solution. Defaults to None.
        z0 (float | None, optional): Centre of region over which transition from non-force-free to force-free takes place. Defaults to None.
        deltaz (float | None, optional): Width of region over which transition from non-force-free to force-free takes place. Defaults to None.
        kappa (float | None, optional): Drop-off parameter for Low solution. Defaults to None.

    Returns:
        Field3dData: Field3dData object
    """

    mf3d, dbz3d = b3d(field2d, alpha, a, which_solution, b, z0, deltaz, kappa)

    data = Field3dData(
        nx=field2d.nx,
        ny=field2d.ny,
        nz=field2d.nz,
        nf=field2d.nf,
        x=field2d.x,
        y=field2d.y,
        z=field2d.z,
        bz=field2d.bz,
        field=mf3d,
        grad_bz=dbz3d,
        flux_balance_state=field2d.flux_balance_state,
        alpha=alpha,
        a=a,
        solution=which_solution,
        b=b,
        z0=z0,
        deltaz=deltaz,
        kappa=kappa,
        EUV=field2d.EUV,
    )

    return data


# def j3d(field3d: Field3dData) -> np.ndarray:
#     """
#     Returns current density, calucated from magnetic field as j = (alpha B + curl(0,0,f(z)Bz))/ mu0.
#     In A/m^2.

#     Args:
#         field3d (Field3dData): magnetic field data

#     Raises:
#         ValueError: In case solution and parameter choices do not match

#     Returns:
#         np.ndarray: current density
#     """

#     sol = get_solution(
#         field3d.solution,
#         a=field3d.a,
#         b=field3d.b,
#         z0=field3d.z0,
#         deltaz=field3d.deltaz,
#         kappa=field3d.kappa,
#     )

#     j = np.zeros_like(field3d.field)

#     j[:, :, :, 2] = field3d.alpha * field3d.field[:, :, :, 2] * 10**-4

#     f_matrix = np.zeros_like(field3d.grad_bz[:, :, :, 0])
#     f_matrix[:, :, :] = sol.f(field3d.z)

#     j[:, :, :, 1] = (
#         field3d.alpha * field3d.field[:, :, :, 1] * 10**-4
#         + f_matrix * field3d.grad_bz[:, :, :, 0] * 10**-4
#     )

#     j[:, :, :, 0] = (
#         field3d.alpha * field3d.field[:, :, :, 0] * 10**-4
#         - f_matrix * field3d.grad_bz[:, :, :, 1] * 10**-4
#     )
#     return j / MU0 * L


# def lf3d(field3d: Field3dData) -> np.ndarray:
#     """
#     Returns Lorentz force calculated from j x B.
#     In kg/sm^2.

#     Args:
#         field3d (Field3dData): magnetic field data

#     Returns:
#         np.ndarray: 3D Lorentz force
#     """

#     j = field3d.j3d

#     lf = np.zeros_like(field3d.field)

#     lf[:, :, :, 0] = (
#         j[:, :, :, 2] * field3d.field[:, :, :, 1] * 10**-4
#         - j[:, :, :, 1] * field3d.field[:, :, :, 2] * 10**-4
#     )
#     lf[:, :, :, 1] = (
#         j[:, :, :, 0] * field3d.field[:, :, :, 2] * 10**-4
#         - j[:, :, :, 2] * field3d.field[:, :, :, 0] * 10**-4
#     )
#     lf[:, :, :, 2] = (
#         j[:, :, :, 1] * field3d.field[:, :, :, 0] * 10**-4
#         - j[:, :, :, 0] * field3d.field[:, :, :, 1] * 10**-4
#     )

#     return lf
