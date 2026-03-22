"""
Reproducible benchmarks for MHSXtraPy performance tracking.
pytest-benchmark suite that can be run via:

    pytest tests/test_benchmarks.py --benchmark-only

benchmarks cover:
- phi / dphidz computation for each solution (Nadol-Neukirch, Neukirch-Wiegelmann)
- Full 3D extrapolation (_extrapolate_3d) for both flux-balance states and both solutions
"""

from __future__ import annotations

import numpy as np
import pytest

from mhsxtrapy._boundary import BoundaryData
from mhsxtrapy._extrapolation import _extrapolate_3d
from mhsxtrapy.examples import dipole
from mhsxtrapy.solutions._nadol_neukirch import dphidz_nn, phi_nn
from mhsxtrapy.solutions._neukirch_wiegelmann import dphidz_nw, phi_nw
from mhsxtrapy.types import FluxBalanceState, WhichSolution

nx, ny, nz, nf = 200, 200, 400, 200
xmin, xmax = 0.0, 20.0
ymin, ymax = 0.0, 20.0
zmin, zmax = 0.0, 20.0

a = 0.22
alpha = 0.05
b = 1.0
z0 = 2.0
deltaz = 0.2


@pytest.fixture(scope="module")
def grid():
    """Pre-compute pixel sizes, coordinate arrays, and dipole boundary."""
    px = (xmax - xmin) / nx
    py = (ymax - ymin) / ny
    pz = (zmax - zmin) / nz

    x_arr = np.linspace(xmin, xmax, nx, dtype=np.float64)
    y_arr = np.linspace(ymin, ymax, ny, dtype=np.float64)
    z_arr = np.linspace(zmin, zmax, nz, dtype=np.float64)

    # Vectorised dipole boundary
    xx, yy = np.meshgrid(x_arr, y_arr)
    data_bz = np.vectorize(dipole)(xx, yy)

    return {
        "px": px,
        "py": py,
        "pz": pz,
        "x_arr": x_arr,
        "y_arr": y_arr,
        "z_arr": z_arr,
        "data_bz": data_bz,
    }


@pytest.fixture(scope="module")
def wavenumber_arrays(grid):
    """Pre-compute p_arr and q_arr used by phi / dphidz benchmarks."""
    length_scale = 2.0

    length_scale_x = 2.0 * nx * grid["px"]
    length_scale_y = 2.0 * ny * grid["py"]

    length_scale_x_norm = length_scale_x / length_scale
    length_scale_y_norm = length_scale_y / length_scale

    kx_arr = np.arange(nf) * np.pi / length_scale_x_norm
    ky_arr = np.arange(nf) * np.pi / length_scale_y_norm
    one_arr = np.ones(nf)

    k2_arr = np.outer(ky_arr**2, one_arr) + np.outer(one_arr, kx_arr**2)
    k2_arr[0, 0] = (np.pi / length_scale_x_norm) ** 2 + (
        np.pi / length_scale_y_norm
    ) ** 2
    k2_arr[1, 0] = k2_arr[0, 0]
    k2_arr[0, 1] = k2_arr[0, 0]

    p_arr = 0.5 * deltaz * np.sqrt(k2_arr[:nf, :nf] * (1.0 - a - a * b) - alpha**2)
    q_arr = 0.5 * deltaz * np.sqrt(k2_arr[:nf, :nf] * (1.0 - a + a * b) - alpha**2)

    return {"p_arr": p_arr, "q_arr": q_arr, "z_arr": grid["z_arr"]}


@pytest.fixture(scope="module")
def boundary_unbalanced(grid):
    return BoundaryData(
        nx,
        ny,
        nz,
        nf,
        grid["px"],
        grid["py"],
        grid["pz"],
        grid["x_arr"],
        grid["y_arr"],
        grid["z_arr"],
        grid["data_bz"],
        flux_balance_state=FluxBalanceState.UNBALANCED,
    )


@pytest.fixture(scope="module")
def boundary_balanced(grid):
    return BoundaryData(
        nx,
        ny,
        nz,
        nf,
        grid["px"],
        grid["py"],
        grid["pz"],
        grid["x_arr"],
        grid["y_arr"],
        grid["z_arr"],
        grid["data_bz"],
        flux_balance_state=FluxBalanceState.BALANCED,
    )


def _compute_phi_over_z(phi_func, z_arr, p_arr, q_arr):
    """Helper: evaluate phi (or dphidz) at every z-level."""
    result = np.empty((nf, nf, nz))
    for iz, z in enumerate(z_arr):
        result[:, :, iz] = phi_func(z, p_arr, q_arr, z0, deltaz)
    return result


@pytest.mark.benchmark(group="phi")
def test_phi_nadol_neukirch(benchmark, wavenumber_arrays):
    benchmark(_compute_phi_over_z, phi_nn, **wavenumber_arrays)


@pytest.mark.benchmark(group="phi")
def test_phi_neukirch_wiegelmann(benchmark, wavenumber_arrays):
    benchmark(_compute_phi_over_z, phi_nw, **wavenumber_arrays)


@pytest.mark.benchmark(group="dphidz")
def test_dphidz_nadol_neukirch(benchmark, wavenumber_arrays):
    benchmark(_compute_phi_over_z, dphidz_nn, **wavenumber_arrays)


@pytest.mark.benchmark(group="dphidz")
def test_dphidz_neukirch_wiegelmann(benchmark, wavenumber_arrays):
    benchmark(_compute_phi_over_z, dphidz_nw, **wavenumber_arrays)


def _run_extrapolation(boundary, solution):
    return _extrapolate_3d(
        boundary,
        a=a,
        b=b,
        alpha=alpha,
        z0=z0,
        deltaz=deltaz,
        solution=solution,
    )


@pytest.mark.benchmark(group="extrapolate_3d")
def test_extrapolate_nn_unbalanced(benchmark, boundary_unbalanced):
    benchmark(_run_extrapolation, boundary_unbalanced, WhichSolution.NADOL_NEUKIRCH)


@pytest.mark.benchmark(group="extrapolate_3d")
def test_extrapolate_nw_unbalanced(benchmark, boundary_unbalanced):
    benchmark(
        _run_extrapolation, boundary_unbalanced, WhichSolution.NEUKIRCH_WIEGELMANN
    )


@pytest.mark.benchmark(group="extrapolate_3d")
def test_extrapolate_nn_balanced(benchmark, boundary_balanced):
    benchmark(_run_extrapolation, boundary_balanced, WhichSolution.NADOL_NEUKIRCH)


@pytest.mark.benchmark(group="extrapolate_3d")
def test_extrapolate_nw_balanced(benchmark, boundary_balanced):
    benchmark(_run_extrapolation, boundary_balanced, WhichSolution.NEUKIRCH_WIEGELMANN)
