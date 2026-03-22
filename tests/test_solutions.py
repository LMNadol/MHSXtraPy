from __future__ import annotations

import numpy as np

from mhsxtrapy.solutions import get_solution
from mhsxtrapy.solutions._low import dfdz_low, f_low, phi_low
from mhsxtrapy.solutions._nadol_neukirch import phi_nn
from mhsxtrapy.solutions._neukirch_wiegelmann import dfdz_nw, f_nw, phi_nw
from mhsxtrapy.types import WhichSolution


class TestGetSolution:
    def test_get_solution_valid(self):
        """get_solution should return correct class for each WhichSolution member."""
        for which in WhichSolution:
            solution = get_solution(which)
            assert solution is not None, f"get_solution({which}) returned None"

    def test_get_solution_invalid(self):
        """get_solution with unknown value should raise ValueError or TypeError."""
        invalid_inputs = ["invalid", 123, None, 3.14]
        for invalid in invalid_inputs:
            try:
                get_solution(invalid)
                assert False, f"get_solution({invalid}) did not raise an error"
            except (ValueError, TypeError):
                pass  # Expected exception was raised


class TestLow:

    def test_phi_low_at_zero(self):
        """phi(z=0) should equal 1."""
        z = 0.0
        p = np.array([1.0])  # arbitrary parameter
        q = np.array([1.0])  # arbitrary parameter
        kappa = 0.5  # arbitrary parameter
        result = phi_low(z, p, q, kappa)
        expected = 1.0
        assert np.isclose(result, expected), f"Expected {expected}, got {result}"

    def test_phi_low_at_infinity(self):
        """phi(z → infinity) should approach 0."""
        z = 1e6  # large value to approximate infinity
        p = np.array([1.0])  # arbitrary parameter
        q = np.array([1.0])  # arbitrary parameter
        kappa = 0.5
        result = phi_low(z, p, q, kappa)
        expected = 0.0
        assert np.isclose(
            result, expected, atol=1e-6
        ), f"Expected {expected}, got {result}"

    def test_f_low_at_zero(self):
        """f(z=0) should return expected analytic value."""
        z = 0.0
        a = 0.5
        kappa = 0.5
        result = f_low(z, a, kappa)
        expected = a
        assert np.isclose(result, expected), f"Expected {expected}, got {result}"

    def test_dfdz_low_at_zero(self):
        """dfdz(z=0) should return expected analytic value."""
        z = 0.0
        a = 0.5
        kappa = 0.5
        result = dfdz_low(z, a, kappa)
        expected = -kappa * a
        assert np.isclose(result, expected), f"Expected {expected}, got {result}"


class TestNeukirchWiegelmann:

    def test_phi_nw_at_zero(self):
        """phi(z=0) should equal 1."""
        z = 0.0
        p = np.array([1.0])  # arbitrary parameter
        q = np.array([1.0])  # arbitrary parameter
        z0 = 2.0
        deltaz = 0.2
        result = phi_nw(z, p, q, z0, deltaz)
        expected = 1.0
        assert np.isclose(result, expected), f"Expected {expected}, got {result}"

    def test_phi_nw_at_infinity(self):
        """phi(z → infinity) should approach 0."""
        z = 1e6  # large value to approximate infinity
        p = np.array([1.0])  # arbitrary parameter
        q = np.array([1.0])  # arbitrary parameter
        z0 = 2.0
        deltaz = 0.2
        result = phi_nw(z, p, q, z0, deltaz)
        expected = 0.0
        assert np.isclose(
            result, expected, atol=1e-6
        ), f"Expected {expected}, got {result}"

    def test_f_nw_at_zero(self):
        """f(z=0) should return expected analytic value."""
        z = 0.0
        a = 0.5  # arbitrary parameter
        b = 1.0  # arbitrary parameter
        z0 = 2.0
        deltaz = 0.2
        result = f_nw(z, z0, deltaz, a, b)
        expected = a * (1 + b)
        assert np.isclose(result, expected), f"Expected {expected}, got {result}"

    def test_f_nw_at_z0(self):
        """f(z=z0) should return expected analytic value."""
        a = 0.5  # arbitrary parameter
        b = 1.0  # arbitrary parameter
        z0 = 2.0
        deltaz = 0.1 * z0
        result = f_nw(z0, z0, deltaz, a, b)
        expected = a
        assert np.isclose(result, expected), f"Expected {expected}, got {result}"

    def test_dfdz_nw_at_z0(self):
        """dfdz(z=z0) should return expected analytic value."""
        a = 0.5  # arbitrary parameter
        b = 1.0  # arbitrary parameter
        z0 = 2.0
        deltaz = 0.1 * z0
        result = dfdz_nw(z0, z0, deltaz, a, b)
        expected = -a * b / deltaz
        assert np.isclose(result, expected), f"Expected {expected}, got {result}"


class TestNadolNeukirch:

    def test_phi_nn_at_zero(self):
        """phi(z=0) should equal 1."""
        z = 0.0
        p = np.array([1.0])  # arbitrary parameter
        q = np.array([1.0])  # arbitrary parameter
        z0 = 2.0
        deltaz = 0.2
        result = phi_nn(z, p, q, z0, deltaz)
        expected = 1.0
        assert np.isclose(result, expected), f"Expected {expected}, got {result}"

    def test_phi_nn_at_infinity(self):
        """phi(z → infinity) should approach 0."""
        z = 1e6  # large value to approximate infinity
        p = np.array([1.0])  # arbitrary parameter
        q = np.array([1.0])  # arbitrary parameter
        z0 = 2.0
        deltaz = 0.2
        result = phi_nn(z, p, q, z0, deltaz)
        expected = 0.0
        assert np.isclose(
            result, expected, atol=1e-6
        ), f"Expected {expected}, got {result}"
