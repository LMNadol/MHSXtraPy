from __future__ import annotations

import numpy as np

from mhsxtrapy._fourier import _seehafer


class TestSeehafer:
    """Test that seehafer() preserves antisymmetry: Bz(-x, y) = -Bz(x, y)."""

    def test_seehafer_output_shape(self):
        """The output shape should be (2*ny, 2*nx) for an input of shape (ny, nx)."""
        rng = np.random.default_rng(42)
        bz = rng.standard_normal((8, 10))

        result = _seehafer(bz)
        expected_shape = (16, 20)

        assert (
            result.shape == expected_shape
        ), f"Expected shape {expected_shape}, got {result.shape}"

    def test_antisymmetry_x(self):
        """Bz(-x, y) = -Bz(x, y): flipping x negates the field."""
        rng = np.random.default_rng(42)
        bz = rng.standard_normal((8, 10))

        result = _seehafer(bz)
        ny, nx = bz.shape

        original = result[ny:, nx:]
        x_flipped = result[ny:, :nx][:, ::-1]

        np.testing.assert_array_equal(original, bz)
        np.testing.assert_array_equal(x_flipped, -original)

    def test_antisymmetry_y(self):
        """Bz(x, -y) = -Bz(x, y): flipping y negates the field."""
        rng = np.random.default_rng(42)
        bz = rng.standard_normal((8, 10))

        result = _seehafer(bz)
        ny, nx = bz.shape

        original = result[ny:, nx:]
        y_flipped = result[:ny, nx:][::-1, :]

        np.testing.assert_array_equal(y_flipped, -original)

    def test_antisymmetry_xy(self):
        """Bz(-x, -y) = Bz(x, y): flipping both axes preserves the field."""
        rng = np.random.default_rng(42)
        bz = rng.standard_normal((8, 10))

        result = _seehafer(bz)
        ny, nx = bz.shape

        original = result[ny:, nx:]
        xy_flipped = result[:ny, :nx][::-1, ::-1]

        np.testing.assert_array_equal(xy_flipped, original)

    def test_antisymmetry_dipole(self):
        """Antisymmetry holds for a physically motivated dipole-like field."""
        y, x = np.mgrid[1:6, 1:8]
        bz = 1.0 / (x**2 + y**2)

        result = _seehafer(bz)
        ny, nx = bz.shape

        original = result[ny:, nx:]
        x_flipped = result[ny:, :nx][:, ::-1]

        np.testing.assert_array_equal(x_flipped, -original)
