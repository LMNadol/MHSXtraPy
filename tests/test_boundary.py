from __future__ import annotations

import numpy as np

from mhsxtrapy._boundary import is_flux_balanced
from mhsxtrapy.examples import dipole, multipole


class TestBoundaryData:

    def test_is_flux_balanced_true(self):
        """is_flux_balanced should return True for zero-mean field."""
        field = [dipole(ix, iy) for ix in range(80) for iy in range(80)]
        field = np.array(field).reshape(80, 80)
        assert is_flux_balanced(field), "Expected flux balanced for zero-mean field"

    def test_is_flux_balanced_false(self):
        """is_flux_balanced should return False for biased field."""
        field = [multipole(ix, iy) for ix in range(80) for iy in range(80)]
        field = np.array(field).reshape(80, 80)
        assert not is_flux_balanced(
            field
        ), "Expected not flux balanced for biased field"
