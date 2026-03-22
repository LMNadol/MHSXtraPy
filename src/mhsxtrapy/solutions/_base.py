from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

__all__ = []


class BaseSolution(ABC):

    @abstractmethod
    def phi(
        self, z: np.float64, p: np.ndarray, q: np.ndarray
    ) -> np.float64 | np.ndarray:
        pass

    @abstractmethod
    def dphidz(
        self, z: np.float64, p: np.ndarray, q: np.ndarray
    ) -> np.float64 | np.ndarray:
        pass

    @abstractmethod
    def f(self, z: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def dfdz(self, z: np.ndarray) -> np.ndarray:
        pass

    def get_phi_dphi(
        self, z: np.float64, p: np.ndarray, q: np.ndarray
    ) -> tuple[np.float64 | np.ndarray, np.float64 | np.ndarray]:
        return self.phi(z, p, q), self.dphidz(z, p, q)
