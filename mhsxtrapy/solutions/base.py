from abc import ABC, abstractmethod


class Solution(ABC):

    @abstractmethod
    def phi(self, z):
        pass

    @abstractmethod
    def dphidz(self, z):
        pass

    @abstractmethod
    def f(self, z):
        pass

    @abstractmethod
    def dfdz(self, z):
        pass

    def get_phi_dphi(self, z):
        return self.phi(z), self.dphidz(z)
