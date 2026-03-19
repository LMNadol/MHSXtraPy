from abc import ABC, abstractmethod


class BaseSolution(ABC):

    @abstractmethod
    def phi(self, z, p, q):
        pass

    @abstractmethod
    def dphidz(self, z, p, q):
        pass

    @abstractmethod
    def f(self, z):
        pass

    @abstractmethod
    def dfdz(self, z):
        pass

    def get_phi_dphi(self, z, p, q):
        return self.phi(z, p, q), self.dphidz(z, p, q)
