from importlib.metadata import version

from mhsxtrapy._field import _extrapolate_3d

__version__ = version("mhsxtrapy")

__all__ = [
    "_extrapolate_3d",
    "__version__",
]
