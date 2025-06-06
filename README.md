# MHSXtraPy

MHSXtraPy is a Python code for three-dimensional linear magnetohydrostatic (MHS) extrapolation from a given two-dimensional boundary condition, intended for the extrapolation of magnetic fields observed on the solar photosphere.

L. Nadol 2025-06-05

An overview of the underlying theory is provided in the following publication:

Nadol, L., Neukirch, T. (2025).
An efficient method for magnetic field extrapolation based on a family of analytical three-dimensional magnetohydrostatic equilibria,
https://link.springer.com/article/10.1007/s11207-025-02469-1

An overview of the codes will be provided in a subsequent publication intended for the journal Royal Astronmical Society Techniques and Instruments.

## Details

In this repository, I present a Python code for local linear 3D MHS magnetic field extrapolation in Cartesian geometry developed during my PhD. The code works efficiently and the method incorporates the non force-free lower chromosphere and photosphere, which other popular methods fail to do as they rely on the assumption of a force-free solar atmosphere throughout.

In the model and code the $x$- and $y$-directions are the latitudinal and longitudinal extent of the boundary condition and the $z$-direction dictates the height above the photosphere perpendicular to the orientation of the boundary condition. The code was written in Python, as Python is a widely used language and provides an ecosystem of libraries useful for our purposes. As of now, MHSXtraPy allows for the extrapolation of the magnetic vector field above a photospheric boundary condition obtained from either a manually defined array or from line-of-sight magnetograms observed by either the Polarimetric and Helioseismic Imager (PHI) onboard Solar Orbiter (Solanki et al. 2019) or the Helioseismic and Magnetic Imager (HMI) onboard the Solar Dynamics Observatory (SDO, Scherrer et al. 2012). Specifically defined input stream routines exist in MHSXtraPy making the use of Solar Orbiter and SDO .fits format files particularly easy. However, magnetogram observations by any other than the mentioned instruments can be used through manual instantiation of all of the attributes of a Field2dData object. Field2dData is a dataclass specifically intended for the handling of boundary conditions (see below). From the resulting three-dimensional magnetic field vector the current density, the Lorentz force and the deviations from hydrostatic balance in plasma pressure and plasma density can be computed. 

<!-- MHSFLEx consists of three main parts which are used successively to calculate the magnetic field vector and resulting model features:

1. **Loading data:** Creates a Field2dData object from a given boundary condition.

2. **Magnetic field calculation:** Calculates the magnetic field vector B and the
partial derivatives of Bz from a Field2dData object.

3. **Full-featured model output:** Creates a Field3dData object from a Field2dData object through extrapolation of B and calculation of the partial derivatives of Bz. From a Field3dData obbject all other model attributes can be accessed.

The code can be found in the folder mhsxtrapy structured into 9 files (alphabetically, not in order of use):

**b3d.py** -- magnetic field calculation (not used explicitly by user) 

**field2d.py** -- dataclass for boundary condition (used explicitly by user for **loading data**) 

**field3d.py** -- dataclass for final extrapolated field (used explicitly by user for **magnetic field calculation** and **full-featured model output**) 

**graphics_balanced.py** -- visualisation for fluxbalanced boundary condition (not used explicitly by user) 

**nff2ff.py** -- transition from non force-free to force-free (not used explicitly by user) 

**phibar.py** -- solution of ODE (not used explicitly by user) 

**vis.py** -- visualisation interface (optionally used by user)  -->

<!-- While not competitive to MHD simulation programs in physical realism or to potential fields in computational simplicity, the presented code tries to balance both aspects. Therefore, it provides all essential building blocks for future magnetic field extrapolations, yet development of the MHSXtraPy package is not complete. -->

Ideas for future developments of the code include but are not limited to: 

- Improved visualisation routines and the creation of a purposeful user interface as well as decoupling of data grid and graphics grid. This would significantly increase the user friendliness of the library.
- Further, for the application to data, pre-processing routines along the lines of Wiegelmann et al. (2006), Zhu et al. (2020) could be included, such that only azimuthally adjusted data is used.
- To further improve the numerical efficiency of the code, the utilisation of the fast Fourier transform can be optimised by usage of certain list sizes. In general, most FFT implementations perform best on sizes that can be decomposed into small prime factors. The number of used Fourier modes can be restricted to such number. Alternatively, by additional zero padding the grid can be extended to a suitable size without changing the result of the computation and retaining the maximal number of possible Fourier modes.
- Integration of GPU support using e.g. PyTorch (Paszke et al. 2019) or JAX.numpy (Bradbury et al. 2018).
- Creation of a global extrapolation version of the code.
- Optimised reading routines for other instruments additionally to Solar Orbiter/PHI and SDO/HMI.

For suggestions, questions and requests regarding MHSXtraPy please email [lillinadol@gmail.com](mailto:lillinadol@gmail.com). 

## Quick Start

For the use of the code only field2d.py, field3d.py and vis.py are relevant. For the most simplistic case, we assume that the boundary condition is given as np.ndarray (here instantiated as a multipole as in the example seen in example-analytical-bc Jupyter notebook). First we import the relevant files:

```python
import numpy as np

from mhsxtrapy.b3d import WhichSolution
from mhsxtrapy.examples import multipole
from mhsxtrapy.field2d import Field2dData, FluxBalanceState, check_fluxbalance
from mhsxtrapy.field3d import calculate_magfield
from mhsxtrapy.plotting.vis import (
    plot_ddensity_xy,
    plot_ddensity_z,
    plot_dpressure_xy,
    plot_dpressure_z,
    plot_magnetogram_2D,
    plot_magnetogram_3D,
)
```

Then we instantiate the Field2dData object from given parameters:

```python
nx, ny, nz, nf = 200, 200, 400, 200
xmin, xmax, ymin, ymax, zmin, zmax = 0.0, 20.0, 0.0, 20.0, 0.0, 20.0

"""
Calculation of pixel sizes and arrays of x-, y- and z-extension of box. 
"""
pixelsize_x = (xmax - xmin) / nx
pixelsize_y = (ymax - ymin) / ny
pixelsize_z = (zmax - zmin) / nz

x_arr = np.linspace(xmin, xmax, nx, dtype=np.float64)
y_arr = np.linspace(ymin, ymax, ny, dtype=np.float64)
z_arr = np.linspace(zmin, zmax, nz, dtype=np.float64)

B_PHOTO = 500

data_bz = np.zeros((ny, nx))

for ix in range(0, nx):
    for iy in range(0, ny):
        x = x_arr[ix] / 10.0
        y = y_arr[iy] / 10.0
        data_bz[iy, ix] = multipole(x, y)

data2d = Field2dData(
    nx,
    ny,
    nz,
    nf,
    pixelsize_x,
    pixelsize_y,
    pixelsize_z,
    x_arr,
    y_arr,
    z_arr,
    data_bz,
    flux_balance_state=FluxBalanceState.UNBALANCED,
)
```

We choose the extrapolation parameters and the type of solution we want to use. From this we calculate the model:

```python
data3d = calculate_magfield(
    data2d,
    alpha=0.05,
    a=0.22,
    which_solution=WhichSolution.ASYMP,
    b=1.0,
    z0=2.0,
    deltaz=0.2,
)
```

Finally, we use one of the provided visualisation routines:

```python
plot_magnetogram_3D(data3d, view="los", footpoints="active-regions")
```

## Examples

Four different examples are provided as Jupyter notebooks:

- **example-analytical-bc** which uses an analytically defined multipole as boundary condition
- **example-low-lou** which uses a semi-analytical non-linear force-free boundary condition extracted from Low and Lou (1990)
- **example-sdo** which uses an SDO/HMI magnetogram as boundary condition
- **example-solar-orbiter** which uses a Solar Orbiter/PHI/HRT magnetogram as boundary condition

As well as the notebook **paper.ipynb** which contains the example used in the RASTI paper.

## Bibliography 

Bradbury, J., Frostig, R., Hawkins, P., Johnson, M. J., Leary, C., Maclaurin, D., Necula, G., Paszke, A., VanderPlas, J., Wanderman-Milne, S. & Zhang, Q. (2018), ‘JAX: composable transformations of Python+NumPy programs’. http://github.com/jax-ml/jax.

DeRosa, M.L. et al. (2009) “A CRITICAL ASSESSMENT OF NONLINEAR FORCE-FREE FIELD MODELING OF THE SOLAR CORONA FOR ACTIVE REGION 10953,” The Astrophysical Journal, 696(2), pp. 1780–1791. Available at: https://doi.org/10.1088/0004-637x/696/2/1780.

ITT Visual Information Solutions (2009), ‘IDL Reference Guide: IDL Version 7.1’.

Low, B.C. and Lou, Y.Q. (1990) “Modeling solar force-free magnetic fields,” The Astrophysical Journal, 352, p. 343. Available at: https://doi.org/10.1086/168541.

Neukirch, T. and Wiegelmann, T. (2019) “Analytical Three-dimensional Magnetohydrostatic Equilibrium Solutions for Magnetic Field Extrapolation Allowing a Transition from Non-force-free to Force-free Magnetic Fields,” Solar Physics, 294(12), p. 171. Available at: https://doi.org/10.1007/s11207-019-1561-0.

Paszke, A. et al. (2019) “PyTorch: An Imperative Style, High-Performance Deep Learning Library,” arXiv [Preprint]. Available at: https://doi.org/10.48550/arxiv.1912.01703.

Pevtsov, A.A., Canfield, R.C. and Metcalf, T.R. (1994) “Patterns of helicity in solar active regions,” The Astrophysical Journal, 425, p. L117. Available at: https://doi.org/10.1086/187324.

Scherrer, P.H. et al. (2012) “The Helioseismic and Magnetic Imager (HMI) Investigation for the Solar Dynamics Observatory (SDO),” Solar Physics, 275(1–2), pp. 207–227. Available at: https://doi.org/10.1007/s11207-011-9834-2.

Solanki, S.K. et al. (2019) “The Polarimetric and Helioseismic Imager on Solar Orbiter,” Astronomy & Astrophysics, 642, p. A11. Available at: https://doi.org/10.1051/0004-6361/201935325.

Wiegelmann, T. et al. (2017) “Magneto-static Modeling from Sunrise/IMaX: Application to an Active Region Observed with Sunrise II,” The Astrophysical Journal Supplement Series, 229(1), p. 18. Available at: https://doi.org/10.3847/1538-4365/aa582f.

Wiegelmann, T. and Sakurai, T. (2021) “Solar force-free magnetic fields,” Living Reviews in Solar Physics, 18(1), p. 1. Available at: https://doi.org/10.1007/s41116-020-00027-4.

Zhu, X., Wiegelmann, T. and Inhester, B. (2020) “Preprocessing of vector magnetograms for magnetohydrostatic extrapolations,” Astronomy & Astrophysics, 644, p. A57. Available at: https://doi.org/10.1051/0004-6361/202039079.

