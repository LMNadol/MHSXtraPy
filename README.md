# MHSXtraPy

MHSXtraPy is a set of Python codes for three-dimensional magnetohydrostatic (MHS) extrapolation from a given two-dimensional array, intended for the extrapolation of magnetic fields observed on the solar photosphere.

L. Nadol 2025-02-11

An overview of the codes is provided in the following publication:

Nadol, L., Neukirch, T. (2025).
MHSXtraPy - A set of codes for three-dimensional magnetohydrostatic solar magnetic field extrapolation,
RAS Techniques and Instruments,
https://doi.org/etc/etc/etc

An overview of the underlying theory is provided in the following publication:

Nadol, L., Neukirch, T. (2025).
An efficient method for magnetic field extrapolation based on a family of analytical three-dimensional magnetohydrostatic equilibria,
Solar Physics,
https://doi.org/etc/etc/etc

Between October 2020 and March 2025, I was studying at the School of Mathematics and Statistics at the University of St Andrews, UK, as a PhD student within the Solar and Magnetospheric Theory Group supervised by Prof Thomas Neukirch. In this time, I collected and documented a set of codes for local 3D MHS magnetic field extrapolation in Cartesian geometry.

In this repository, I present the resulting Python library. On the one hand, there may be limited interest in the codes given the linear nature of their mathematical description and given that non-linear solutions regularly outperform such (Wiegelmann & Sakurai, 2021) when it comes to agreement with observations (Pevtsov et al., 1994, DeRosa et al., 2009). On the other hand, the codes work efficiently and the method incorporates the non force-free lower chromosphere and photosphere, which other popular methods fail to do as they rely on the assumption of a force-free solar atmosphere throughout. I am not aware of any alternative open-source codes that extrapolate in an MHS manner. 

A .pdf file will be added in the near future with detailed information on the codes and their performance, which will be based on a chapter from my PhD thesis. This will be uploaded and this README updated once the final version of my thesis is submitted (apporox. May 2025).

In the following a condensed version of this information is provided, based on the mentioned chapter in my PhD thesis. 

## Details

MHSXtraPy is used for the calculation of solar MHS magnetic fields in Cartesian coordinates, in which the $x$- and $y$-directions are the latitudinal and longitudinal extent of the boundary condition and the $z$-direction dictates the height above the photosphere perpendicular to the orientation of the boundary condition. The code was written in Python, as Python is a widely used language and provides an ecosystem of libraries useful for our purposes. As of now, MHSXtraPy allows for the extrapolation of the magnetic vector field above a photospheric boundary condition obtained from either a manually defined array or from line-of-sight magnetograms observed by either the Polarimetric and Helioseismic Imager (PHI) onboard Solar Orbiter (Solanki et al. 2019) or the Helioseismic and Magnetic Imager (HMI) onboard the Solar Dynamics Observatory (SDO, Scherrer et al. 2012). Specifically defined input stream routines exist in MHSXtraPy making the use of Solar Orbiter and SDO .fits format files particularly easy. However, magnetogram observations by any other than the mentioned instruments can be used through manual instantiation of all of the attributes of a Field2dData object. Field2dData is a dataclass specifically intended for the handling of boundary conditions (see below). Fromt the resulting three-dimensional magnetic field vector the current density, the Lorentz force and the variations in plasma pressure and plasma density can be computed. 

MHSFLEx consists of three main parts which are used successively to calculate the magnetic field vector and resulting model features:

1. **Loading data:** Creates a Field2dData object from a given boundary condition.

2. **Magnetic field calculation:** Calculates the magnetic field vector B and the
partial derivatives of Bz from a Field2dData object.

3. **Full-featured model output:** Creates a Field3dData object from a Field2dData object through extrapolation of B and calculation of the partial derivatives of Bz. From a Field3dData obbject all other model attributes can be accessed.

The code can be found in the folder mhsxtrapy structured into 9 files (alphabetically, not in order of use):

**b3d.py** -- magnetic field calculation (not used explicitly by user) 

**field2d.py** -- dataclass for boundary condition (used explicitly by user for **loading data**) 

**field3d.py** -- dataclass for final extrapolated field (used explicitly by user for **magnetic field calculation** and **full-featured model output**) 

**graphics_balanced.py** -- visualisation for fluxbalanced boundary condition (not used explicitly by user) 

**graphics.py** -- visualisation for non fluxbalanced boundary condition (not used explicitly by user) 

**nff2ff.py** -- transition from non force-free to force-free (not used explicitly by user) 

**phibar.py** -- solution of ODE (not used explicitly by user) 

**vis.py** -- visualisation interface (optionally used by user) 

While not competitive to MHD simulation programs in physical realism or to potential fields in computational simplicity, the presented code tries to balance both aspects. Therefore, it provides all essential building blocks for future magnetic field line extrapolations, yet development of the MHSXtraPy package is not complete. Further features should be added, which include but are not limited to:

- Improved visualisation routines and the creation of a purposeful user interface as well as decoupling of data grid and graphics grid. This would significantly increase the user friendliness of the library.
- Further, for the application to data, pre-processing routines along the lines of Wiegelmann et al. (2006), Zhu et al. (2020) could be included, such that only azimuthally adjusted data is used.
- To further improve the numerical efficiency of the code, the utilisation of the fast Fourier transform can be optimised by usage of certain list sizes. In general, most FFT implementations perform best on sizes that can be decomposed into small prime factors. The number of used Fourier modes can be restricted to such number. Alternatively, by additional zero padding the grid can be extended to a suitable size without changing the result of the computation and retaining the maximal number of possible Fourier modes.
- Integration of GPU support using e.g. PyTorch (Paszke et al. 2019) or JAX.numpy (Bradbury et al. 2018).
- Creation of a global extrapolation version of the code.
- Optimised reading routines for other instruments additionally to Solar Orbiter/PHI and SDO/HMI.

For suggestions, questions and requests regarding MHSXtraPy please email [lillinadol@gmail.com](mailto:lillinadol@gmail.com). 

## Quick Start

As seen above, for the user of the code only field2d.py, field3d.py and vis.py are relevant. For the most simplistivc case, we assume that the boundary condition is given as np.ndarray (here instantiated as a multipole as in the example seen in EXAMPLE-analytical-bc file). First we import the relevant files:

```python
import numpy as np
from mhsxtrapy.field2d import Field2dData, check_fluxbalance, FluxBalanceState
from mhsxtrapy.field3d import calculate_magfield
from mhsxtrapy.b3d import WhichSolution
from mhsxtrapy.vis import plot_magnetogram_3D
from mhsxtrapy.examples import multipole
```

Then we instantiate the Field2dData object from given parameters:

```python
nx, ny, nz, nf = 200, 200, 400, 200
xmin, xmax, ymin, ymax, zmin, zmax = 0.0, 20.0, 0.0, 20.0, 0.0, 20.0

px = (xmax - xmin) / nx
py = (ymax - ymin) / ny
pz = (zmax - zmin) / nz

x = np.linspace(xmin, xmax, nx, dtype=np.float64)
y = np.linspace(ymin, ymax, ny, dtype=np.float64)
z = np.linspace(zmin, zmax, nz, dtype=np.float64)

data_bz = np.random.randn((nx, ny))

if np.fabs(check_fluxbalance(data_bz)) < 0.01:
    data2d = Field2dData(nx, ny, nz, nf, px, py, pz, x, y, z, data_bz, flux_balance_state=FluxBalanceState.UNBALANCED)
else: 
    data2d = Field2dData(nx, ny, nz, nf, px, py, pz, x, y, z, data_bz, flux_balance_state=FluxBalanceState.BALANCED)

```

We choose the extrapolation parameters and the type of solution we want to use. From this we calculate the model:

```python
data3d = calculate_magfield(data2d,alpha=0.05, a=0.22, which_solution=WhichSolution.ASYMP, b=1.0, z0=2.0, deltaz=0.2)
```

Finally, we use one of the provided visualisation routines:

```python
plot_magnetogram_3D(data3d, view="side", footpoints="active-regions")
```

## Examples

Four different examples are provided:

- **EXAMPLE-analytical-bc** which uses an analytically defined multipole as boundary condition
- **EXAMPLE-Low-Lou** which uses a semi-analytical non-linear force-free boundary condition extracted from Low and Lou (1990)
- **EXAMPLE-SDO** which uses an SDO/HMI magnetogram as boundary condition
- **EXAMPLE-Solar-Orbiter** which uses a Solar Orbiter/PHI/HRT magnetogram as boundary condition

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

