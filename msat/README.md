# Magnetic Skeleton Analysis Tools

These are a set of tools written in modern fortran to analyse the topological skeletons of magnetic fields and other divergence free vector fields.

The package is made up of three main tools:
1. Null Finder -- Finds the points where **F** = **0**
2. Sign Finder -- Finds the sign of each null point together with two vectors which correspond to the spine and fan planes.
3. Separatrix Surface Finder -- Finds the separatrix surfaces, spine lines and separators in the fields.

These three in turn can be used to fully identify the skeletons of divergence free fields.

There are several extra tools available:
1. Heliospheric Current Sheet Finder -- for global solar spherical magnetic fields
2. Cut Maker -- makes a cut through the magnetic skeleton enabling easier 2D visualisation

Tools are also available for the visualisation of the output of the MSAT codes. These are written in both Python and IDL (versions 3 and 8 respectively) and stored in the `pyvis` and `idlvis` directories respectively. The python tools require NumPy, Matplotlib and Mayavi.

A manual is provided in LaTeX, this can be compiled using `make doc`

## Quick Start
Assuming that the magnetic field data file is in the correct format and is in the location `data/field3d.dat`:
1. Typing `make` should compile all the fortran codes
2. Run the following on the codes successively. `***` represents the appropriate coordinate system -- e.g. `ssfxyz` for cartesian coordinates and `ssfrtp` for spherical coordinates.
```sh
./nf -i data/field3d.dat
./sf*** -i data/field3d.dat
./ssf*** -i data/field3d.dat
```
3. The field can now be visualised in 3D in iPython, e.g.
```python
import pyvis.model3d as m3d
m3d.make('data/field3d.dat', ['nulls', 'spines', 'separators', 'sepsurf_rings'])
```
Please refer to the manual for other setups.
