from __future__ import print_function, division
import numpy as np
from math import sqrt, floor
from numba import njit, float64 as f64, void, boolean

# rkf45 coefficients
b2 = 0.25
b3, c3 = 3 / 32, 9 / 32
b4, c4, d4 = 1932 / 2197, -7200 / 2197, 7296 / 2197
b5, c5, d5, e5 = 439 / 216, -8, 3680 / 513, -845 / 4104
b6, c6, d6, e6, f6 = -8 / 27, 2, -3544 / 2565, 1859 / 4104, -11 / 40

# used to determine y_i+1 from y_i if using rkf45 (4th order)
n1, n3, n4, n5 = 25 / 216, 1408 / 2565, 2197 / 4104, -1 / 5

# used to determine y_i+1 from y_i if using rkf54 (5th order)
nn1, nn3, nn4, nn5, nn6 = 16 / 135, 6656 / 12825, 28561 / 56430, -9 / 50, 2 / 55


@njit
def trilinear3d(pt, grid, xx, yy, zz):
    """
    Given a point, pt, in the grid with dimensions xx, yy and zz,
    returns the value of grid under the trilinear assumption.
    """
    ix = np.where(pt[0] > xx)[0][-1]
    iy = np.where(pt[1] > yy)[0][-1]
    iz = np.where(pt[2] > zz)[0][-1]

    x = (pt[0] - xx[ix]) / (xx[ix + 1] - xx[ix])
    y = (pt[1] - yy[iy]) / (yy[iy + 1] - yy[iy])
    z = (pt[2] - zz[iz]) / (zz[iz + 1] - zz[iz])

    cube = grid[ix : ix + 2, iy : iy + 2, iz : iz + 2, ...]
    square = (1 - x) * cube[0, :, :, ...] + x * cube[1, :, :, ...]
    line = (1 - y) * square[0, :, ...] + y * square[1, :, ...]
    return (1 - z) * line[0, ...] + z * line[1, ...]


@njit
def trilinear3d_grid(pt, grid):
    """
    Given a point, pt, in grid coordinates in the grid,
    returns the value of grid under the trilinear assumption.
    """
    # print("pt", pt)
    ix = floor(pt[0])
    iy = floor(pt[1])
    iz = floor(pt[2])

    x = pt[0] - ix
    y = pt[1] - iy
    z = pt[2] - iz

    # print("ix,iy,iz", ix, iy, iz)
    # print("x,y,z", x, y, z)

    cube = grid[ix : ix + 2, iy : iy + 2, iz : iz + 2, ...]
    square = (1 - x) * cube[0, :, :, ...] + x * cube[1, :, :, ...]
    line = (1 - y) * square[0, :, ...] + y * square[1, :, ...]

    # print((1 - z) * line[0, ...] + z * line[1, ...])

    return (1 - z) * line[0, ...] + z * line[1, ...]


@njit
def getdr(r, x, y, z, csystem):
    """
    Returns the infinitesimal line element at a point, r, in the correct coordinate system.
    """
    ix = floor(r[0])
    iy = floor(r[1])
    iz = floor(r[2])

    dx = x[ix + 1] - x[ix]
    dy = y[iy + 1] - y[iy]
    dz = z[iz + 1] - z[iz]

    xp = x[ix] + (r[0] - ix) * dx
    yp = y[iy] + (r[1] - iy) * dy

    if csystem[2]:
        dr = np.array([dx, xp * dy, xp * np.sin(yp) * dz], dtype=np.float64)
    elif csystem[0]:
        dr = np.array([dx, dy, dz], dtype=np.float64)
    elif csystem[1]:
        dr = np.array([dx, xp * dy, dz], dtype=np.float64)
    return dr


@njit
def gtr(pt, x, y, z):
    """
    Converts a point from grid coordinates to real coordinates.
    """
    ix = floor(pt[0])
    iy = floor(pt[1])
    iz = floor(pt[2])

    if ix == x.shape[0] - 1:
        ix -= 1
    if iy == y.shape[0] - 1:
        iy -= 1
    if iz == z.shape[0] - 1:
        iz -= 1

    pt[0] = x[ix] + (pt[0] - ix) * (x[ix + 1] - x[ix])
    pt[1] = y[iy] + (pt[1] - iy) * (y[iy + 1] - y[iy])
    pt[2] = z[iz] + (pt[2] - iz) * (z[iz + 1] - z[iz])


@njit
def edgecheck(r, minmax, csystem, periodicity):
    """
    Checks whether a point, r, has exited the grid through a periodic boundary and if so, moves the point back into the grid using the periodicity.
    """
    if np.any(periodicity):
        # spherical
        if csystem[2]:
            # if theta periodic
            if periodicity[1]:
                if r[1] < minmax[2] or r[1] > minmax[3]:
                    if r[1] < minmax[2]:
                        r[1] = 2 * minmax[2] - r[1]
                    if r[1] > minmax[3]:
                        r[1] = 2 * minmax[3] - r[1]
                    if r[2] < (minmax[5] + minmax[4]) / 2:
                        r[2] = r[2] + (minmax[5] - minmax[4]) / 2
                    else:
                        r[2] = r[2] - (minmax[5] - minmax[4]) / 2
            # if phi periodic
            if periodicity[2]:
                if r[2] <= minmax[4]:
                    r[2] = r[2] + (minmax[5] - minmax[4])
                if r[2] >= minmax[5]:
                    r[2] = r[2] - (minmax[5] - minmax[4])
        # cartesian
        elif csystem[0]:
            # if x periodic
            if periodicity[0]:
                if r[0] <= minmax[0]:
                    r[0] = r[0] + (minmax[1] - minmax[0])
                if r[0] >= minmax[1]:
                    r[0] = r[0] - (minmax[1] - minmax[0])
            # if y periodic
            if periodicity[1]:
                if r[1] <= minmax[2]:
                    r[1] = r[1] + (minmax[3] - minmax[2])
                if r[1] >= minmax[3]:
                    r[1] = r[1] - (minmax[3] - minmax[2])
            # if z periodic
            if periodicity[2]:
                if r[2] <= minmax[4]:
                    r[2] = r[2] + (minmax[5] - minmax[4])
                if r[2] >= minmax[5]:
                    r[2] = r[2] - (minmax[5] - minmax[4])
        # cylindrical
        elif csystem[1]:
            # if phi periodic
            if periodicity[1]:
                if r[1] < minmax[2]:
                    r[1] = r[1] + (minmax[3] - minmax[2])
                if r[1] > minmax[3]:
                    r[1] = r[1] - (minmax[3] - minmax[2])


@njit
def outedge(r, minmax_box, csystem, periodicity):
    """
    Checks whether a point, r, has left the domain.
    """
    if np.any(periodicity):
        # cartesian
        if csystem[0]:
            outedge = False
            # if no x periodicity
            if not periodicity[0]:
                outedge = outedge or r[0] >= minmax_box[1] or r[0] <= minmax_box[0]
            # if no y periodicity
            if not periodicity[1]:
                outedge = outedge or r[1] >= minmax_box[3] or r[1] <= minmax_box[2]
            # if no z periodicity
            if not periodicity[2]:
                outedge = outedge or r[2] >= minmax_box[5] or r[2] <= minmax_box[4]
        # spherical
        elif csystem[2]:
            outedge = r[0] >= minmax_box[1] or r[0] <= minmax_box[0]
            # if no theta periodicity
            if not periodicity[1]:
                outedge = outedge or r[1] >= minmax_box[3] or r[1] <= minmax_box[2]
            # if no phi periodicity
            if not periodicity[2]:
                outedge = outedge or r[2] >= minmax_box[5] or r[2] <= minmax_box[4]
        # cylindrical
        elif csystem[1]:
            outedge = (
                r[0] >= minmax_box[1]
                or r[0] <= minmax_box[0]
                or r[2] >= minmax_box[5]
                or r[2] <= minmax_box[4]
            )
            # if no phi periodicity
            if not periodicity[1]:
                outedge = outedge or r[1] >= minmax_box[3] or r[1] <= minmax_box[2]
    else:
        outedge = (
            r[0] >= minmax_box[1]
            or r[0] <= minmax_box[0]
            or r[1] >= minmax_box[3]
            or r[1] <= minmax_box[2]
            or r[2] >= minmax_box[5]
            or r[2] <= minmax_box[4]
        )

    return outedge


@njit
def rkf45(
    r0,
    bgrid,
    x,
    y,
    z,
    h,
    hmin,
    hmax,
    epsilon,
    maxpoints,
    oneway,
    stop_criteria,
    t_max,
    minmax,
    minmax_box,
    csystem,
    periodicity,
):
    """
    The actual line tracer after the checks have been made and set up by fieldline3d function. This has been separated from fieldline3d in order to use numba's jit.
    """
    ih = [h] if oneway else [h, -h]

    line = [r0]

    for h in ih:

        count = 1
        out = False
        bounce = False

        while count < maxpoints:

            r0 = line[-1].copy()

            dr = getdr(r0, x, y, z, csystem)
            mindist = dr.min() * h
            hvec = mindist / dr

            rt = r0
            b = trilinear3d_grid(rt, bgrid)
            k1 = hvec * b / sqrt(np.sum(b**2))
            rt = r0 + b2 * k1

            if outedge(rt, minmax_box, csystem, periodicity):
                out = True
                break

            edgecheck(rt, minmax, csystem, periodicity)
            b = trilinear3d_grid(rt, bgrid)
            k2 = hvec * b / sqrt(np.sum(b**2))
            rt = r0 + b3 * k1 + c3 * k2

            if outedge(rt, minmax_box, csystem, periodicity):
                out = True
                break

            edgecheck(rt, minmax, csystem, periodicity)
            b = trilinear3d_grid(rt, bgrid)
            k3 = hvec * b / sqrt(np.sum(b**2))
            rt = r0 + b4 * k1 + c4 * k2 + d4 * k3

            if outedge(rt, minmax_box, csystem, periodicity):
                out = True
                break

            edgecheck(rt, minmax, csystem, periodicity)
            b = trilinear3d_grid(rt, bgrid)
            k4 = hvec * b / sqrt(np.sum(b**2))
            rt = r0 + b5 * k1 + c5 * k2 + d5 * k3 + e5 * k4

            if outedge(rt, minmax_box, csystem, periodicity):
                out = True
                break

            edgecheck(rt, minmax, csystem, periodicity)
            b = trilinear3d_grid(rt, bgrid)
            k5 = hvec * b / sqrt(np.sum(b**2))
            rt = r0 + b6 * k1 + c6 * k2 + d6 * k3 + e6 * k4 + f6 * k5

            if outedge(rt, minmax_box, csystem, periodicity):
                out = True
                break

            edgecheck(rt, minmax, csystem, periodicity)
            b = trilinear3d_grid(rt, bgrid)
            k6 = hvec * b / sqrt(np.sum(b**2))

            # 4th order estimate
            rtest4 = r0 + n1 * k1 + n3 * k3 + n4 * k4 + n5 * k5
            # 5th order estimate
            rtest5 = r0 + nn1 * k1 + nn3 * k3 + nn4 * k4 + nn5 * k5 + nn6 * k6

            # optimum stepsize
            diff = rtest5 - rtest4
            err = sqrt(np.sum(diff**2))
            if err > 0:
                t = (epsilon * abs(h) / (2 * err)) ** 0.25
                if t > t_max:
                    t = t_max
            else:
                t = t_max

            h = t * h
            if abs(h) < hmin:
                h = hmin * np.sign(h)
            if abs(h) > hmax:
                h = hmax * np.sign(h)

            thvec = t * hvec

            rt = r0
            b = trilinear3d_grid(rt, bgrid)
            k1 = thvec * b / sqrt(np.sum(b**2))
            rt = r0 + b2 * k1

            if outedge(rt, minmax_box, csystem, periodicity):
                out = True
                break

            edgecheck(rt, minmax, csystem, periodicity)
            b = trilinear3d_grid(rt, bgrid)
            k2 = thvec * b / sqrt(np.sum(b**2))
            rt = r0 + b3 * k1 + c3 * k2

            if outedge(rt, minmax_box, csystem, periodicity):
                out = True
                break

            edgecheck(rt, minmax, csystem, periodicity)
            b = trilinear3d_grid(rt, bgrid)
            k3 = thvec * b / sqrt(np.sum(b**2))
            rt = r0 + b4 * k1 + c4 * k2 + d4 * k3

            if outedge(rt, minmax_box, csystem, periodicity):
                out = True
                break

            edgecheck(rt, minmax, csystem, periodicity)
            b = trilinear3d_grid(rt, bgrid)
            k4 = thvec * b / sqrt(np.sum(b**2))
            rt = r0 + b5 * k1 + c5 * k2 + d5 * k3 + e5 * k4

            if outedge(rt, minmax_box, csystem, periodicity):
                out = True
                break

            edgecheck(rt, minmax, csystem, periodicity)
            b = trilinear3d_grid(rt, bgrid)
            k5 = thvec * b / sqrt(np.sum(b**2))
            rt = r0 + n1 * k1 + n3 * k3 + n4 * k4 + n5 * k5
            edgecheck(rt, minmax, csystem, periodicity)

            if outedge(rt, minmax_box, csystem, periodicity):
                out = True
                break

            count += 1

            line.append(rt)

            if stop_criteria:
                # check line is still moving
                if count >= 3:
                    dl = line[-1] - line[-2]
                    mdl = sqrt(np.sum(dl**2))
                    if mdl < hmin / 2:
                        break

                    dl = line[-1] - line[-3]
                    mdl = sqrt(np.sum(dl**2))
                    if mdl < hmin / 2:
                        bounce = True
                        break

        # move exited point back to boundary of box
        if out:
            rout = rt.copy()
            rin = line[-1].copy()
            if rout[0] > minmax[1] or rout[0] < minmax[0]:
                xedge = minmax[1] if rout[0] > minmax[1] else minmax[0]
                s = (xedge - rin[0]) / (rout[0] - rin[0])
                rout = s * (rout - rin) + rin
            if rout[1] > minmax[3] or rout[1] < minmax[2]:
                yedge = minmax[3] if rout[1] > minmax[3] else minmax[2]
                s = (yedge - rin[1]) / (rout[1] - rin[1])
                rout = s * (rout - rin) + rin
            if rout[2] > minmax[5] or rout[2] < minmax[4]:
                zedge = minmax[5] if rout[2] > minmax[5] else minmax[4]
                s = (zedge - rin[2]) / (rout[2] - rin[2])
                rout = s * (rout - rin) + rin
            line.append(rout)
        elif bounce:
            line = line[:-1]

        line.reverse()

    # linearr = np.array(line, dtype=np.float64)
    if oneway:
        line.reverse()

    return line


def fieldline3d(
    startpt,
    bgrid,
    x,
    y,
    z,
    h,
    hmin,
    hmax,
    epsilon,
    maxpoints=50000,
    t_max=1.1,
    oneway=False,
    boxedge=None,
    coordsystem="cartesian",
    gridcoord=False,
    stop_criteria=True,
    periodicity=None,
):
    """
    Calculates 3D field line which goes through the point startpt
    startpt - 3 element, 1D array as start point for field line calculation
    bgrid - magnetic field array of shape (nx, ny, nz, 3)
    x, y, z - 1D arrays of grid points on which magnetic field given with shapes (nx,), (ny,) and (nz,) respectively

    h - initial step length
    hmin - minimum step length
    hmax - maximum step length
    epsilon - tolerance to which we require point on field line known

    maxpoints - maximum number of points (including starting point) on a fieldline in one direction (maximum of 2*maxpoints - 1 if traced in both directions)
    t_max - maximum value of correction factor in RKF45 method
    oneway - whether to only calculate field in one direction (sign of h)
    boxedge - use a smaller domain than edge of the grids x, y and z (2x3 array)
    coordsystem - set the coordinate system of the magnetic field grid
    gridcoord - use grid coordinates as input and output to routine (startpt and boxedge)
    stop_critieria - use the stopping criteria to detect if field line has stopped inside domain
    periodicity - set the periodicity of the grid e.g. 'xy' for periodicity in both x and y direction
    """
    # create array for coordsystems
    csystem = np.array(
        [
            True if coordsystem == "cartesian" else False,
            True if coordsystem == "cylindrical" else False,
            True if coordsystem == "spherical" else False,
        ],
        dtype=np.bool_,
    )

    # create array for periodicities
    if periodicity is None:
        if coordsystem == "cartesian":
            periodicity = np.array([False, False, False], dtype=np.bool_)
        elif coordsystem == "cylindrical":
            periodicity = np.array([False, True, False], dtype=np.bool_)
        elif coordsystem == "spherical":
            periodicity = np.array([False, True, True], dtype=np.bool_)
    else:
        if coordsystem == "cartesian":
            periodicity = np.array(
                [
                    True if "x" in periodicity else False,
                    True if "y" in periodicity else False,
                    True if "z" in periodicity else False,
                ],
                dtype=np.bool_,
            )
        elif coordsystem == "cylindrical":
            periodicity = np.array(
                [
                    False,
                    False if "p" in periodicity else True,
                    True if "z" in periodicity else False,
                ],
                dtype=np.bool_,
            )
        elif coordsystem == "spherical":
            periodicity = np.array(
                [
                    False,
                    False if "t" in periodicity else True,
                    False if "p" in periodicity else True,
                ],
                dtype=np.bool_,
            )

    # define edges of box
    minmax = np.array(
        [0, x.shape[0] - 1, 0, y.shape[0] - 1, 0, z.shape[0] - 1], dtype=np.float64
    )

    # print("minmax", minmax)

    # print("boxedge", boxedge)

    if boxedge is not None:
        periodicity[:] = False
        boxedge1 = boxedge.copy()
        if not gridcoord:
            for idim, dim in enumerate([x, y, z]):
                for im in range(2):
                    index = (
                        np.argwhere(boxedge[im, idim] >= dim).max() - 1
                    )  # Had to add -1 here
                    boxedge1[im, idim] = index + (boxedge[im, idim] - dim[index]) / (
                        dim[index + 1] - dim[index]
                    )

        minmax_box = np.array(
            [
                max([boxedge1[0, 0], minmax[0]]),
                min([boxedge1[1, 0], minmax[1]]),
                max([boxedge1[0, 1], minmax[2]]),
                min([boxedge1[1, 1], minmax[3]]),
                max([boxedge1[0, 2], minmax[4]]),
                min([boxedge1[1, 2], minmax[5]]),
            ],
            dtype=np.float64,
        )
    else:
        minmax_box = minmax

    # first convert point into grid coordinates
    if not gridcoord:
        ix = np.argwhere(startpt[0] >= x).max()
        iy = np.argwhere(startpt[1] >= y).max()
        iz = np.argwhere(startpt[2] >= z).max()

        r0 = np.empty_like(startpt)
        r0[0] = ix + (startpt[0] - x[ix]) / (x[ix + 1] - x[ix])
        r0[1] = iy + (startpt[1] - y[iy]) / (y[iy + 1] - y[iy])
        r0[2] = iz + (startpt[2] - z[iz]) / (z[iz + 1] - z[iz])
    else:
        r0 = startpt.copy()

    # print("r0", r0)

    # print("minmax_box", minmax_box)

    # Produce an error if the first point isn't in the box
    if (
        r0[0] < minmax_box[0]
        or r0[0] > minmax_box[1]
        or r0[1] < minmax_box[2]
        or r0[1] > minmax_box[3]
        or r0[2] < minmax_box[4]
        or r0[2] > minmax_box[5]
    ):
        print("Error: Start point not in range")
        print("Start point is: {} {} {}".format(startpt[0], startpt[1], startpt[2]))

        if r0[0] < minmax_box[0] or r0[0] > minmax_box[1]:
            print("{} (x) is the issue".format(startpt[0]))
        if r0[1] < minmax_box[2] or r0[1] > minmax_box[3]:
            print("{} (y) is the issue".format(startpt[1]))
        if r0[2] < minmax_box[4] or r0[2] > minmax_box[5]:
            print("{} (z) is the issue".format(startpt[2]))

        raise ValueError
    elif not (hmin < np.abs(h) < hmax):
        print("You need to satisfy hmin ({}) < h ({}) < hmax({})".format(hmin, h, hmax))
        raise ValueError
    elif np.all(trilinear3d_grid(r0, bgrid) == np.zeros(3)):
        print("Start point is a null point")
        raise ValueError

    line = rkf45(
        r0,
        bgrid,
        x,
        y,
        z,
        h,
        hmin,
        hmax,
        epsilon,
        maxpoints,
        oneway,
        stop_criteria,
        t_max,
        minmax,
        minmax_box,
        csystem,
        periodicity,
    )

    if gridcoord == False:
        for pt in line:
            gtr(pt, x, y, z)

    line = np.array(line)

    return line
