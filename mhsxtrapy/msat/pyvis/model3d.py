import math
import sys

import numpy as np
import mayavi.mlab as ml
import vtk

from . import read as rd
from . import fieldline3d as fl

try:
    __IPYTHON__
except NameError:
    pass
else:
    from IPython import get_ipython

    ipython = get_ipython()
    ipython.magic(r"%gui qt")

# turn of warnings while vtk/mayavi compatibility is fixed -- still works (error from older vtk)
vtk.vtkObject.GlobalWarningDisplayOff()

sign_names = {-2: "Sink", -1: "Neg", 0: "Zero", 1: "Pos", 2: "Source"}


def sphr2cart(rs, ts, ps):
    # convert (r, theta, phi) to (x, y, z)
    return rs * np.sin(ts) * np.cos(ps), rs * np.sin(ts) * np.sin(ps), rs * np.cos(ts)


def cyl2cart(Rs, ps, zs):
    # convert (R, phi, z) to (x, y, z)
    return Rs * np.cos(ps), Rs * np.sin(ps), zs


class Model3D:

    def __init__(
        self,
        filename,
        addlist,
        null_list=None,
        box=True,
        fieldlines=None,
        linecolor=(0, 0, 0),
        nskip=20,
        nullrad=1,
        nfanlines=40,
        nring=None,
        coordsystem="cartesian",
        no_nulls=False,
        sun=True,
        axes=False,
        outdir=None,
        periodicity="",
        only_nf=False,
    ):
        """
        Makes a 3D visualisation of the output from Magnetic Skeleton Analysis Tools

            fname: name of the file containing the original magnetic field
            addlist: list of features to be plotted e.g. ['nulls', 'separators'] will
                only plot the nulls and separators

            null_list: if None (default), will plot all nulls, otherwise give a list (starting at 1) of nulls to plot
                e.g. nulls=list(range(45,65)) for all nulls between 45 and 64 inclusive
            nullrad: will scale radius of null spheres by this factor (default 1)
                e.g. nullrad=0.5 will halve the size of the nulls
            box: if True (default), plots a box otherwise set to False
            sun: turn on and off plotting a sun and z-axis when in spherical coordinates
            axes: add axes to the plot
            fieldlines: provide a numpy array of shape (3, n) of start points and it will trace magnetic field lines
            linecolor: color of fieldlines (defaults to black)
            nskip: how many rings to skip in plotting
            nfanlines: number of fieldlines in the fan plane to plot
            nring: ring number in the separatrix surfaces to trace fieldlines from

            outdir: change output directory of data
            only_nf: only read in data from NF if SF anf SSF haven't been run
            no_nulls: turn off reading of any nulls - useful just to plot fieldlines/field
            coordsystem: set the coordinate system of field (cartesian or spherical)
            periodicity: set periodicity of cartesian coordinate system
        """

        self.filename = filename
        self.coordsystem = coordsystem
        self.periodicity = periodicity
        self.nskip = nskip

        self.bgrid, self.xx, self.yy, self.zz = rd.field(self.filename, grid=True)
        self.ds = (
            min(
                [np.diff(self.xx).min(), np.diff(self.yy).min(), np.diff(self.zz).min()]
            )
            / 2
        )

        print("Using {} coordinates".format(self.coordsystem))

        self.figure = ml.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(800, 800))

        if outdir is not None:
            rd.outprefix = outdir

        self.periodic_check = False
        if self.periodicity != "":
            self.periodic_dist = 1e100
            self.periodic_check = True
            if "x" in self.periodicity:
                self.periodic_dist = min(
                    (self.periodic_dist, (self.xx[-1] - self.xx[0]) ** 2)
                )
            if "y" in self.periodicity:
                self.periodic_dist = min(
                    (self.periodic_dist, (self.yy[-1] - self.yy[0]) ** 2)
                )
            if "z" in self.periodicity:
                self.periodic_dist = min(
                    (self.periodic_dist, (self.zz[-1] - self.zz[0]) ** 2)
                )

        if not no_nulls and not only_nf:
            self.nulldata = rd.nulls(filename)
            self.set_null_list(null_list)
        if only_nf:
            self.nulldata = rd.nulls(filename, simple=True)
            self.set_null_list(null_list)

        self.add_structures(*addlist, nullrad=nullrad, nfanlines=nfanlines, nring=nring)

        if self.coordsystem == "spherical":
            if sun:
                self.add_sun()
            box = False
        if box:
            self.add_box()
        if axes:
            self.add_axes()

        if fieldlines is not None:
            self.add_fieldlines(fieldlines, col=linecolor)

    def set_null_list(self, lst):
        if lst is None:
            self.nulllist = self.nulldata.number
        else:
            if max(lst) > self.nulldata.number.max() or min(lst) < 1:
                raise ValueError("Invalid list of nulls given")
            self.nulllist = np.array(lst)

    def add_structures(self, *args, **kwargs):
        if "nulls" in args:
            self.add_nulls(size=kwargs["nullrad"])
        if "sepsurf_flines" in args:
            self.add_sepsurf(
                draw="fieldlines", nlines=kwargs["nfanlines"], nring=kwargs["nring"]
            )
        if "spines" in args:
            self.add_spines()
        if "sepsurf_rings" in args:
            self.add_sepsurf()
        if "separators" in args:
            self.add_separators()
        if "hcs_rings" in args:
            self.add_hcs()
        if "hcs_flines" in args:
            self.add_hcs(draw="fieldlines")
        if "hcs_sep" in args:
            self.add_separators(hcs=True)

    def add_sepsurf(self, draw="rings", nlines=50, nring=None):
        """
        Adds separatrix surfaces to 3D plot, either 'rings' or 'flines'.
            nlines: number of fieldlines to plot
            nring: sets the ring to trace fieldlines from. Set as an integer for ring number or float in [0, 1] as fraction of all rings.
        """

        cols = {
            -2: (218 / 255, 112 / 255, 214 / 255),
            -1: (0.5, 0.5, 1),
            0: (0.5, 1, 0.5),
            1: (1, 0.5, 0.5),
            2: (1.0, 178 / 255, 102 / 255),
        }

        nulls = self.nulldata[self.nulllist - 1]

        if draw == "rings":
            print("Adding separatrix surface rings")

            rings, breaks = rd.rings(
                self.filename, breaks=True, nskip=self.nskip, null_list=self.nulllist
            )

            for isign in np.unique(nulls.sign):
                # new very efficient routine for plotting many lines
                # two lists, one for positive and the other for negative nulls
                x, y, z, s, ptcons = ([] for _ in range(5))
                index = 0
                for inull in nulls.number[nulls.sign == isign] - 1:
                    print("Null {:5d}".format(inull + 1))
                    sys.stdout.write("\033[F")
                    for iring, ring in enumerate(rings[inull]):
                        # convert points if required
                        if self.coordsystem == "spherical":
                            ring[:, 0], ring[:, 1], ring[:, 2] = sphr2cart(
                                ring[:, 0], ring[:, 1], ring[:, 2]
                            )
                        elif self.coordsystem == "cylindrical":
                            ring[:, 0], ring[:, 1], ring[:, 2] = cyl2cart(
                                ring[:, 0], ring[:, 1], ring[:, 2]
                            )
                        # add ring points to lists
                        x.append(ring[:, 0])
                        y.append(ring[:, 1])
                        z.append(ring[:, 2])
                        s.append(np.zeros_like(ring[:, 0]))
                        if self.periodic_check:
                            # use distances between consectutive points to detect extra breaks for periodicity
                            dists = np.r_[
                                np.sum(np.diff(ring, axis=0) ** 2, axis=1), [0]
                            ]
                            # use break data to plot the individual lines in each ring as the break apart
                            brks = np.unique(
                                np.r_[
                                    -1,
                                    np.where(breaks[inull][iring] == 1)[0],
                                    np.where(dists > 0.9 * self.periodic_dist)[0],
                                    ring.shape[0] - 1,
                                ]
                            )
                        else:
                            # use break data to plot the individual lines in each ring as the break apart
                            brks = np.unique(
                                np.r_[
                                    -1,
                                    np.where(breaks[inull][iring] == 1)[0],
                                    ring.shape[0] - 1,
                                ]
                            )
                        for ib0, ib1 in zip(brks[:-1], brks[1:]):
                            # add the right indicies based on the breaks
                            ptcons.append(
                                np.vstack(
                                    [
                                        np.arange(index + ib0 + 1, index + ib1),
                                        np.arange(index + ib0 + 2, index + ib1 + 1),
                                    ]
                                ).T
                            )
                        index += ring.shape[0]
                # and plot...
                if len(x) > 0:
                    src = ml.pipeline.scalar_scatter(
                        np.hstack(x),
                        np.hstack(y),
                        np.hstack(z),
                        np.hstack(s),
                        figure=self.figure,
                    )
                    src.mlab_source.dataset.lines = np.vstack(ptcons)
                    src.update()

                    lines = ml.pipeline.stripper(src, figure=self.figure)
                    ml.pipeline.surface(
                        lines,
                        color=cols[isign],
                        line_width=1,
                        name=sign_names[isign] + "SeparatrixRings",
                        figure=self.figure,
                    )

        elif draw == "fieldlines":
            print("Adding separatrix surface field lines")

            rings = rd.rings(self.filename, nskip=self.nskip, null_list=self.nulllist)

            for isign in np.unique(nulls.sign):
                x, y, z, s, ptcons = ([] for _ in range(5))
                index = 0
                for inull in nulls.number[nulls.sign == isign] - 1:
                    print("Null {}".format(inull + 1))
                    sys.stdout.write("\033[F")

                    # select the right ring to trace from
                    if nring is None:
                        # choose ring a fifth of the way through
                        ring = rings[inull][len(rings[inull]) // 5]
                    else:
                        if isinstance(nring, int):
                            # choose specific ring
                            ring = rings[inull][nring]
                        elif isinstance(nring, float):
                            # choose ring as a fraction of one of total
                            ring = rings[inull][math.floor(nring * len(rings[inull]))]

                    nskip = len(ring[:, 0]) // nlines

                    for startpt in ring[::nskip, :]:
                        # choose some good parameters
                        h = 1e-2
                        hmin = 1e-3
                        hmax = 0.5
                        epsilon = 1e-5

                        # calculate the fieldline
                        line = fl.fieldline3d(
                            startpt,
                            self.bgrid,
                            self.xx,
                            self.yy,
                            self.zz,
                            h,
                            hmin,
                            hmax,
                            epsilon,
                            coordsystem=self.coordsystem,
                            periodicity=self.periodicity,
                        )

                        # cut off the fieldline at the point closest to the null - only want the fan, not the spine
                        dists = np.sqrt(
                            (line[:, 0] - self.nulldata[inull].pos[0]) ** 2
                            + (line[:, 1] - self.nulldata[inull].pos[1]) ** 2
                            + (line[:, 2] - self.nulldata[inull].pos[2]) ** 2
                        )
                        imin = dists.argmin()
                        line = (
                            line[0 : imin + 1, :]
                            if self.nulldata[inull].sign == -1
                            else line[imin:, :]
                        )

                        if self.coordsystem == "spherical":
                            line[:, 0], line[:, 1], line[:, 2] = sphr2cart(
                                line[:, 0], line[:, 1], line[:, 2]
                            )
                        elif self.coordsystem == "cylindrical":
                            line[:, 0], line[:, 1], line[:, 2] = cyl2cart(
                                line[:, 0], line[:, 1], line[:, 2]
                            )

                        x.append(line[:, 0])
                        y.append(line[:, 1])
                        z.append(line[:, 2])
                        length = len(line[:, 0])
                        s.append(np.zeros(length))
                        if (
                            self.coordsystem == "spherical"
                            or self.coordsystem == "cylindical"
                            or not self.periodic_check
                        ):
                            ptcons.append(
                                np.vstack(
                                    [
                                        np.arange(index, index + length - 1),
                                        np.arange(index + 1, index + length),
                                    ]
                                ).T
                            )
                        else:
                            dists = np.r_[np.sum(np.diff(line, axis=0) ** 2, axis=1), 0]
                            brks = np.unique(
                                np.r_[
                                    -1,
                                    np.where(dists > 0.9 * self.periodic_dist)[0],
                                    dists.shape[0] - 1,
                                ]
                            )
                            for ib0, ib1 in zip(brks[:-1], brks[1:]):
                                ptcons.append(
                                    np.vstack(
                                        [
                                            np.arange(index + ib0 + 1, index + ib1),
                                            np.arange(index + ib0 + 2, index + ib1 + 1),
                                        ]
                                    ).T
                                )
                        index += length

                # add points to model
                if len(x) > 0:
                    src = ml.pipeline.scalar_scatter(
                        np.hstack(x),
                        np.hstack(y),
                        np.hstack(z),
                        np.hstack(s),
                        figure=self.figure,
                    )
                    src.mlab_source.dataset.lines = np.vstack(ptcons)
                    src.update()

                    lines = ml.pipeline.stripper(src, figure=self.figure)
                    ml.pipeline.surface(
                        lines,
                        color=cols[isign],
                        line_width=1,
                        name=sign_names[isign] + "SeparatrixFieldlines",
                        figure=self.figure,
                    )
        else:
            raise ValueError("Set draw to be either 'rings' or 'fieldlines'")

    def add_hcs(self, draw="rings", nlines=100):
        """
        Adds heliospheric current sheet curtains to 3D plot, either 'rings' or 'flines'.
            nlines: number of fieldlines to plot
            nring: sets the ring to trace fieldlines from. Set as an integer for ring number or float in [0, 1] as fraction of all rings.
        """

        x, y, z, s, ptcons = ([] for _ in range(5))
        index = 0

        if draw == "rings":
            print("Adding heliospheric current sheet curtain surface rings")

            rings, breaks = rd.rings(
                self.filename, breaks=True, nskip=self.nskip, hcs=True
            )

            for inull in range(len(rings)):
                print("HCS {:5d}".format(inull // 2 + 1))
                sys.stdout.write("\033[F")
                for iring, ring in enumerate(rings[inull]):
                    # convert points, it's sphericals!
                    ring[:, 0], ring[:, 1], ring[:, 2] = sphr2cart(
                        ring[:, 0], ring[:, 1], ring[:, 2]
                    )
                    # add ring points to lists
                    x.append(ring[:, 0])
                    y.append(ring[:, 1])
                    z.append(ring[:, 2])
                    s.append(np.zeros_like(ring[:, 0]))
                    # use break data to plot the individual lines in each ring as the break apart
                    brks = np.unique(
                        np.r_[
                            [-1],
                            np.where(breaks[inull][iring] == 1)[0],
                            [ring.shape[0] - 1],
                        ]
                    )
                    for ib0, ib1 in zip(brks[:-1], brks[1:]):
                        # add the right indicies based on the breaks
                        ptcons.append(
                            np.vstack(
                                [
                                    np.arange(index + ib0 + 1, index + ib1),
                                    np.arange(index + ib0 + 2, index + ib1 + 1),
                                ]
                            ).T
                        )
                    index += ring.shape[0]

            # add points to model
            if len(x) > 0:
                src = ml.pipeline.scalar_scatter(
                    np.hstack(x),
                    np.hstack(y),
                    np.hstack(z),
                    np.hstack(s),
                    figure=self.figure,
                )
                src.mlab_source.dataset.lines = np.vstack(ptcons)
                src.update()

                lines = ml.pipeline.stripper(src, figure=self.figure)
                ml.pipeline.surface(
                    lines,
                    color=(0, 1, 0),
                    line_width=1,
                    name="HCSRings",
                    figure=self.figure,
                )

        elif draw == "fieldlines":
            print("Adding heliospheric current sheet curtain surface field lines")

            rings = rd.rings(self.filename, nskip=self.nskip, hcs=True)

            for inull in range(0, len(rings), 2):
                print("HCS {:5d}".format(inull // 2 + 1))
                sys.stdout.write("\033[F")

                iring = 1
                nskip = len(rings[inull][iring][:, 0]) // nlines

                for idir in range(2):
                    for startpt in rings[inull + idir][iring][::nskip, :]:
                        # choose some good parameters
                        h = 2e-2
                        hmin = h * 0.1
                        hmax = h * 10
                        epsilon = h * 0.01

                        # calculate the fieldline
                        line = fl.fieldline3d(
                            startpt,
                            self.bgrid,
                            self.xx,
                            self.yy,
                            self.zz,
                            h,
                            hmin,
                            hmax,
                            epsilon,
                            coordsystem=self.coordsystem,
                        )
                        imax = np.argmax(line[:, 0])
                        line = line[: imax + 1, :] if idir == 1 else line[imax:, :]

                        line[:, 0], line[:, 1], line[:, 2] = sphr2cart(
                            line[:, 0], line[:, 1], line[:, 2]
                        )

                        x.append(line[:, 0])
                        y.append(line[:, 1])
                        z.append(line[:, 2])
                        length = len(line[:, 0])
                        s.append(np.zeros(length))
                        ptcons.append(
                            np.vstack(
                                [
                                    np.arange(index, index + length - 1),
                                    np.arange(index + 1, index + length),
                                ]
                            ).T
                        )
                        index += length

            if len(x) > 0:
                src = ml.pipeline.scalar_scatter(
                    np.hstack(x),
                    np.hstack(y),
                    np.hstack(z),
                    np.hstack(s),
                    figure=self.figure,
                )
                src.mlab_source.dataset.lines = np.vstack(ptcons)
                src.update()

                lines = ml.pipeline.stripper(src, figure=self.figure)
                ml.pipeline.surface(
                    lines,
                    color=(0, 1, 0),
                    line_width=1,
                    name="HCSFieldlines",
                    figure=self.figure,
                )

        else:
            raise ValueError("Set draw to be either 'rings' or 'fieldlines'")

        for inull in range(0, len(rings), 2):
            if draw == "fieldlines":
                rings[inull][0][:, 0], rings[inull][0][:, 1], rings[inull][0][:, 2] = (
                    sphr2cart(
                        rings[inull][0][:, 0],
                        rings[inull][0][:, 1],
                        rings[inull][0][:, 2],
                    )
                )
            ml.plot3d(
                rings[inull][0][:, 0],
                rings[inull][0][:, 1],
                rings[inull][0][:, 2],
                color=(0, 1, 0),
                line_width=6,
                tube_radius=None,
                name="HCSBase",
                figure=self.figure,
            )

    def add_fieldlines(self, startpts, col=(0, 0, 0), lw=2):
        """
        Add fieldlines to the 3D plot starting at startpts. Also set the colour and width of the lines.
        """
        print("Adding field lines")

        x, y, z, s, ptcons = ([] for _ in range(5))
        index = 0

        for i, startpt in enumerate(startpts, start=1):
            print("Calculating fieldline {}".format(i))
            sys.stdout.write("\033[F")

            # choose fieldline parameters
            h = 1e-2
            hmin = 1e-3
            hmax = 0.2
            epsilon = 1e-5

            # calculate the fieldline
            line = fl.fieldline3d(
                startpt,
                self.bgrid,
                self.xx,
                self.yy,
                self.zz,
                h,
                hmin,
                hmax,
                epsilon,
                coordsystem=self.coordsystem,
                periodicity=self.periodicity,
            )

            if self.coordsystem == "spherical":
                line[:, 0], line[:, 1], line[:, 2] = sphr2cart(
                    line[:, 0], line[:, 1], line[:, 2]
                )
            elif self.coordsystem == "cylindrical":
                line[:, 0], line[:, 1], line[:, 2] = cyl2cart(
                    line[:, 0], line[:, 1], line[:, 2]
                )

            x.append(line[:, 0])
            y.append(line[:, 1])
            z.append(line[:, 2])
            length = len(line[:, 0])
            s.append(np.zeros(length))
            if (
                self.coordsystem == "spherical"
                or self.coordsystem == "cylindical"
                or not self.periodic_check
            ):
                ptcons.append(
                    np.vstack(
                        [
                            np.arange(index, index + length - 1),
                            np.arange(index + 1, index + length),
                        ]
                    ).T
                )
            else:
                dists = np.r_[np.sum(np.diff(line, axis=0) ** 2, axis=1), 0]
                brks = np.unique(
                    np.r_[
                        -1,
                        np.where(dists > 0.9 * self.periodic_dist)[0],
                        dists.shape[0] - 1,
                    ]
                )
                for ib0, ib1 in zip(brks[:-1], brks[1:]):
                    ptcons.append(
                        np.vstack(
                            [
                                np.arange(index + ib0 + 1, index + ib1),
                                np.arange(index + ib0 + 2, index + ib1 + 1),
                            ]
                        ).T
                    )
            index += length

        # add points to model
        if len(x) > 0:
            src = ml.pipeline.scalar_scatter(
                np.hstack(x),
                np.hstack(y),
                np.hstack(z),
                np.hstack(s),
                figure=self.figure,
            )
            src.mlab_source.dataset.lines = np.vstack(ptcons)
            src.update()

            lines = ml.pipeline.stripper(src, figure=self.figure)
            ml.pipeline.surface(
                lines, color=col, line_width=lw, name="Fieldlines", figure=self.figure
            )

    def add_spines(self):
        """
        Adds the spine lines to the 3D plot.
        """
        print("Adding spines")

        cols = {
            -2: (0.5, 0, 0.5),
            -1: (0, 0, 1),
            0: (0, 1, 0),
            1: (1, 0, 0),
            2: (1, 165 / 255, 0),
        }

        spines = rd.spines(self.filename, null_list=self.nulllist)

        nulls = self.nulldata[self.nulllist - 1]

        # very similar to ring algorithm without breaks
        for isign in np.unique(nulls.sign):
            # set up lists like rings
            x, y, z, s, ptcons = ([] for _ in range(5))
            index = 0
            for inull in nulls.number[nulls.sign == isign]:
                print("Null {:5d}".format(inull))
                sys.stdout.write("\033[F")
                for spine in spines[inull - 1]:
                    if self.coordsystem == "spherical":
                        spine[:, 0], spine[:, 1], spine[:, 2] = sphr2cart(
                            spine[:, 0], spine[:, 1], spine[:, 2]
                        )
                    elif self.coordsystem == "cylindical":
                        spine[:, 0], spine[:, 1], spine[:, 2] = cyl2cart(
                            spine[:, 0], spine[:, 1], spine[:, 2]
                        )
                    x.append(spine[:, 0])
                    y.append(spine[:, 1])
                    z.append(spine[:, 2])
                    s.append(np.zeros_like(spine[:, 0]))
                    if (
                        self.coordsystem == "spherical"
                        or self.coordsystem == "cylindical"
                        or not self.periodic_check
                    ):
                        ptcons.append(
                            np.vstack(
                                [
                                    np.arange(index, index + spine.shape[0] - 1),
                                    np.arange(index + 1, index + spine.shape[0]),
                                ]
                            ).T
                        )
                    else:
                        dists = np.r_[np.sum(np.diff(spine, axis=0) ** 2, axis=1), 0]
                        brks = np.unique(
                            np.r_[
                                -1,
                                np.where(dists > 0.9 * self.periodic_dist)[0],
                                dists.shape[0] - 1,
                            ]
                        )
                        for ib0, ib1 in zip(brks[:-1], brks[1:]):
                            ptcons.append(
                                np.vstack(
                                    [
                                        np.arange(index + ib0 + 1, index + ib1),
                                        np.arange(index + ib0 + 2, index + ib1 + 1),
                                    ]
                                ).T
                            )
                    index += spine.shape[0]
            # and plot...
            if len(x) > 0:
                src = ml.pipeline.scalar_scatter(
                    np.hstack(x),
                    np.hstack(y),
                    np.hstack(z),
                    np.hstack(s),
                    figure=self.figure,
                )
                src.mlab_source.dataset.lines = np.vstack(ptcons)
                src.update()

                lines = ml.pipeline.stripper(src, figure=self.figure)
                ml.pipeline.surface(
                    lines,
                    color=cols[isign],
                    line_width=4,
                    name=sign_names[isign] + "Spines",
                    figure=self.figure,
                )

    def add_separators(self, hcs=False, colour=None, nskip=1):
        """
        Add separators to 3D plot
            hcs: controls whether null or hcs separators
            colour: override custom colours
        """
        print("Adding separators")

        seps, conn = rd.separators(self.filename, null_list=self.nulllist, hcs=hcs)

        # simpler version of spines and rings - no need for positive and negative
        x, y, z, s, ptcons = ([] for _ in range(5))
        index = 0

        if hcs:
            linecol = (1.0, 0.6470588235294118, 0.0)  # orange
        else:
            linecol = (1, 1, 0)  # yellow

        if colour is not None:
            linecol = colour

        if hcs:
            to_do = [0]
        else:
            to_do = self.nulllist - 1

        for inull in to_do:
            print("Null {:5d}".format(inull + 1))
            sys.stdout.write("\033[F")
            for con, sep in zip(conn[inull], seps[inull]):
                if con in self.nulllist:
                    if nskip > 1:
                        sep = sep[::nskip, :]
                    if self.coordsystem == "spherical":
                        sep[:, 0], sep[:, 1], sep[:, 2] = sphr2cart(
                            sep[:, 0], sep[:, 1], sep[:, 2]
                        )
                    elif self.coordsystem == "cylindical":
                        sep[:, 0], sep[:, 1], sep[:, 2] = cyl2cart(
                            sep[:, 0], sep[:, 1], sep[:, 2]
                        )
                    x.append(sep[:, 0])
                    y.append(sep[:, 1])
                    z.append(sep[:, 2])
                    s.append(np.zeros_like(sep[:, 0]))
                    if (
                        self.coordsystem == "spherical"
                        or self.coordsystem == "cylindical"
                        or not self.periodic_check
                    ):
                        ptcons.append(
                            np.vstack(
                                [
                                    np.arange(index, index + sep.shape[0] - 1),
                                    np.arange(index + 1, index + sep.shape[0]),
                                ]
                            ).T
                        )
                    else:
                        dists = np.r_[np.sum(np.diff(sep, axis=0) ** 2, axis=1), [0]]
                        brks = np.unique(
                            np.r_[
                                [-1],
                                np.where(dists > 0.9 * self.periodic_dist)[0],
                                dists.shape[0] - 1,
                            ]
                        )
                        for ib0, ib1 in zip(brks[:-1], brks[1:]):
                            ptcons.append(
                                np.vstack(
                                    [
                                        np.arange(index + ib0 + 1, index + ib1),
                                        np.arange(index + ib0 + 2, index + ib1 + 1),
                                    ]
                                ).T
                            )
                    index += sep.shape[0]

        if len(x) > 0:
            src = ml.pipeline.scalar_scatter(
                np.hstack(x),
                np.hstack(y),
                np.hstack(z),
                np.hstack(s),
                figure=self.figure,
            )
            src.mlab_source.dataset.lines = np.vstack(ptcons)
            src.update()

            lines = ml.pipeline.stripper(src, figure=self.figure)
            ml.pipeline.surface(
                lines,
                color=linecol,
                line_width=6,
                name="Separators",
                figure=self.figure,
            )

    def add_nulls(self, size=1, no_sf=False):
        """
        Add nulls to the 3D plot
        """
        print("Adding nulls")

        cols = {
            -2: (0.5, 0, 0.5),
            -1: (0, 0, 1),
            0: (0, 1, 0),
            1: (1, 0, 0),
            2: (1, 165 / 255, 0),
        }

        boxsize = (
            min(
                [
                    self.xx[-1] - self.xx[0],
                    self.yy[-1] - self.yy[0],
                    self.zz[-1] - self.zz[0],
                ]
            )
            / 40
        )

        # find a nice size for the radius of the null points
        r = max([boxsize, self.ds]) * size

        # pick out only the nulls required
        nulldata1 = self.nulldata[self.nulllist - 1]

        if no_sf:
            pos = nulldata1.pos
            if self.coordsystem == "spherical":
                pos[:, 0], pos[:, 1], pos[:, 2] = sphr2cart(
                    pos[:, 0], pos[:, 1], pos[:, 2]
                )
            ml.points3d(
                pos[:, 0],
                pos[:, 1],
                pos[:, 2],
                color=cols[0],
                scale_factor=r,
                resolution=32,
                name=sign_names[0] + "Nulls",
                figure=self.figure,
            )
        else:
            for sign in np.unique(nulldata1.sign):
                pos = nulldata1.pos[nulldata1.sign == sign]
                if self.coordsystem == "spherical":
                    pos[:, 0], pos[:, 1], pos[:, 2] = sphr2cart(
                        pos[:, 0], pos[:, 1], pos[:, 2]
                    )
                ml.points3d(
                    pos[:, 0],
                    pos[:, 1],
                    pos[:, 2],
                    color=cols[sign],
                    scale_factor=r,
                    resolution=32,
                    name=sign_names[sign] + "Nulls",
                    figure=self.figure,
                )

        # r = max([boxsize, ds])
        # r = r*size
        # ntheta, nphi = 16, 32
        # theta, phi = np.mgrid[0:np.pi:ntheta*1j, 0:2*np.pi:nphi*1j]
        # x = r * np.sin(theta) * np.cos(phi)
        # y = r * np.sin(theta) * np.sin(phi)
        # z = r * np.cos(theta)

        # for inull in nulllist:
        #     pos = nulldata[inull].pos
        #     if self.coordsystem == 'spherical':
        #         pos[0], pos[1], pos[2] = sphr2cart(*pos)
        #     ml.mesh(x + pos[0], y + pos[1], z + pos[2], color=cols[nulldata[inull].sign], figure=self.figure)

    def change_null_size(self, size):
        # not working properly... also removes outline
        objs = self.figure.children
        count_remove = 0
        for obj in objs:
            if "Nulls" in obj.name:
                count_remove += 1
        while count_remove > 0:
            objs = self.figure.children
            for obj in objs:
                if "Nulls" in obj.name:
                    obj.remove()
                    count_remove -= 1

        self.add_nulls(size)

    def add_box(self):
        print("Adding box")
        # create line with all corners to be normalised
        line = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 0, 1],
                [1, 0, 0],
                [1, 1, 0],
                [1, 1, 1],
                [1, 1, 0],
                [0, 1, 0],
                [0, 1, 1],
                [0, 1, 0],
                [0, 0, 0],
                [0, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 1],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )

        # rescale line to form box around the proper coordinate system
        line[:, 0] = line[:, 0] * (self.xx[-1] - self.xx[0]) + self.xx[0]
        line[:, 1] = line[:, 1] * (self.yy[-1] - self.yy[0]) + self.yy[0]
        line[:, 2] = line[:, 2] * (self.zz[-1] - self.zz[0]) + self.zz[0]

        ml.plot3d(
            line[:, 0],
            line[:, 1],
            line[:, 2],
            color=(0, 0, 0),
            tube_radius=None,
            line_width=1,
            name="Box",
            figure=self.figure,
        )

        # ml.outline(extent=[*xx[[0, -1]], *yy[[0, -1]], *zz[[0, -1]]], color=(0,0,0))

    def add_axes(self):
        ml.axes(
            extent=[*self.xx[[0, -1]], *self.yy[[0, -1]], *self.zz[[0, -1]]],
            figure=self.figure,
        )

    def add_base(self, coord, pos, vmin=-10, vmax=10):
        """
        Adds a magnetogram on the side of the box
        coord: 'x', 'y' or 'z'
        pos: 'top' or 'bottom'
        """

        if pos == "top":
            ipos = -1
        elif pos == "bottom":
            ipos = 0
        else:
            raise ValueError("Only 'top' and 'bottom' are options for pos argument")

        colon = slice(None, None, None)

        if coord == "x":
            y, z = np.meshgrid(self.yy, self.zz, indexing="ij")
            x = np.ones_like(y) * self.xx[ipos]
            indices = (ipos, colon, colon, 0)
        elif coord == "y":
            x, z = np.meshgrid(self.xx, self.zz, indexing="ij")
            y = np.ones_like(x) * self.yy[ipos]
            indices = (colon, ipos, colon, 1)
        elif coord == "z":
            x, y = np.meshgrid(self.xx, self.yy, indexing="ij")
            z = np.ones_like(y) * self.zz[ipos]
            indices = (colon, colon, ipos, 2)

        if self.coordsystem == "spherical":
            x, y, z = sphr2cart(x, y, z)
        elif self.coordsystem == "cylindrical":
            x, y, z = cyl2cart(x, y, z)

        ml.mesh(
            x,
            y,
            z,
            scalars=-self.bgrid[indices],
            colormap="Greys",
            vmin=vmin,
            vmax=vmax,
            figure=self.figure,
        )

    def add_sun(self, absbrmax=10):
        # make theta, phi grids of bgrid coords and create the coords of sphere
        theta, phi = np.meshgrid(self.yy, self.zz, indexing="ij")
        x = self.xx[0] * np.sin(theta) * np.cos(phi)
        y = self.xx[0] * np.sin(theta) * np.sin(phi)
        z = self.xx[0] * np.cos(theta)

        # -bgrid because otherwise want reversed colourtable
        ml.mesh(
            x,
            y,
            z,
            scalars=-self.bgrid[0, :, :, 0],
            colormap="Greys",
            vmin=-absbrmax,
            vmax=absbrmax,
            name="Solar surface",
            figure=self.figure,
        )

        # make z-axis
        ml.plot3d(
            [0, 0],
            [0, 0],
            [-self.xx.max(), self.xx.max()],
            color=(0, 0, 0),
            tube_radius=None,
            line_width=4,
            name="Z-axis",
            figure=self.figure,
        )

    def save(self):
        ml.savefig(
            "figures/" + rd.prefix(self.filename) + "-model3d.png", figure=self.figure
        )

    def add_surface(self):
        print("Adding separatrix surface rings")

        rings, breaks, assocs = rd.rings(self.filename, breaks=True, assocs=True)

        cols = {-1: (0.5, 0.5, 1), 0: (0.5, 1, 0.5), 1: (1, 0.5, 0.5)}

        acc = 6

        # ringlist = []
        # trianglelist = []
        # ptnums = [0]
        # for lst in break1: ptnums.append(ptnums[-1] + len(lst))
        # for ring in rings[inull]: ringlist.extend(ring.tolist())

        if True:
            for inull in self.nulllist - 1:
                ringlist = []
                trianglelist = []
                ptnums = [0]
                for lst in breaks[inull]:
                    ptnums.append(ptnums[-1] + len(lst))
                ptnums = ptnums[:-1]
                # print(ptnums)
                for ring in rings[inull]:
                    ringlist.extend(ring.tolist())
                ringlist = np.array(ringlist)
                print("Null {}".format(inull + 1))
                rings1 = rings[inull][::-1]
                breaks1 = breaks[inull][::-1]
                assocs1 = assocs[inull][::-1]
                ptnums = ptnums[::-1]
                for iring, ring in enumerate(rings1[:-1]):
                    dists = np.r_[np.sum(np.diff(ring, axis=0) ** 2, axis=1), [0]]
                    breaks1[iring][dists > acc] = 1
                    brks = (
                        np.r_[
                            [-1],
                            np.where(breaks1[iring] == 1)[0],
                            [breaks1[iring].shape[0] - 1],
                        ]
                        + 1
                    )
                    for ipt, pt in enumerate(ring):
                        if ipt not in brks - 1 and ipt != brks[-1]:
                            if assocs1[iring][ipt] != assocs1[iring][ipt + 1]:
                                tri1 = [
                                    ptnums[iring] + ipt,
                                    ptnums[iring + 1] + assocs1[iring][ipt] - 1,
                                    ptnums[iring + 1] + assocs1[iring][ipt],
                                ]
                                trianglelist.append(tri1)
                            tri2 = [
                                ptnums[iring] + ipt,
                                ptnums[iring + 1] + assocs1[iring][ipt],
                                ptnums[iring] + ipt + 1,
                            ]
                            trianglelist.append(tri2)
                # return trianglelist, ringlist, ptnums
                ml.triangular_mesh(
                    ringlist[:, 0],
                    ringlist[:, 1],
                    ringlist[:, 2],
                    trianglelist,
                    color=cols[self.nulldata[inull].sign],
                    tube_radius=None,
                    opacity=0.5,
                    mask_points=2,
                    figure=self.figure,
                )

        if False:
            nskp = 20  # nskipglob
            for inull in self.nulllist - 1:
                ringlist = []
                trianglelist = []
                ptnums = [0]
                for lst in breaks[inull][::-nskp]:
                    ptnums.append(ptnums[-1] + len(lst))
                ptnums = ptnums[:-1]
                # print(ptnums)
                for ring in rings[inull][::-nskp]:
                    ringlist.extend(ring.tolist())
                ringlist = np.array(ringlist)
                print("Null {}".format(inull + 1))
                rings1 = rings[inull][::-nskp]
                breaks1 = breaks[inull][::-nskp]
                assocs1 = assocs[inull][::-1]
                # ptnums = ptnums[::-1]
                for iring, ring in enumerate(rings1[:-1]):
                    if iring / nskp - iring // nskp > 0:
                        dists = np.r_[np.sum(np.diff(ring, axis=0) ** 2, axis=1), [0]]
                        breaks1[iring][dists > acc] = 1
                        brks = (
                            np.r_[
                                [-1],
                                np.where(breaks1[iring] == 1)[0],
                                [breaks1[iring].shape[0] - 1],
                            ]
                            + 1
                        )
                        newass = assocs1[iring * nskp]
                        for iskp in range(nskp):
                            newass = assocs1[iring * nskp + iskp + 1][newass - 1]
                        for ipt, pt in enumerate(ring):
                            if ipt not in brks - 1 and ipt != brks[-1]:
                                if newass[ipt] != newass[ipt + 1]:
                                    tri1 = [
                                        ptnums[iring] + ipt,
                                        ptnums[iring + 1] + newass[ipt] - 1,
                                        ptnums[iring + 1] + newass[ipt],
                                    ]
                                    trianglelist.append(tri1)
                                tri2 = [
                                    ptnums[iring] + ipt,
                                    ptnums[iring + 1] + newass[ipt],
                                    ptnums[iring] + ipt + 1,
                                ]
                                trianglelist.append(tri2)
                # return trianglelist, ringlist, ptnums
                ml.triangular_mesh(
                    ringlist[:, 0],
                    ringlist[:, 1],
                    ringlist[:, 2],
                    trianglelist,
                    color=cols[self.nulldata[inull].sign],
                    tube_radius=None,
                    opacity=1,
                    figure=self.figure,
                )


if __name__ == "__main__":
    m3d = Model3D(
        "data/field_20100601_0081_fft.dat",
        ["separators", "nulls", "hcs_sep", "hcs_flines", "sepsurf_rings", "spines"],
        coordsystem="spherical",
    )
    fig = ml.figure()
