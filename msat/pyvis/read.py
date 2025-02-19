import numpy as np
import glob
import sys
import os.path

# if system has enough memory - could read in all at once and then split

files = sorted(glob.glob("data/*.dat"))

outdir = "output"


def prefix(filename):
    dot = filename[::-1].find(".")
    slash = filename[::-1].find("/")
    return filename[-slash : -dot - 1]


def synmap(filename):
    with open(filename, "rb") as fieldfile:
        nlon, nlat = np.fromfile(fieldfile, count=2, dtype=np.int32)

        bsyn = np.fromfile(fieldfile, count=nlon * nlat, dtype=np.float64).reshape(
            (nlon, nlat), order="f"
        )

        lons = np.fromfile(fieldfile, count=nlon, dtype=np.float64)
        lats = np.fromfile(fieldfile, count=nlat, dtype=np.float64)

    return bsyn, lons, lats


def field(filename, grid=False):
    with open(filename, "rb") as fieldfile:
        shape = np.fromfile(fieldfile, count=3, dtype=np.int32)
        nx, ny, nz = (int(n) for n in shape)
        num = nx * ny * nz

        bx = np.fromfile(fieldfile, count=num, dtype=np.float64).reshape(
            shape, order="f"
        )
        by = np.fromfile(fieldfile, count=num, dtype=np.float64).reshape(
            shape, order="f"
        )
        bz = np.fromfile(fieldfile, count=num, dtype=np.float64).reshape(
            shape, order="f"
        )

        x = np.fromfile(fieldfile, count=nx, dtype=np.float64)
        y = np.fromfile(fieldfile, count=ny, dtype=np.float64)
        z = np.fromfile(fieldfile, count=nz, dtype=np.float64)

    if grid:
        return np.stack((bx, by, bz), axis=-1), x, y, z
    else:
        return bx, by, bz, x, y, z


def nulls(filename, simple=False):
    # create datatypes depending on simple or not
    if simple:
        dtype = [
            ("number", np.int32),
            ("pos", np.float64, 3),
            ("gridpos", np.float64, 3),
        ]
    else:
        dtype = [
            ("number", np.int32),
            ("pos", np.float64, 3),
            ("gridpos", np.float64, 3),
            ("sign", np.int32),
            ("spine", np.float64, 3),
            ("fan", np.float64, 3),
            ("warning", np.int32),
        ]

    with open(f"{outdir}/{prefix(filename)}-nullpos.dat", "rb") as nullfile:
        # get three data sets from the null finder file
        (nnulls,) = np.fromfile(nullfile, dtype=np.int32, count=1)
        # create record array
        nulls = np.recarray(nnulls, dtype=dtype)
        # start reading and storing data
        nulls.gridpos = np.fromfile(
            nullfile, dtype=np.float64, count=3 * nnulls
        ).reshape(nnulls, 3)
        nulls.pos = np.fromfile(nullfile, dtype=np.float64, count=3 * nnulls).reshape(
            nnulls, 3
        )
        nulls.number = np.arange(1, nnulls + 1)

    if not simple:
        # if you want the sign finder data too...
        with open(f"{outdir}/{prefix(filename)}-nulldata.dat", "rb") as nullfile:
            (nnulls,) = np.fromfile(nullfile, dtype=np.int32, count=1)
            nulls.sign = np.fromfile(nullfile, dtype=np.int32, count=nnulls)
            nulls.spine = np.fromfile(
                nullfile, dtype=np.float64, count=3 * nnulls
            ).reshape(nnulls, 3)
            nulls.fan = np.fromfile(
                nullfile, dtype=np.float64, count=3 * nnulls
            ).reshape(nnulls, 3)
            nulls.warning = np.fromfile(nullfile, dtype=np.int32, count=nnulls)

    return nulls


def separators(filename, null_list=None, lines=True, connectivity=True, hcs=False):
    # read in null data
    nulldata = nulls(filename, simple=True)

    if not hcs:
        # filenames for the nulls
        connectivityfile = f"{outdir}/{prefix(filename)}-connectivity.dat"
        separatorsfile = f"{outdir}/{prefix(filename)}-separators.dat"
        allnulls_list = nulldata.number
    else:
        # filenames for the hcs
        connectivityfile = f"{outdir}/{prefix(filename)}-hcs-connectivity.dat"
        separatorsfile = f"{outdir}/{prefix(filename)}-hcs-separators.dat"
        allnulls_list = [1]  # need to fix this

    # if none set, read in data for all nulls
    if null_list is None:
        null_list = allnulls_list

    # create final lists
    conlist = []
    seplist = []

    with open(connectivityfile, "rb") as sepinfo:
        with open(separatorsfile, "rb") as seps:
            for inull in allnulls_list:
                coni = []
                sepi = []
                (nseps,) = np.fromfile(sepinfo, dtype=np.int32, count=1)
                for _ in range(nseps):
                    sepinfo.seek(4, 1)
                    coni.append(
                        np.asscalar(np.fromfile(sepinfo, dtype=np.int32, count=1))
                    )
                    sepinfo.seek(8, 1)
                    (length,) = np.fromfile(seps, dtype=np.int32, count=1)
                    sepi.append(
                        np.fromfile(seps, dtype=np.float64, count=3 * length).reshape(
                            -1, 3
                        )
                    )
                if inull in null_list:
                    # add data to final list, empty if null not in null_list
                    conlist.append(coni)
                    seplist.append(sepi)
                else:
                    conlist.append([])
                    seplist.append([])

    # return data based on keyword arguments
    if not lines:
        return conlist
    elif not connectivity:
        return seplist
    else:
        return seplist, conlist


def spines(filename, null_list=None):
    # read in null data
    nulldata = nulls(filename, simple=True)

    # do all nulls if null_list set to none
    if null_list is None:
        null_list = nulldata.number

    spinelist = []

    with open(f"{outdir}/{prefix(filename)}-spines.dat", "rb") as spinefile:
        for inull in nulldata.number:
            spinelisti = []
            for _ in range(2):  # spine in each direction
                (length,) = np.fromfile(spinefile, dtype=np.int32, count=1)
                if inull in null_list:
                    spinelisti.append(
                        np.fromfile(
                            spinefile, dtype=np.float64, count=3 * length
                        ).reshape(-1, 3)
                    )
                else:
                    # skip spine if not required
                    spinefile.seek(length * 3 * 8, 1)
            spinelist.append(spinelisti)

    return spinelist


def ringinfo(filename):

    nulldata = nulls(filename, simple=True)

    with open(f"{outdir}/{prefix(filename)}-ringinfo.dat", "rb") as ringinfo:
        ringsmax, writeskip, bytesize = np.fromfile(ringinfo, dtype=np.int32, count=3)
        (stepsize,) = np.fromfile(ringinfo, dtype=np.float64, count=1)
        ringsmax1 = np.ceil(ringsmax / writeskip).astype(np.int)
        ringnums = []
        for _ in nulldata.number:
            ringnums.append(np.fromfile(ringinfo, dtype=np.int32, count=ringsmax1))

    return ringsmax, writeskip, bytesize, stepsize, ringnums


def rings(filename, breaks=False, assocs=False, nskip=1, null_list=None, hcs=False):

    if nskip != 1 and assocs:
        print(
            "Warning: Associations not corrected for when nskip != 1. May be updated in future."
        )

    nulldata = nulls(filename, simple=True)

    ringlist = []
    breaklist = []
    if assocs:
        assoclist = []

    fprefix = f"{outdir}/{prefix(filename)}"

    if hcs:
        assocs_filename = f"{fprefix}-hcs-assocs.dat"
        info_filename = f"{fprefix}-hcs-ringinfo.dat"
        ring_filename = f"{fprefix}-hcs-rings.dat"
        break_filename = f"{fprefix}-hcs-breaks.dat"
    else:
        assocs_filename = f"{fprefix}-assocs.dat"
        info_filename = f"{fprefix}-ringinfo.dat"
        ring_filename = f"{fprefix}-rings.dat"
        break_filename = f"{fprefix}-breaks.dat"

    assoc_exist = os.path.isfile(assocs_filename)
    if assocs and not assoc_exist:
        print("Not reading in associations: file does not exist")
    do_assocs = assocs and assoc_exist

    ringinfo = open(info_filename, "rb")
    ringfile = open(ring_filename, "rb")
    brkfile = open(break_filename, "rb")

    ringsmax, writeskip, bytesize = np.fromfile(ringinfo, dtype=np.int32, count=3)
    if bytesize == 4:
        floattype = np.float32
    elif bytesize == 8:
        floattype = np.float64
    _ = np.fromfile(ringinfo, dtype=np.float64, count=1)
    if hcs:
        (n_hcs,) = np.fromfile(ringinfo, dtype=np.int32, count=1)
    ringsmax = np.ceil(ringsmax / writeskip).astype(np.int)

    if hcs:
        to_do = range(1, n_hcs // 2 + 1)
        ndir = 2
        name = "hcs"
    else:
        to_do = nulldata.number
        ndir = 1
        name = "null"

    if null_list is None and not hcs:
        null_list = nulldata.number
    elif hcs:
        null_list = to_do

    for inull in to_do:
        print("Reading rings from {} {:5d}".format(name, inull))
        sys.stdout.write("\033[F")
        for _ in range(ndir):
            breaklisti = []
            ringlisti = []
            lengths = np.fromfile(ringinfo, dtype=np.int32, count=ringsmax)
            if inull in null_list:
                for iring, length in enumerate(lengths[lengths != 0]):
                    if iring % nskip == 0:
                        breaklisti.append(
                            np.fromfile(brkfile, dtype=np.int32, count=length)
                        )
                        ringlisti.append(
                            np.fromfile(
                                ringfile, dtype=floattype, count=3 * length
                            ).reshape(-1, 3)
                        )
                    else:
                        # skip ring if not a multiple of nskip
                        ringfile.seek(length * 3 * bytesize, 1)
                        brkfile.seek(length * 4, 1)
            else:
                # skip all rings from null if not required
                total = lengths.sum(dtype=np.int64)
                ringfile.seek(total * 3 * bytesize, 1)
                brkfile.seek(total * 4, 1)
            ringlist.append(ringlisti)
            breaklist.append(breaklisti)

    ringinfo.close()
    ringfile.close()
    brkfile.close()

    if do_assocs:
        assfile = open(assocs_filename, "rb")
        ringinfo = open(info_filename, "rb")
        np.fromfile(ringinfo, dtype=np.int32, count=3)
        np.fromfile(ringinfo, dtype=np.float64, count=1)
        for inull in nulldata.number:
            assoclisti = []
            lengths = np.fromfile(ringinfo, dtype=np.int32, count=ringsmax)
            if inull in null_list:
                for iring, length in enumerate(lengths[lengths != 0]):
                    if iring % nskip == 0:
                        assoclisti.append(
                            np.fromfile(assfile, dtype=np.int32, count=length)
                        )
                    else:
                        assfile.seek(length * 4, 1)
            else:
                assfile.seek(lengths.sum(dtype=np.int64) * 4, 1)
            assoclist.append(assoclisti)
        assfile.close()
        ringinfo.close()

    if breaks and not do_assocs:
        return ringlist, breaklist
    if not breaks and do_assocs:
        return ringlist, assoclist
    if breaks and do_assocs:
        return ringlist, breaklist, assoclist
    else:
        return ringlist


def get_cut_filename_plane(normal, d):
    try:
        len(d)
    except TypeError:
        d_pln = d
    else:
        pt = np.array(d, dtype=np.float64)
        d_pln = np.sum(pt * normal)

    l = [*normal, d_pln]
    fmt = "_".join(r"{:08.4f}" if i >= 0 else r"{:09.4f}" for i in l)

    return fmt.format(*normal, d_pln)


def cut_sepsurf(filename, normal, d):
    filename_plane = get_cut_filename_plane(normal, d)
    null_nums = []
    sepsurfs = []
    with open(
        f"{outdir}/{prefix(filename)}-rings-cut_{filename_plane}.dat", "rb"
    ) as ringfile:
        (nlines,) = np.fromfile(ringfile, dtype=np.int32, count=1)
        for _ in range(nlines):
            (inull,) = np.fromfile(ringfile, dtype=np.int32, count=1)
            null_nums.append(inull)
            (length,) = np.fromfile(ringfile, dtype=np.int32, count=1)
            sepsurf = np.fromfile(ringfile, dtype=np.float64, count=3 * length).reshape(
                -1, 3
            )
            sepsurfs.append(sepsurf)
    return null_nums, sepsurfs


def cut_separators(filename, normal, d, hcs=False):
    filename_plane = get_cut_filename_plane(normal, d)
    null_nums = []
    sep_pts = []
    if hcs:
        with open(
            f"{outdir}/{prefix(filename)}-hcs-separators-cut_{filename_plane}.dat", "rb"
        ) as sepfile:
            (npts,) = np.fromfile(sepfile, dtype=np.int32, count=1)
            for _ in range(npts):
                (end,) = np.fromfile(sepfile, dtype=np.int32, count=1)
                null_nums.append(end)
                sep = np.fromfile(sepfile, dtype=np.float64, count=3)
                sep_pts.append(sep)
    else:
        with open(
            f"{outdir}/{prefix(filename)}-separators-cut_{filename_plane}.dat", "rb"
        ) as sepfile:
            (npts,) = np.fromfile(sepfile, dtype=np.int32, count=1)
            for _ in range(npts):
                _, end = np.fromfile(sepfile, dtype=np.int32, count=2)
                null_nums.append(end)
                sep = np.fromfile(sepfile, dtype=np.float64, count=3)
                sep_pts.append(sep)
    return null_nums, sep_pts


def cut_spines(filename, normal, d):
    filename_plane = get_cut_filename_plane(normal, d)
    null_nums = []
    spine_pts = []
    with open(
        f"{outdir}/{prefix(filename)}-spines-cut_{filename_plane}.dat", "rb"
    ) as spinefile:
        (npts,) = np.fromfile(spinefile, dtype=np.int32, count=1)
        for _ in range(npts):
            (inull,) = np.fromfile(spinefile, dtype=np.int32, count=1)
            null_nums.append(inull)
            spine = np.fromfile(spinefile, dtype=np.float64, count=3)
            spine_pts.append(spine)
    return null_nums, spine_pts


def cut_hcs(filename, normal, d):
    filename_plane = get_cut_filename_plane(normal, d)
    lines = []
    with open(
        f"{outdir}/{prefix(filename)}-hcs-cut_{filename_plane}.dat", "rb"
    ) as hcsfile:
        (nlines,) = np.fromfile(hcsfile, dtype=np.int32, count=1)
        for _ in range(0, nlines):
            _ = np.fromfile(hcsfile, dtype=np.int32, count=1)
            (length,) = np.fromfile(hcsfile, dtype=np.int32, count=1)
            lines.append(
                np.fromfile(hcsfile, dtype=np.float64, count=3 * length).reshape(-1, 3)
            )

    return lines


def set_output_dir(loc):
    global outdir
    if loc[-1] == "/":
        loc = loc[:-1]
    outdir = loc
    print('Changed output data directory to be "./{}"'.format(outdir))
