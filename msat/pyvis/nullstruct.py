from . import read as rd
import numpy as np
import pyvis.fieldline3d as fl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def plot(n, filename, converge=True, fan=False, ball=True, rsphere=1e-4, h0=5e-2,
    coordsystem='cartesian', draw_scale=1, figsize=(10, 10), sf_debug=False):

    if converge == True:
        if coordsystem == 'cartesian':
            coord = 'xyz'
        elif coordsystem == 'cylindrical':
            coord = 'rpz'
        elif coordsystem == 'spherical':
            coord = 'rtp'
        makestr = 'make sf{}'.format(coord)
        if sf_debug:
            makestr = makestr + ' debug=on'
            print('Removing sf exectutable to recompile in debug')
            os.system('rm -f sf{}'.format(coord))
        os.system('make sf{} debug=on'.format(coord))
        os.system('./sf{0} -i {1} -n {2:04d}'.format(coord, filename, n))

    bgrid, xgc, ygc, zgc = rd.field(filename, grid=True)
    
    xgc = np.arange(bgrid.shape[0])
    ygc = np.arange(bgrid.shape[1])
    zgc = np.arange(bgrid.shape[2])

    nulls = rd.nulls(filename, simple=True)
    nullpos = nulls[n-1].gridpos-1 # -1 translation from fortran to idl grid spacing

    xsize = 20
    xthick = 5

    plt.figure(figsize=figsize, tight_layout=True)
    plt.axes(projection='3d')
    plt.plot(*nullpos[:, None], 'rx ', ms=xsize, mew=xthick)

    plt.axis('off')

    if converge:
        fnames = ['maxvec.dat', 'minvec.dat', 'spine.dat']
        col = ['black', 'green', 'red']

        for j, name in enumerate(fnames):
            with open('output/' + name, 'rb') as file:
                ny = np.asscalar(np.fromfile(file, count=1, dtype=np.int32))
                r = np.fromfile(file, count=3*ny, dtype=np.float64).reshape(ny, 3).T
                r = r * rsphere

                for i in range(3):
                    r[i, :] = r[i, :] + nullpos[i]

                plt.plot(*r, 'x ', c=col[j])

                if name == fnames[0]:
                    maxvec = np.fromfile(file, count=3, dtype=np.float64)
                    max1 = maxvec.copy()
                    maxvec = maxvec*rsphere
                    maxvec = maxvec + nullpos
                    plt.plot(*maxvec[:, None],'x ', c='cyan', ms=xsize, mew=xthick)

                if name == fnames[1]:
                    minvec = np.fromfile(file, count=3, dtype=np.float64)
                    min1 = minvec.copy()
                    minvec = minvec*rsphere
                    minvec = minvec + nullpos
                    plt.plot(*minvec[:, None],'x ', c='green', ms=xsize, mew=xthick)

                if name == fnames[2]:
                    spine = np.fromfile(file, count=3, dtype=np.float64)
                    spine = spine*rsphere
                    spine = spine + nullpos
                    plt.plot(*spine[:, None],'x ', c='blue', ms=xsize, mew=xthick)

        npts = 121
        angles = np.linspace(0, 2*np.pi, npts)
        fanpts = np.array([(max1*np.cos(angle) + min1*np.sin(angle))*rsphere + nullpos for angle in angles])
        plt.plot(*fanpts.T, c='black')

        spineline = np.array([spine, 2*nullpos - spine])
        plt.plot(*spineline.T, c='blue')
    
    boxedge = np.zeros((2, 3), dtype=np.float64)
    for i in range(3):
        boxedge[:, i] = np.array([nullpos[i] - draw_scale*rsphere, nullpos[i] + draw_scale*rsphere])
    boxedge = np.array([[nullpos[i] - draw_scale*rsphere, nullpos[i] + draw_scale*rsphere] for i in range(3)]).T
    
    startpts_fan = []
    if fan == True:
        rad = 0.6*rsphere
        nphi = 20
        phis = 2*np.pi*(np.arange(nphi)+0.5)/nphi
        startpts_fan = [rad*(max1*np.cos(phi[i]) + min1*np.sin(phi[i])) for phi in phis]
    
    startpts_ball = []
    if ball:
        rad = rsphere*5/2e1
        ntheta = 6
        nphi = 6
        thetas = np.pi*((np.arange(ntheta)+0.5)/ntheta)
        phis = 2*np.pi*((np.arange(nphi)+0.5)/nphi)
        startpts_ball = [np.array([rad*np.cos(ip)*np.sin(it), rad*np.sin(ip)*np.sin(it), rad*np.cos(it)]) for it in thetas for ip in phis]
    
    startpts = np.array([*startpts_fan, *startpts_ball])

    if ball or fan:
        h0 = rsphere*h0
        for startpt in startpts:
            h = h0
            line = fl.fieldline3d(startpt + nullpos, bgrid, xgc,ygc,zgc, h, 0.1*h, 10*h, 0.01*h,
                boxedge=boxedge, oneway=True, gridcoord=True, coordsystem=coordsystem,
                stop_criteria=False, periodicity=None)
            plt.plot(*line.T, c='red')

            h = h0
            line = fl.fieldline3d(startpt + nullpos, bgrid, xgc,ygc,zgc, -h, 0.1*h, 10*h, 0.01*h,
                boxedge=boxedge, oneway=True, gridcoord=True, coordsystem=coordsystem,
                stop_criteria=False, periodicity=None)
            plt.plot(*line.T, c='orange')

        plt.suptitle(r'File: {} Null: {}'.format(filename, n).replace('_', r'\_'))

    print("----------------------------------------")
    print("Key:")
    print("----")
    print("Null - Red cross")
    print("Spine - Dark blue cross")
    print("Spine line - Blue line")
    print("Maxvec - Cyan cross")
    print("Minvec - Green cross")
    print("Fan plane - Black circle")
    print("Converged fan - Small black crosses")
    print("Converged spine - Small red crosses")
    print("Lines traced forwards - Red lines")
    print("Lines traced backwards - Orange Lines")
    print("----------------------------------------")
