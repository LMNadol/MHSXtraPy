# example file to plot radial cuts through global magnetic field models of the solar corona
# can produce the majority of plots found in the Williams (2018)

import numpy as np
import matplotlib.pyplot as plt
from . import read as rd
import os
from random import shuffle
from fractions import Fraction as fr

plt.ion()

def latex_ticks(nsplit, npi):
    """
    Produces strings for the ticks used for some spherical coordinate plots in
    latex for multiples of pi
    """
    list = []
    for i in range(nsplit+1):
        string = r'$'
        fract = fr(npi*i,nsplit)
        if fract.numerator == 0:
            string = r'$0$'
        elif fract.denominator != 1:
            string = string + r'\frac{'
            if fract.numerator != 1:
                string = string + str(fract.numerator) 
            string = string + r'\pi' + r'}{' + str(fract.denominator) + r'}$'
        else:
            if fract.numerator != 1:
                string = string + str(fract.numerator)
            string = string + r'\pi$'
        list.append(string)
    return list

#################################################################

class PlotCut:

    def __init__(self, filename, norm, d,
                 title=False, levels=np.linspace(-20,20,101), colmap=plt.cm.Greys_r,
                 nticks=4, newcolours=True, linerast=False, output_dir=None):
        self.datafile = filename
        self.prefix = rd.prefix(filename)

        if output_dir is not None:
            rd.outprefix = output_dir

        self.nulldata = rd.nulls(self.datafile)

        self.plane_norm = norm
        try:
            len(d)
        except TypeError:
            self.plane_d = d
        else:
            pt = np.array(d, dtype=np.float64)
            self.plane_d = np.sum(pt*norm)

        self.figure = plt.figure(figsize=(15/2.55, 6.6/2.55)) # set-up for A4
        self.ax = plt.gca()
        self.ax.set_xlabel(r'Longitude, \(\phi\)')
        self.ax.set_xlim([0, 2*np.pi])
        self.ax.set_xticks(np.linspace(0, 2*np.pi, nticks+1), latex_ticks(nticks, 2))
        self.ax.set_ylabel(r'Colatitude, \(\theta\)')
        self.ax.set_ylim([np.pi, 0])
        self.ax.set_yticks(np.linspace(0, np.pi, nticks+1), latex_ticks(nticks, 1))

        if linerast == False:
            self.ax.set_rasterization_zorder(-6)
        else:
            self.ax.set_rasterization_zorder(0)
            
        self.br, _, _, self.rads, self.thetas, self.phis = rd.field(self.datafile)
        ir = np.where(self.plane_d >= self.rads)[0].max()
        plotfield = self.br[ir, :, :] + (self.plane_d - self.rads[ir])/(self.rads[ir+1] - self.rads[ir])*(self.br[ir+1, :, :] - self.br[ir, :, :])
        contours = self.ax.contourf(self.phis, self.thetas, plotfield, levels, cmap=colmap, extend='both', zorder=-10)
        cb = self.figure.colorbar(contours, ax=self.ax, fraction=0.05, pad=0.025)
        diff = levels[-1] - levels[0]
        cb.set_ticks(np.linspace(0, 1.0, 5)*diff + levels[0])
        cb.set_label('Radial Field Strength (G)')
        self.figure.tight_layout()
        if title:
            rstr = str(d)
            self.ax.set_title('Cut: ' + self.datafile + ' r = ' + rstr)

        if newcolours:
            self.colours = self.get_colours()


    def all(self, labels=False, ms=3):
        """
        Just calls all other functions for convenience
        """
        lw = 1
        self.rings(labels=labels, lw=lw)
        self.hcs(lw=lw)
        self.spines(labels=labels, ms=ms)
        self.separators(labels=labels, ms=ms)
        self.separators(labels=labels, ms=ms, hcs=True)

    #################################################################

    def spines(self, labels=False, ms=3):
        """
        Plots all the spine points
        """
        null_nums, spines = rd.cut_spines(self.datafile, self.plane_norm, self.plane_d)
        for inull, spine in zip(null_nums, spines):
            self.ax.plot(spine[2], spine[1], '.', c=self.colours[inull-1], ms=ms, zorder=-5)
            if labels:
                self.ax.text(spine[2], spine[1], '{}'.format(inull), color='green')

    #################################################################

    def separators(self, labels=False, ms=3, hcs=False):
        """
        Plots all the separator points
        """
        null_nums, separators = rd.cut_separators(self.datafile, self.plane_norm, self.plane_d, hcs=hcs)
        col = 'orange' if hcs else 'yellow'
        for inull, sep in zip(null_nums, separators):
            self.ax.plot(sep[2], sep[1], '*', c=col, ms=ms, zorder=-5)
            if labels:
                self.ax.text(sep[2], sep[1], '{}'.format(inull))

    #################################################################

    def rings(self, dots=False, labels=False, lw=1):
        """
        Plots all the lines denoting the ring crossings
        """
        null_nums, rings = rd.cut_sepsurf(self.datafile, self.plane_norm, self.plane_d)
        for inull, ring in zip(null_nums, rings):
            self.ax.plot(ring[:, 2], ring[:, 1], c=self.colours[inull-1], lw=lw, zorder=-5)
            if labels:
                self.ax.text(ring[ring.length[0]//2, 2], ring[ring.length[0]//2, 1], '{}'.format(inull))
            if dots:
                self.ax.plot(ring[:, 2], ring[:, 1], '.', c='green')

    #################################################################

    def hcs(self, dots=False, lw=1):
        """
        Plots all the hcs crossings
        """
        lines = rd.cut_hcs(self.datafile, self.plane_norm, self.plane_d)
        for line in lines:
            self.ax.plot(line[:, 2], line[:, 1], c='lime', ls=':', lw=lw, zorder=-0.5)
            if dots:
                self.ax.plot(line[:, 2], line[:, 1], '.', c='blue')

    #################################################################

    def nulls(self, labels=False, size=1, zorder=-1):
        """
        Plots all the null point locations in theta/phi
        """
        for i in range(self.nulldata.shape[0]):
            self.ax.plot(self.nulldata.pos[i, 2], self.nulldata.pos[i, 1], '.', color=self.colours[i], ms=size, zorder=zorder)
            if labels == True:
                self.ax.text(self.nulldata.pos[i,2], self.nulldata.pos[i,1], '{}, {:04f}'.format(i, self.nulldata.pos[i,0]))

    #################################################################

    def field(self):
        """
        Plots the background radial field on the solar surface
        """
        br, _, _, rads, thetas, phis = rd.field(self.datafile)
        levels = np.linspace(-10, 10, 101)
        # plt.contourf(phis, thetas, br[0, :, :], levels, cmap=plt.cm.RdBu_r, extend='both', vmax=10, vmin=-10, zorder=-10)
        self.ax.contourf(phis, thetas, br[0, :, :], levels, cmap=plt.cm.RdBu_r, extend='both', vmax=10, vmin=-10, zorder=-10)
        # cb = plt.colorbar(fraction=0.05, pad=0.025)
        cb = self.figure.colorbar(fraction=0.05, pad=0.025)
        cb.set_ticks([-10, -5, 0, 5, 10])
        cb.set_label('Magnetic Field Strength (G)')
        self.figure.tight_layout()

    def get_colours(self):
        """
        Pick random colours for each null point in shades of red and blue
        """
        negcolors, poscolors = plt.get_cmap('Blues'), plt.get_cmap('Reds')
        negcolors = [negcolors(i) for i in np.linspace(0.5, 1, np.count_nonzero(self.nulldata.sign == -1))]
        poscolors = [poscolors(i) for i in np.linspace(0.5, 1, np.count_nonzero(self.nulldata.sign == 1))]
        shuffle(poscolors)
        shuffle(negcolors)
        colours, ipos, ineg = [], 0, 0
        for inull in range(self.nulldata.number[-1]):
            if self.nulldata[inull].sign == -1:
                col = negcolors[ineg]
                ineg += 1
            elif self.nulldata[inull].sign == 1:
                col = poscolors[ipos]
                ipos += 1
            colours.append(col)
        
        return colours
