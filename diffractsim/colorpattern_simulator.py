'''
colorpattern_simulator.py
'''
from typing import Any
import scipy
from jax.experimental.array_api import linspace
from numpy import floating
from scipy.ndimage import gaussian_filter

from .util.constants import *
from .util.backend_functions import backend as bd
from .util.backend_functions import backend_name
from .light_sources.RGB_array import RGBarray
from .visualization import plot_intensity
from diffractsim import PolychromaticField, cf, mm, cm, CircularAperture, GaussianBeam
from diffractsim import AG_Reflect,Lens

from .polychromatic_simulator_farfield import PolychromaticFieldFar

class ColorpatternSimulator:
    def __init__(self, spectrum, extent_x, extent_y, Nx, Ny, Saperture=0):
        '''
        initializes the field, representing the cross-section profile of a plane wave
        Parameters
        ----------
        wavelength: wavelength of the light wave
        extent_x: length of the field in the x direction
        extent_y: length of the field in the y direction
        Nx: number of grid in the x direction
        Ny: number of grid in the y direction
        '''
        self.spectrum = spectrum
        self.extent_x = extent_x
        self.extent_y = extent_y
        self.dx = extent_x / Nx
        self.dy = extent_y / Ny
        self.x = bd.linspace(-extent_x/2, extent_x/2, Nx+1)[:-1]
        self.y = bd.linspace(-extent_y/2, extent_y/2, Ny+1)[:-1]
        self.xx, self.yy = bd.meshgrid(self.x, self.y)
        self.Nx = Nx
        self.Ny = Ny
        # self.E = bd.ones((self.Nx, self.Ny))*bd.sqrt(intensity)
        self.z = 0
        self.cs = cf.ColourSystem(clip_method=0)
        self.Saperture = Saperture

    def DoSimulate(self, z_LighttoAG, AGetch_params):

        F = PolychromaticFieldFar(spectrum=self.spectrum, extent_x=self.extent_x,extent_y=self.extent_y, Nx=self.Nx, Ny=self.Ny,spectrum_divisions = 10)
        # F = PolychromaticField(spectrum=self.spectrum, extent_x=self.extent_x, extent_y=self.extent_y, Nx=self.Nx, Ny=self.Ny, spectrum_divisions=10)
        width = AGetch_params['width']
        height = AGetch_params['height']
        sag = AGetch_params['sag']
        ## 添加球面相位
        F.add(Lens(f=z_LighttoAG))

        # F.add(CircularAperture(self.Saperture))
        # F.add(GaussianBeam(0.1*mm))
        # rgb = F.get_colors()
        # F.plot_colors(rgb)

        # F.propagate(z_LighttoAG)
        # rgb = F.get_colors()
        # F.plot_colors(rgb)

        F.add(AG_Reflect(width,height,sag,1))

        # rgb = F.get_colors()
        # F.plot_colors(rgb)

        F.propagate(z_LighttoAG)
        rgb = F.get_colors()
        F.plot_colors(rgb)

        R, G, B = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        colorpattern = bd.mean(
            bd.sqrt(
                ((R - G) ** 2 + (G - B) ** 2 + (B - R) ** 2) / 3
            )
        )
        print('colorpattern: ', colorpattern)