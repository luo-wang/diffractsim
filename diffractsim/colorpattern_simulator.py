'''
colorpattern_simulator.py
'''
import matplotlib.pyplot as plt
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
from diffractsim import PolychromaticField, cf, mm,nm, cm, CircularAperture, GaussianBeam
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

        F = PolychromaticFieldFar(spectrum=self.spectrum, extent_x=self.extent_x,extent_y=self.extent_y, Nx=self.Nx, Ny=self.Ny,spectrum_divisions = 4)
        # F = PolychromaticField(spectrum=self.spectrum, extent_x=self.extent_x, extent_y=self.extent_y, Nx=self.Nx, Ny=self.Ny, spectrum_divisions=10)
        width = AGetch_params['width']
        height = AGetch_params['height']
        sag = AGetch_params['sag']
        ## 添加球面相位
        # F.add(Lens(f=z_LighttoAG))

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

    def DoSimulate2(self, z_LighttoAG, AGetch_params):
        # F = PolychromaticFieldFar(spectrum=self.spectrum, extent_x=self.extent_x,extent_y=self.extent_y, Nx=self.Nx, Ny=self.Ny,spectrum_divisions = 2)
        width = AGetch_params['width']
        height = AGetch_params['height']
        sag = AGetch_params['sag']


        #    Reference:
        # [1] Computational fourier optics : a MATLAB tutorial. Chapter 5, section 5.5.
        ps = self.dx
        # print(ps)
        Wimg, Himg = self.Nx, self.Ny
        # side length
        wvln = 550*nm
        k = 2 * bd.pi / wvln
        ag_ref = AG_Reflect(width,height,sag,1)
        E = ag_ref.get_transmittance(self.xx, self.yy, wvln)

        # plt.figure()
        # plt.imshow(bd.abs(E)**2, cmap='gray')
        # plt.colorbar()
        # plt.title('Intensity pattern before Fraunhofer method')
        #
        # plt.figure()
        # plt.imshow(bd.angle(E), cmap='jet')
        # plt.colorbar()
        # plt.title('Intensity pattern before Fraunhofer method')
        # plt.show()

        # Compute x, y, fx, fy
        L2 = wvln * z_LighttoAG / ps

        print(f"Fraunhofer method: L2 = {L2:.4f} m")
        x2, y2 = bd.meshgrid(
            bd.linspace(-L2 / 2, L2 / 2, Wimg, device=E.device),
            bd.linspace(-L2 / 2, L2 / 2, Himg, device=E.device),
            indexing="xy",
        )

        h = 1/ wvln/z_LighttoAG
        u = h  * bd.fft.ifftshift(bd.fft.fft2(bd.fft.fftshift(E)))*ps*ps
        I = bd.abs(u) ** 1


        # optional: sensor blur
        from scipy.ndimage import gaussian_filter
        I = gaussian_filter(I, sigma=1)
        I = I / I.max()
        # I[I > 0.01] = 0.01


        # optional: camera simulation
        # avg pooling

        # display
        plt.figure()
        plt.imshow(
            (I + 1e-8),
            cmap='gray'
        )
        plt.colorbar()
        plt.show()

    @staticmethod
    def jetimg2sag(image_path, v_min, v_max):
        import numpy as np
        from PIL import Image
        import matplotlib.pyplot as plt
        from scipy.spatial import cKDTree

        N_COLORS = 256

        # 1. 读图
        img = Image.open(image_path).convert("RGB")
        img = np.asarray(img, dtype=np.float32) / 255.0
        H, W, _ = img.shape
        print(f"Etch Image loaded: {H} x {W}")

        # 2. jet colormap
        jet = plt.get_cmap("jet", N_COLORS)
        jet_rgb = jet(np.linspace(0.0, 1.0, N_COLORS))[:, :3]

        # 3. 构建KDTree
        tree = cKDTree(jet_rgb)

        # 4. 查询最近颜色
        img_flat = img.reshape(-1, 3)
        _, idx = tree.query(img_flat, k=1)

        # 5. 归一化
        value_norm = idx.astype(np.float32) / (N_COLORS - 1)
        value_norm = (value_norm - value_norm.min()) / (value_norm.max() - value_norm.min())
        value_norm = value_norm.reshape(H, W)

        # 6. 映射物理量
        value = v_min + value_norm * (v_max - v_min)

        return value
