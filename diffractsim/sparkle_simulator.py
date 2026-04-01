'''
sparkle_simulator.py
'''
from typing import Any
import scipy
from jax.experimental.array_api import linspace
from numpy import floating
from scipy.ndimage import gaussian_filter

from . import colour_functions as cf
from .util.constants import *
from .util.backend_functions import backend as bd
from .util.backend_functions import backend_name
from .light_sources.RGB_array import RGBarray
from .visualization import plot_intensity


class SparkleSimulator:
    def __init__(self, wavelength, extent_x, extent_y, Nx, Ny):
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
        self.wavelength = wavelength
        self.z = 0
        self.cs = cf.ColourSystem(clip_method=0)

    def add_RGB(self, spectrum = 'G', intensity=1.0 * W / m**2):
        """
        add RGB light source
        Parameters
        ----------
        spectrum: str
            'R', 'G', 'B'  to define which color channels
        intensity: float
            intensity of the light source in W/m^2
        Returns
        -------
        """
        rgbarray = RGBarray(spectrum=spectrum)
        self.pixelperiodx = rgbarray.x_period
        self.pixelperiody = rgbarray.y_period
        self.E = rgbarray.get_E(intensity, self.xx, self.yy, self.wavelength)
        I = bd.abs(self.E) ** 2
        print("Initial enerage: ", bd.sum(I)* self.dx * self.dy)
        self.plot_intensity(I, square_root=False, units=um, text='Intensity at RGB layer',colormap = 'gray')
        # self.mask = bd.zeros_like(self.E)
        #
        # self.mask[(bd.abs(self.xx) <= self.pixelperiodx / 2) &
        #           (bd.abs(self.yy) <= self.pixelperiody / 2)] = 1.0

    def prop2AG1(self, n_OLED, z_OLED, n_OCA, z_OCA, n_AG, z_AG):
        """
        propagate through the layers to the AG glass
        Parameters
        ----------
        n_OLED: refractive index of OLED layer
        z_OLED: thickness of OLED layer
        n_OCA: refractive index of OCA layer
        z_OCA: thickness of OCA layer
        n_AG: refractive index of air gap layer
        z_AG: thickness of air gap layer
        Returns
        -------
        """

        # compute angular spectrum
        fft_c = bd.fft.fft2(self.E)
        c = bd.fft.fftshift(fft_c)

        fx = bd.fft.fftshift(bd.fft.fftfreq(self.Nx, d=self.dx))
        fy = bd.fft.fftshift(bd.fft.fftfreq(self.Ny, d=self.dy))
        fxx, fyy = bd.meshgrid(fx, fy)

        argument = (2 * bd.pi) ** 2 * ((n_OLED / self.wavelength) ** 2 - fxx ** 2 - fyy ** 2)

        # Calculate the propagating and the evanescent (complex) modes
        tmp = bd.sqrt(bd.abs(argument))
        # kz1 = bd.where(argument >= 0, tmp, 1j * tmp)
        kz1 = tmp
        # eliminate the evanescent modes
        mask = bd.where(argument >= 0, 1.0, 0.0)
        # propagate the angular spectrum a distance z_OLED
        self.z += z_OLED
        Ef_OLED = c * bd.exp(1j * kz1 * z_OLED) * mask

        # propagate through OCA layer
        argument = (2 * bd.pi) ** 2 * ((n_OCA / self.wavelength) ** 2 - fxx ** 2 - fyy ** 2)
        tmp = bd.sqrt(bd.abs(argument))
        # kz2 = bd.where(argument >= 0, tmp, 1j * tmp)
        kz2 = tmp
        # eliminate the evanescent modes
        mask = bd.where(argument >= 0, 1.0, 0.0)
        self.z += z_OCA
        Ef_OCA = Ef_OLED * bd.exp(1j * kz2 * z_OCA) * mask

        # propagate through AG layer

        argument = (2 * bd.pi) ** 2 * ((n_AG / self.wavelength) ** 2 - fxx ** 2 - fyy ** 2)
        tmp = bd.sqrt(bd.abs(argument))
        # kz3 = bd.where(argument >= 0, tmp, 1j * tmp)
        kz3 = tmp
        # eliminate the evanescent modes
        mask = bd.where(argument >= 0, 1.0, 0.0)
        self.z += z_AG
        Ef_AG = Ef_OCA * bd.exp(1j * kz3 * z_AG) * mask

        self.E = bd.fft.ifft2(bd.fft.ifftshift(Ef_AG))

    def prop2AG(self, n_OLED, z_OLED, n_OCA, z_OCA, n_AG, z_AG):
        """
        Propagate optical field through multilayer structure using angular spectrum method
        with Fresnel transmission at interfaces.

        Parameters
        ----------
        n_OLED : float
        z_OLED : float
        n_OCA  : float
        z_OCA  : float
        n_AG   : float
        z_AG   : float
        """

        # ---------- FFT ----------
        Ef = bd.fft.fftshift(bd.fft.fft2(self.E))

        # self.plot_intensity(bd.abs(Ef)**2, square_root=False, units=um,text='frequency spectrum ')
        # ---------- frequency grid ----------
        fx = bd.fft.fftshift(bd.fft.fftfreq(self.Nx, d=self.dx))
        fy = bd.fft.fftshift(bd.fft.fftfreq(self.Ny, d=self.dy))
        fxx, fyy = bd.meshgrid(fx, fy)

        # ---------- wavevector ----------
        k0 = 2 * bd.pi / self.wavelength
        kx = 2 * bd.pi * fxx
        ky = 2 * bd.pi * fyy

        # ---------- layer structure ----------
        layers = [
            (n_OLED, z_OLED),
            (n_OCA, z_OCA),
            (n_AG, z_AG),
        ]

        # 默认入射介质（OLED前一层），一般是空气
        n_prev = 1.0

        for n, z in layers:
            # ---------- kz（允许复数 → evanescent 自动衰减） ----------
            kz_prev = bd.sqrt((n_prev * k0) ** 2 - kx ** 2 - ky ** 2 + 0j)
            kz = bd.sqrt((n * k0) ** 2 - kx ** 2 - ky ** 2 + 0j)

            # ---------- Fresnel transmission（标量近似，未区分偏振） ----------
            # 避免除0
            denom = (n_prev * kz_prev + n * kz) + 1e-12
            # t = 2 * n_prev * kz_prev / denom

            # ---------- 传播 ----------
            Ef = Ef  * bd.exp(1j * kz * z)

            # ---------- 更新 ----------
            n_prev = n
            self.z += z

        # ---------- IFFT ----------
        self.E = bd.fft.ifft2(bd.fft.ifftshift(Ef))

    def propAGetch(self,width, height, sag, n_AG, xyorigin=[0.,0.]):
        """
        progate through the AG etch layer: use fourier optice: convert to phase
        Parameters
        ----------
        width
        height
        sag
        n_AG

        Returns
        -------

        """

        from scipy.interpolate import RectBivariateSpline
        import numpy as np

        M, N = sag.shape
        x0 = np.linspace(-width / 2 + xyorigin[0], width / 2 + xyorigin[0], N)
        y0 = np.linspace(-height / 2 + xyorigin[1], height / 2 + xyorigin[1], M)
        if backend_name == 'cupy':
            sag_np = sag.get()
            y_np = self.y.get()
            x_np = self.x.get()
        else:
            sag_np = sag
            y_np = self.y
            x_np = self.x

        sag_np = sag_np-np.min(sag_np)
        interp = RectBivariateSpline(y0, x0, sag_np, kx=3, ky=3)
        sag_interp = interp(y_np, x_np)

        sag_interp = bd.asarray(sag_interp)

        phase_delay = bd.exp(
            1j * 2 * bd.pi * (n_AG - 1) * sag_interp / self.wavelength
        )
        self.E *= phase_delay



    def get_Sparkle_origin(self, spectrum, layer_Structure, AGetch_params, intensity = 1.0 * W / m**2, ):
        """
        get the origin sparkle pattern through inverse propagation to the RGB layer
        Parameters
        ----------
        spectrum: str
            'R', 'G', 'B'  to define which color channels
        intensity: float
            intensity of the light source in W/m^2
        layer_Structure: dict
            each dict contains 'n' and 'z' of each layer
        AGetch_params: dict
            contains 'width', 'height', 'sag', 'n_AG'

        Returns
        -------

        """

        self.add_RGB(spectrum, intensity)
        ## propagate to AG layer
        n_OLED = layer_Structure['n_OLED']
        z_OLED = layer_Structure['z_OLED']
        n_OCA = layer_Structure['n_OCA']
        z_OCA = layer_Structure['z_OCA']
        n_AG = layer_Structure['n_AG']
        z_AG = layer_Structure['z_AG']
        self.prop2AG( n_OLED, z_OLED, n_OCA, z_OCA, n_AG, z_AG)

        width = AGetch_params['width']
        height = AGetch_params['height']
        sag = AGetch_params['sag']
        xyorigin = AGetch_params['xyorigin']
        self.propAGetch(width, height, sag, n_AG, xyorigin=xyorigin )

        self.plot_intensity(bd.abs(self.E)**2, square_root=False, units=um,text='Intensity after AG etch layer')
        self.plot_phase(self.E, units=um, text='Phase after AG etch layer')
        ## inverse propagate to RGB layer
        self.prop2AG(n_AG, -z_AG, n_OCA, -z_OCA, n_OLED, -z_OLED)
        self.Iorigin = bd.abs(self.E) ** 2

        # print("Sparkle origin enerage(unfiltered): ", bd.sum(self.Iorigin)* self.dx * self.dy)
        return self.Iorigin

    #
    # def SIM(self, ):
    #     """
    #     SIM method: simulate the sparkle pattern through the entire structure
    #     Returns
    #     -------
    #
    #     """
    #     # compute I spectrum
    #     fft_c = bd.fft.fft2(self.Iorigin)
    #     c = bd.fft.fftshift(fft_c)
    #     self.plot_intensity(bd.abs(c), square_root=False, units=1/um,text='spectrum of Iorigin')
    #
    #     fx = bd.fft.fftshift(bd.fft.fftfreq(self.Nx, d=self.dx))
    #     fy = bd.fft.fftshift(bd.fft.fftfreq(self.Ny, d=self.dy))
    #     fxx, fyy = bd.meshgrid(fx, fy)
    #     # filter design
    #     px = self.pixelperiodx
    #     py = self.pixelperiody
    #
    #     # 矩形对应滤波器
    #     filterx = bd.sin(bd.pi * fxx * px) / (bd.pi * fxx * px)
    #     filterx[bd.isnan(filterx)] = 1.0
    #     filtery = bd.sin(bd.pi * fyy * py) / (bd.pi * fyy * py)
    #     filtery[bd.isnan(filtery)] = 1.0
    #     filter = filterx * filtery
    #     filter_ifft = bd.fft.ifftshift(bd.fft.ifft2((filter)))
    #     self.plot_intensity(bd.abs(filter_ifft), square_root=False, units=um,text='Point spread function of filter')
    #
    #     # 论文滤波器
    #     # filter = bd.sin(bd.pi * fxx * px) / (bd.sin(bd.pi * fxx)) * bd.sin(bd.pi * fyy * py) / (bd.sin(bd.pi * fyy)) /px/py
    #     self.plot_intensity(bd.abs(filter), square_root=False, units=1/um,text='intensity of filter')
    #
    #     # # 卷积核ft滤波
    #     # filter = bd.fft.fftshift(bd.fft.fft2(self.mask))
    #
    #     c1 = c * filter
    #     self.plot_intensity(bd.abs(c1), square_root=False, units=1/um,text='Filtered spectrum of Iorigin')
    #
    #     I_sparkle_SIM = bd.fft.ifft2(bd.fft.ifftshift(c1))
    #     self.plot_intensity(bd.real(I_sparkle_SIM), square_root=False, units=um,text="I_sparkle_SIM")
    #
    #
    #
    #     self.SparkValue = bd.std(bd.real(I_sparkle_SIM))/bd.mean(bd.real(I_sparkle_SIM))
    #     print("Sparkle Value (SIM method): ", self.SparkValue)
    #     return

    def SIM(self):
        """
        SIM method (improved):
        - Box filtering in spatial domain
        - FFT only for diagnostics
        """

        I = self.Iorigin

        # # ==============================
        # # # 1. 空间域 box 滤波（关键改动）
        # # ==============================
        # # box 尺寸（以像素为单位）
        Nx_box = int(round(self.pixelperiodx / self.dx))*1
        Ny_box = int(round(self.pixelperiody / self.dy))*1
        #
        # if Nx_box % 2 == 0:
        #     Nx_box += 1
        # if Ny_box % 2 == 0:
        #     Ny_box += 1

        kernel = bd.ones((Ny_box, Nx_box)) / (Nx_box * Ny_box)

        # 边界采用 reflect，避免周期伪影
        I_sparkle_SIM = scipy.signal.convolve2d(
            I, kernel, mode="same", boundary="symm"
        )
        #
        # self.plot_intensity(
        #     I_sparkle_SIM,
        #     square_root=False,
        #     units=um,
        #     text="I_sparkle_SIM (box filtered, spatial domain)",
        #     colormap = 'gray',
        # )

        # ##  1. 空间域高斯滤波（关键改动）
        # # ==============================
        # # sigma_x, sigma_y 以像素为单位
        # sigma_x = self.pixelperiodx / (2 * self.dx)  # 一半像素周期
        # sigma_y = self.pixelperiody / (2 * self.dy)
        #
        # I_sparkle_SIM = gaussian_filter(
        #     I,
        #     sigma=(sigma_y, sigma_x),
        #     mode='reflect'  # 边缘镜像平滑
        # )
        # self.plot_intensity(I_sparkle_SIM, square_root=False, units=um,text="I_sparkle_SIM (Gaussian filtered, spatial domain)")


        # # ==============================
        # # 2. FFT 仅用于诊断（可选）
        # # ==============================
        # c0 = bd.fft.fftshift(bd.fft.fft2(I))
        # c1 = bd.fft.fftshift(bd.fft.fft2(I_sparkle_SIM))
        #
        # self.plot_intensity(
        #     bd.abs(c0),
        #     square_root=False,
        #     units=1 / um,
        #     text="Spectrum of Iorigin"
        # )
        #
        # self.plot_intensity(
        #     bd.abs(c1),
        #     square_root=False,
        #     units=1 / um,
        #     text="Spectrum after box filtering"
        # )

        # ==============================
        # 3. Sparkle Value
        # ==============================
        print("Sparkle enerage(SIM filtered): ", bd.sum(I_sparkle_SIM) * self.dx * self.dy)
        ## 截掉周围一个像素区域
        I_sparkle_SIM = I_sparkle_SIM[Nx_box:-Nx_box, Ny_box:-Ny_box]
        self.plot_intensity(I_sparkle_SIM, square_root=False, units=um, text="I_sparkle_SIM (filtered & cropped)",colormap = 'gray')
        mean_I = bd.mean(I_sparkle_SIM)
        std_I = bd.std(I_sparkle_SIM)


        self.SparkValue = std_I / mean_I
        print("Sparkle Value (SIM method): ", self.SparkValue)

        return None






    #
    # def SIM1(self):
    #     Wx = int(round(self.pixelperiodx / self.dx))
    #     Wy = int(round(self.pixelperiody / self.dy))
    #     I_sparkle_SIM = self.box_filter_random_shift(self.Iorigin, Wx, Wy, n_samples=5)
    #     ## 截掉周围一个像素区域
    #     I_sparkle_SIM = I_sparkle_SIM[Wx:-Wx, Wy:-Wy]
    #     self.plot_intensity(I_sparkle_SIM, square_root=False, units=um, text="I_sparkle_SIM (cropped)")
    #     mean_I = bd.mean(I_sparkle_SIM)
    #     std_I = bd.std(I_sparkle_SIM)
    #
    #
    #     self.SparkValue = std_I / mean_I
    #     print("Sparkle Value (SIM method): ", self.SparkValue)
    #
    #

    #

    def DIM(self, Iorigin1):
        """
        DIM method: Difference image method
        Returns
        -------

        """

        I1 = Iorigin1
        I2 = self.Iorigin

        diffI = I1-I2
        # ==============================
        # # 1. 空间域 box 滤波（关键改动）
        # ==============================
        # box 尺寸（以像素为单位）
        Nx_box = int(round(self.pixelperiodx / self.dx))
        Ny_box = int(round(self.pixelperiody / self.dy))
        # print(Ny_box)
        # if Nx_box % 2 == 0:
        #     Nx_box += 1
        # if Ny_box % 2 == 0:
        #     Ny_box += 1

        kernel = bd.ones((Ny_box, Nx_box)) / (Nx_box * Ny_box)

        # 边界采用 reflect，避免周期伪影
        I_sparkle_DIM = scipy.signal.convolve2d(
            diffI, kernel, mode="same", boundary="symm"
        )

        # self.plot_intensity(
        #     I_sparkle_DIM,
        #     square_root=False,
        #     units=um,
        #     text="I_sparkle_DIM (box filtered, spatial domain)",
        #     colormap = 'gray',
        # )

        # ##  1. 空间域高斯滤波（关键改动）
        # # ==============================
        # # sigma_x, sigma_y 以像素为单位
        # sigma_x = self.pixelperiodx / (2 * self.dx)  # 一半像素周期
        # sigma_y = self.pixelperiody / (2 * self.dy)
        #
        # I_sparkle_DIM = gaussian_filter(
        #     diffI,
        #     sigma=(sigma_y, sigma_x),
        #     mode='reflect'  # 边缘镜像平滑
        # )
        # self.plot_intensity(I_sparkle_DIM, square_root=False, units=um,text="I_sparkle_DIM (Gaussian filtered, spatial domain)")
        #



        # ==============================
        # 2. FFT 仅用于诊断（可选）
        # ==============================
        # c0 = bd.fft.fftshift(bd.fft.fft2(diffI))
        # c1 = bd.fft.fftshift(bd.fft.fft2(I_sparkle_DIM))
        #
        # self.plot_intensity(
        #     bd.abs(c0),
        #     square_root=False,
        #     units=1 / um,
        #     text="Spectrum of diffIorigin"
        # )
        #
        # self.plot_intensity(
        #     bd.abs(c1),
        #     square_root=False,
        #     units=1 / um,
        #     text="Spectrum of diffIorigin after box filtering"
        # )

        # ==============================
        # 3. Sparkle Value
        # ==============================

        ## 截掉周围一个像素区域
        I_sparkle_DIM = I_sparkle_DIM[Nx_box:-Nx_box, Ny_box:-Ny_box]
        self.plot_intensity(I_sparkle_DIM, square_root=False, units=um, text="I_sparkle_DIM (filtered & cropped)",colormap = 'gray')
        mean_I = bd.mean(I1)
        std_I = bd.std(I_sparkle_DIM)
        self.SparkValue = std_I / mean_I
        print("Sparkle Value (DIM method): ", self.SparkValue)

        return None


    def VisualPhoto(self,sigma,):

        sigmax_pixel = sigma / self.dx
        sigmay_pixel = sigma / self.dy
        CNx = int(round(sigmax_pixel))
        CNy = int(round(sigmay_pixel))
        I_visual = gaussian_filter(self.Iorigin, sigma=(sigmax_pixel, sigmay_pixel), mode='reflect')
        I_visual = I_visual[CNx:-CNx, CNy:-CNy]
        #处理一下让最小值为0方便绘图
        I_visual[0] = 0.0
        ## gamma视觉响应
        # I_visual = I_visual**0.4

        return I_visual



    @staticmethod
    def box_filter_random_shift(I, Wx, Wy, n_samples=1):
        """
        I: 原始图像 (2D)
        Wx, Wy: box 尺寸 (像素)
        n_samples: 每个像素随机采样次数，可调增平滑效果
        """
        H, W = I.shape
        I_pad = bd.pad(I, ((Wy, Wy), (Wx, Wx)), mode='reflect')  # 边缘镜像
        S = I_pad.cumsum(axis=0).cumsum(axis=1)  # 积分图

        out = bd.zeros_like(I, dtype=bd.float64)

        for i in range(H):
            for j in range(W):
                vals = []
                for _ in range(n_samples):
                    # 随机偏移 [-0.5,0.5] 像素
                    dx = bd.random.uniform(-0.5, 0.5)
                    dy = bd.random.uniform(-0.5, 0.5)

                    x1 = int(i + Wy + dy - Wy // 2)
                    y1 = int(j + Wx + dx - Wx // 2)
                    x2 = x1 + Wy
                    y2 = y1 + Wx

                    # 用积分图求和
                    val = S[x2, y2] - S[x1, y2] - S[x2, y1] + S[x1, y1]
                    vals.append(val / (Wx * Wy))
                out[i, j] = bd.mean(vals)
        return out

    @staticmethod
    def img2sag(image, sag_mean):
        """
        convert an image to a sag profile
        Parameters
        ----------
        image: 2D array
            grayscale image
        sag_mean: float
            average sag value

        Returns
        -------
        sag: 2D array
            sag profile

        """
        img_average = bd.mean(image)
        sag = sag_mean / img_average * image
        return sag

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


    @staticmethod
    def mosaic_array(sag, N):
        """
        将二维数组sag在行和列方向上分别拼接N次，形成更大的数组

        参数:
            sag: 输入的二维数组
            N: 每个维度拼接的次数，可以是整数或二元组
               如果N是整数，则在行和列方向都拼接N次
               如果N是二元组 (N_rows, N_cols)，则分别指定行和列的拼接次数

        返回:
            拼接后的大数组
        """
        # 处理N参数
        if isinstance(N, (int, float)):
            N_rows = N_cols = int(N)
        elif isinstance(N, (tuple, list)) and len(N) == 2:
            N_rows, N_cols = int(N[0]), int(N[1])
        else:
            raise ValueError("N必须是整数或长度为2的元组/列表")

        # 使用numpy的tile函数进行拼接
        return bd.tile(sag, (N_rows, N_cols))

    from .visualization import plot_colors, plot_phase, plot_intensity, plot_longitudinal_profile_colors, \
        plot_longitudinal_profile_intensity, plot_farfield, plot_farfield_spherical_coordinates