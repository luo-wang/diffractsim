# create new class for simulate BOE AG_reflect effect: Color pattern (2026/3/5 whb)

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from ..util.backend_functions import backend as bd
from .diffractive_element import DOE
class AG_Reflect(DOE):
    def __init__(self, width, height, sag, n, x0 = 0, y0 = 0):
        '''create micro lens's sag,
        reflective index:n, physical size: width, height
        '''
        global bd
        self.width = width
        self.height = height
        self.x0 = x0
        self.y0 = y0
        self.sag = sag.T
        self.n = n


    def get_transmittance(self, xx, yy, λ):

        t = bd.where((((xx > (self.x0 - self.width / 2)) & (xx < (self.x0 + self.width / 2)))
                      & ((yy > (self.y0 - self.height / 2)) & (yy < (self.y0 + self.height / 2)))),
                     bd.ones_like(xx), bd.zeros_like(xx))

        nx, ny = self.sag.shape
        x_grid = np.linspace(-self.width / 2, self.width / 2, nx)  # x 网格范围
        y_grid = np.linspace(-self.height / 2, self.height / 2, ny)  # y 网格范围

        # 创建插值器 (RegularGridInterpolator 更适合规则网格数据)
        interpolator = RegularGridInterpolator((x_grid, y_grid), self.sag, bounds_error=False, fill_value=0)

        # 将 xx 和 yy 转换为插值点
        points = np.array([xx.ravel(), yy.ravel()]).T  # 转换为二维点列表

        # 对插值点进行插值
        interpolated_sag = interpolator(points)

        # 恢复为与 xx 和 yy 相同的形状
        interpolated_sag = interpolated_sag.reshape(xx.shape)

        phase_shift = -1 * interpolated_sag *(self.n) * 2 * bd.pi / λ
        tt = t * bd.exp(1j * phase_shift)
        # self.t = tt
        return tt



