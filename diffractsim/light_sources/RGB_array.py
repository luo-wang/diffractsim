from ..util.backend_functions import backend as bd

from matplotlib.path import Path
from .light_source import LightSource
from diffractsim import  mm, um
class RGBarray(LightSource):
    def __init__(self, spectrum='G',Module = 13.2):
        """
        Creates a light source defined by a 2D RGB pixel array.

        Parameters:
        width : float
            lightsource width.
        height : float
            lightsource height.
        x_period : float
            Period in x direction.
        y_period : float
            Period in y direction.
        spectrum : str
            'R', 'G', 'B'  to define which color channels
        """

        # self.width = width
        # self.height = height
        if Module == 13.2:
            self.x_period = 96.8*um
            self.y_period = 96.8*um
        elif Module == 8.8:
            self.x_period = 74.0*um
            self.y_period = 74.0*um
        elif Module == 13.8:
            self.x_period = 114.9*um
            self.y_period = 116.3*um
        else:
            raise ValueError("请输入正确的Module选项: 8.8, 13.2 或 13.8")

        self.Moudle = Module
        self.spectrum = spectrum


    def get_E1(self, E2, xx, yy, λ,div=1, effective_area=1):
        """
        Returns the electric field distribution defined by the pixel array.

        Parameters:
        E : float or complex
            The amplitude scaling factor for the electric field.
        xx :
            The x-coordinates grid .
        yy :
            The y-coordinates grid .
        λ : float
            The wavelength .

        Returns:
        2D array-like
            The electric field distribution scaled by E.
        """

        E = bd.sqrt(E2)##intensity to amplitude
        # 计算在每个周期内的局部坐标，使像素居中于每个周期
        # x_local, y_local 的范围为 (-period/2, period/2]

        x_local = bd.mod(xx + self.x_period / 2.0, self.x_period) - self.x_period / 2.0
        y_local = bd.mod(yy + self.y_period / 2.0, self.y_period) - self.y_period / 2.0

        if self.Moudle == 13.2:
            # 在像素矩形内部为 True (1)，否则为 False (0)
            if self.spectrum == 'R':
                mask = (x_local >= 9.655*um) & (x_local <= 41.765*um) & \
                       (y_local >= 10.515*um) & (y_local <= 37.095*um)
                print("Red subpixels selected, Luminance(nit):",E2*(41.765-9.655)*(37.095-10.515)*um**2/(self.x_period*self.y_period))
            elif self.spectrum == 'G':
                mask = (x_local >= -36.095*um) & (x_local <= -9.345*um) & \
                        (y_local >= 10.515*um) & (y_local <= 37.095*um)
                print("Green subpixels selected, Luminance(nit):",E2*( -9.345+36.095)*(37.095-10.515)*um**2/(self.x_period*self.y_period))
            elif self.spectrum == 'B':
                mask = (x_local >= -27.155*um) & (x_local <= 25.715*um) & \
                       (y_local >= -40.775*um) & (y_local <= -8.485*um)
                print("Blue subpixels selected, Luminance(nit):",E2*(25.715+27.155)*(-8.485+40.775)*um**2/(self.x_period*self.y_period))
            else:
                raise ValueError("请输入RGB光谱选项: 'R', 'G' 或 'B'")
        elif self.Moudle == 8.8:
            if self.spectrum == 'G':
                mask = (x_local >= -17.81/2*um) & (x_local <= 17.81/2*um) & \
                          (y_local >= -18.96/2*um) & (y_local <= 18.96/2*um)
                print("Green subpixels selected")


        elif self.Moudle == 13.8:
            if self.spectrum == 'G':
                # 1. 定义你图中不规则区域的多边形顶点（单位和你的um一致即可）
                # 这里是示例，你需要替换成你图中区域的实际顶点坐标
                poly_vertices = bd.array([
                    [16.3*um, 42.89*um],  # 左上角点
                    [9.1*um, 0],  # 右上角点
                    [16.3*um, -42.89*um],  # 右下角点
                    [-16.3*um, -42.89*um],  # 左下角点
                    [-23.5 * um, 0],
                    [-16.3 * um, 42.89 * um],

                    # 可以加更多点，只要按顺序构成闭合多边形
                ])

                # 2. 创建路径对象
                path = Path(poly_vertices)

                # 3. 将网格点展平为(N,2)的坐标数组
                points = bd.column_stack((x_local.ravel(), y_local.ravel()))

                # 4. 判断点是否在多边形内，生成mask
                mask = path.contains_points(points).reshape(x_local.shape)

                # 高斯修饰一下边缘， 更真实的子像素亮度分布
                # =========================
                # 5. 高斯分布（更真实的子像素亮度）
                # =========================


                sigma_x = (32.6*um) / 1
                sigma_y = (85.8*um) / 1.5

                E = bd.exp(
                    -((x_local ) ** 2 / (1 * sigma_x ** 2) +
                      (y_local ) ** 2 / (1 * sigma_y ** 2))
                ) * bd.exp(1j * 2 * bd.pi * bd.sqrt(
                    (x_local ) ** 2 + (y_local ) ** 2 + (15 * um / bd.tan(div * bd.pi / 180)) ** 2) / λ)

                # =========================
                # 6. 最终电场
                # =========================





        # mask = (bd.abs(x_local) <= (self.width / 2.0)) & (bd.abs(y_local) <= (self.height / 2.0))

        # 使用后端的 where 生成数值场（1 和 0），再乘以振幅 E
        field = bd.where(mask, 1.0, 0.0)

        return field * E


    def get_E(self, E2, xx, yy, λ,div=1, effective_area=1):
        """
        Returns the electric field distribution defined by the pixel array.
        """
        effective_area_ = effective_area*bd.max(bd.abs(xx))*2
        # =========================
        # 1. 强度 → 振幅
        # =========================
        E = bd.sqrt(E2)

        # =========================
        # 2. 映射到周期单元
        # =========================
        x_local = bd.mod(xx + self.x_period / 2.0, self.x_period) - self.x_period / 2.0
        y_local = bd.mod(yy + self.y_period / 2.0, self.y_period) - self.y_period / 2.0

        # =========================
        # 3. 定义子像素区域
        # =========================
        if self.Moudle == 13.2:
            if self.spectrum == 'R':
                x_min, x_max = 9.655 * um, 41.765 * um
                y_min, y_max = 10.515 * um, 37.095 * um

                print("Red subpixels selected, Luminance(nit):",
                      E2 * (x_max - x_min) * (y_max - y_min) / (self.x_period * self.y_period))

            elif self.spectrum == 'G':
                x_min, x_max = -36.095 * um, -9.345 * um
                y_min, y_max = 10.515 * um, 37.095 * um

                print("Green subpixels selected, Luminance(nit):",
                      E2 * (x_max - x_min) * (y_max - y_min) / (self.x_period * self.y_period))

            elif self.spectrum == 'B':
                x_min, x_max = -27.155 * um, 25.715 * um
                y_min, y_max = -40.775 * um, -8.485 * um

                print("Blue subpixels selected, Luminance(nit):",
                      E2 * (x_max - x_min) * (y_max - y_min) / (self.x_period * self.y_period))

            else:
                raise ValueError("请输入RGB光谱选项: 'R', 'G' 或 'B'")

        if self.Moudle == 8.8:
            if self.spectrum == 'G':
                x_min, x_max = -17.81/2 * um, 17.81/2 * um
                y_min, y_max = -18.96/2 * um, 18.96/2 * um

                print("Green subpixels selected, Luminance(nit)")



        # =========================
        # 4. 硬mask（几何范围）
        # =========================
        mask = (x_local >= x_min) & (x_local <= x_max) & \
               (y_local >= y_min) & (y_local <= y_max)

        # =========================
        # 5. 高斯分布（更真实的子像素亮度）
        # =========================
        xc = (x_min + x_max) / 2.0
        yc = (y_min + y_max) / 2.0

        sigma_x = (x_max - x_min) / 1.7
        sigma_y = (y_max - y_min) / 1.7

        gaussian = bd.exp(
            -((x_local - xc) ** 2 / (1 * sigma_x ** 2) +
              (y_local - yc) ** 2 / (1 * sigma_y ** 2))
        ) * bd.exp(1j*2*bd.pi*bd.sqrt((x_local-xc)**2+(y_local-yc)**2+(15*um/bd.tan(div*bd.pi/180))**2)/λ)

        # =========================
        # 6. 最终电场
        # =========================

        mask1 = (xx >= -effective_area_/2) & (xx <= effective_area_/2) & (yy >= -effective_area_/2) & (yy <= effective_area_/2)
        field = bd.where(mask1, E * gaussian, 0.0)



        return field