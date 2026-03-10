from ..util.backend_functions import backend as bd
from .light_source import LightSource
from diffractsim import  mm, um
class RGBarray(LightSource):
    def __init__(self, spectrum='B'):
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
        self.x_period = 96.8*um
        self.y_period = 96.8*um
        self.spectrum = spectrum


    def get_E(self, E2, xx, yy, λ):
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

        # mask = (bd.abs(x_local) <= (self.width / 2.0)) & (bd.abs(y_local) <= (self.height / 2.0))

        # 使用后端的 where 生成数值场（1 和 0），再乘以振幅 E
        field = bd.where(mask, 1.0, 0.0)

        return field * E
