import diffractsim
diffractsim.set_backend("CUDA") #Change the string to "CUDA" to use GPU acceleration
from diffractsim.util.backend_functions import backend as bd
from diffractsim import  mm, um, nm
# 测试 RGBarray 类
from diffractsim.light_sources.RGB_array import RGBarray
# import numpy as np
import matplotlib.pyplot as plt

# 创建坐标网格
x = bd.linspace(-500*um, 500*um, 500)
y = bd.linspace(-500*um, 500*um, 500)
xx, yy = bd.meshgrid(x, y)
# 创建 RGBArray 光源
R_array = RGBarray(spectrum='R')
# 获取电场分布
E_R = R_array.get_E(1.0,xx,yy,λ=500*nm)

G_array = RGBarray(spectrum='G')
E_G = G_array.get_E(1.0,xx,yy,λ=500*nm)
B_array = RGBarray(spectrum='B')
E_B = B_array.get_E(1.0,xx,yy,λ=500*nm)
# 合成 RGB 光场
E = bd.sqrt(E_R**2 + E_G**2 + E_B**2)
# 可视化结果
plt.imshow(E_G.get(), extent=(-500, 500, -500, 500), origin='lower', cmap='gray')
plt.colorbar(label='Electric Field Amplitude')
plt.title('Pixel Array Light Source')
plt.xlabel('x')
plt.ylabel('y')
plt.show()