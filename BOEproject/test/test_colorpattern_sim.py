from PIL import Image
from matplotlib import pyplot as plt
import diffractsim
diffractsim.set_backend("CPU") #Change the string to "CUDA" to use GPU acceleration
from diffractsim.util.backend_functions import backend as bd
from diffractsim import  cm, mm, um, nm, cf
# 测试 sparkle_sim 类
from diffractsim.sparkle_simulator import SparkleSimulator
from diffractsim.colorpattern_simulator import ColorpatternSimulator


# 读取刻蚀图像并转换为示例 sag 数据
# image_path = r"F:\PHD\GitFiles\diffractsim\BOEproject\test\etchphoto\upper2.png"  # 你的图片路径
image_path = r"F:\PHD\GitFiles\diffractsim\BOEproject\test\etchphoto\image4.png"  # 你的图片路径

# image_path = r"F:\PHD\GitFiles\diffractsim\BOEproject\test\rectangular_grating.jpg"
sag = SparkleSimulator.jetimg2sag(image_path = image_path, v_min=-2.204*um, v_max=3.043*um)
sag = sag*1
plt.figure();plt.imshow(sag, cmap='jet');plt.colorbar();plt.title('Sag map from jet image');plt.show()

# 定义刻蚀层参数
AGetchparams = {
    'width': 276*um*1,
    'height':207*um*1,
    'sag': sag,
    'xyorigin': [0.0*um, 0.0*um]
}


##########################colorpattern simulate##############


colorsim = ColorpatternSimulator(spectrum=1* cf.illuminant_d65,
    extent_x=1.0 * mm,
    extent_y=1.0 * mm,
    Nx=256*8,
    Ny=256*8,
    )
colorsim.DoSimulate(10*cm,AGetchparams)


# #############弗朗禾费衍射计算########
#
# from diffractsim import MonochromaticField, AG_Reflect
# sim11 = MonochromaticField(wavelength=500 * nm, extent_x=1*mm, extent_y=1*mm, Nx=256*8, Ny=256*8,intensity=1)
# sim11.add(AG_Reflect(width=AGetchparams['width'], height=AGetchparams['height'], sag=AGetchparams['sag'], n=1))
#
# sim11.plot_intensity(abs(sim11.E)**2, square_root= False, units = um, text = 'Fraunhofer Diffraction Pattern',colormap = 'gray')
# sim11.plot_phase(sim11.E, units = um, text = 'Fraunhofer Diffraction Pattern Phase')
#
#
# # sim11.propagate(100*cm)
# # sim11.plot_intensity(abs(sim11.E)**2, square_root= False, units = um, text = 'Fraunhofer Diffraction Pattern',colormap = 'gray')
# # # sim11.plot_phase(sim11.E, units = um, text = 'Fraunhofer Diffraction Pattern Phase')
# #
# # k = 2*bd.pi/(500*nm)
# # phase = k*AGetchparams['sag']
# # field = bd.exp(1j*phase)
# field = sim11.E
# # 对field做傅里叶变换
# field_fft = bd.fft.fftshift(bd.fft.fft2(field))
#
# # 绘制频谱的幅度
# plt.figure()
# plt.imshow((bd.abs(field_fft)**2)**0.4, cmap='hot')
# plt.colorbar()
# plt.title('Fourier Transform Magnitude')
# plt.show()
#
# # # 绘制频谱的相位
# # plt.figure()
# # plt.imshow(bd.angle(field_fft), cmap='jet')
# # plt.colorbar()
# # plt.title('Fourier Transform Phase')
# # plt.show()
#
