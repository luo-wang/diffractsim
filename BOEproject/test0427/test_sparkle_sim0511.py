from PIL import Image
from matplotlib import pyplot as plt
import diffractsim
diffractsim.set_backend("CPU") #Change the string to "CUDA" to use GPU acceleration
from diffractsim.util.backend_functions import backend as bd
from diffractsim import  cm, mm, um, nm, cf
# 测试 sparkle_sim 类
from diffractsim.sparkle_simulator import SparkleSimulator


# 定义膜层结构参数
layerstructure = {
    'n_OLED': 1.6,
    'z_OLED': 150 * um,
    'n_OCA': 1.5,
    'z_OCA': 250 * um,
    'n_AG': 1.5,
    'z_AG': 300 * um
}


# 读取刻蚀图像并转换为示例 sag 数据
####闪点
# image_path = r"F:\PHD\GitFiles\diffractsim\BOEproject\sparkle0511\image1.png"  # 你的图片路径
# sag = SparkleSimulator.jetimg2sag(image_path = image_path, v_min=-0.339*um, v_max=0.78*um)
# image_path = r"F:\PHD\GitFiles\diffractsim\BOEproject\sparkle0511\image2.png"  # 你的图片路径
# sag = SparkleSimulator.jetimg2sag(image_path = image_path, v_min=-0.503*um, v_max=1.095*um)
image_path = r"F:\PHD\GitFiles\diffractsim\BOEproject\sparkle0511\image3.png"  # 你的图片路径
sag = SparkleSimulator.jetimg2sag(image_path = image_path, v_min=-0.635*um, v_max=1.301*um)
# image_path = r"F:\PHD\GitFiles\diffractsim\BOEproject\sparkle0511\image4.png"  # 你的图片路径
# sag = SparkleSimulator.jetimg2sag(image_path = image_path, v_min=-0.925*um, v_max=0.705*um)
# image_path = r"F:\PHD\GitFiles\diffractsim\BOEproject\sparkle0511\image5.png"  # 你的图片路径
# sag = SparkleSimulator.jetimg2sag(image_path = image_path, v_min=-0.731*um, v_max=0.69*um)




sag = SparkleSimulator.mosaic_array(sag,5)
plt.figure();plt.imshow(sag, cmap='jet');plt.colorbar();plt.title('Sag map from jet image');plt.show()

# 定义刻蚀层参数
AGetchparams = {
    'width': 277.*um*5,
    'height':207.*um*5,
    'sag': sag*1,
    'xyorigin': [0.0*um, 0.0*um]
}
# 创建 SparkleSimulator 实例， 得到闪点图的仿真结果

sim = SparkleSimulator(wavelength=500 * nm, extent_x=1000*um, extent_y=1000*um,Nx=512*1, Ny=512*1 )######8.8


Isparkle = sim.get_Sparkle_origin(spectrum='G', layer_Structure=layerstructure, AGetch_params=AGetchparams,intensity=5, div = 0.001, Module = 8.8 )
sim.plot_intensity(Isparkle, square_root= False, units = um, text = 'Sparkle Pattern at RGB layer(unfilter)',colormap = 'gray')

# # #####################Visual photo###################
# Ivisual = sim.VisualPhoto(sigma= 62*um)
# sim.plot_intensity(Ivisual, square_root= False, units = um, text = 'Visual Photo of Sparkle Pattern',colormap = 'gray')
#
# #
# # #################### SIM method ######################
sim.SIM()
# ##
# # #################### DIM test ########################
# AGetchparams['xyorigin'] = [100*um, 10*um]
# # 创建 SparkleSimulator 实例， 得到闪点图的仿真结果
# sim = SparkleSimulator(wavelength=500 * nm, extent_x=1000*um, extent_y=1000*um,Nx=1024, Ny=1024 )
# Isparkle1 = sim.get_Sparkle_origin(spectrum='G', layer_Structure=layerstructure, AGetch_params=AGetchparams,intensity=10000, div = 1.2, effective_area = 1 )
# sim.plot_intensity(Isparkle1, square_root= False, units = um, text = 'Second Sparkle Pattern at RGB layer(unfilter)',colormap = 'gray')
# sim.DIM(Isparkle)
