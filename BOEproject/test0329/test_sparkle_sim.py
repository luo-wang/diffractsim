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
    'n_OLED': 1.8,
    'z_OLED': 200 * um,
    'n_OCA': 1.5,
    'z_OCA': 250 * um,
    'n_AG': 1.5,
    'z_AG': 500 * um
}
# 读取刻蚀图像并转换为示例 sag 数据
#### 0闪点
# image_path = r"F:\PHD\GitFiles\diffractsim\BOEproject\test0329\etchphoto\image0_1.png"  # 你的图片路径
# sag = SparkleSimulator.jetimg2sag(image_path = image_path, v_min=-0.611*um, v_max=1.27*um)
# image_path = r"F:\PHD\GitFiles\diffractsim\BOEproject\test0329\etchphoto\image0_2.png"  # 你的图片路径
# sag = SparkleSimulator.jetimg2sag(image_path = image_path, v_min=-0.553*um, v_max=1.251*um)
# image_path = r"F:\PHD\GitFiles\diffractsim\BOEproject\test0329\etchphoto\image0_3.png"  # 你的图片路径
# sag = SparkleSimulator.jetimg2sag(image_path = image_path, v_min=-0.525*um, v_max=1.089*um)


#### 1.8闪点
# image_path = r"F:\PHD\GitFiles\diffractsim\BOEproject\test0329\etchphoto\image1.81_1.png"  # 你的图片路径
# sag = SparkleSimulator.jetimg2sag(image_path = image_path, v_min=-0.577*um, v_max=1.29*um)
# image_path = r"F:\PHD\GitFiles\diffractsim\BOEproject\test0329\etchphoto\image1.81_2.png"  # 你的图片路径
# sag = SparkleSimulator.jetimg2sag(image_path = image_path, v_min=-0.567*um, v_max=1.154*um)
# image_path = r"F:\PHD\GitFiles\diffractsim\BOEproject\test0329\etchphoto\image1.81_3.png"  # 你的图片路径
# sag = SparkleSimulator.jetimg2sag(image_path = image_path, v_min=-0.586*um, v_max=1.169*um)

# ### 2.5闪点
# image_path = r"F:\PHD\GitFiles\diffractsim\BOEproject\test0329\etchphoto\image2.5_1.png"  # 你的图片路径
# sag = SparkleSimulator.jetimg2sag(image_path = image_path, v_min=-0.685*um, v_max=1.329*um)
# image_path = r"F:\PHD\GitFiles\diffractsim\BOEproject\test0329\etchphoto\image2.5_2.png"  # 你的图片路径
# sag = SparkleSimulator.jetimg2sag(image_path = image_path, v_min=-0.707*um, v_max=1.387*um)
image_path = r"F:\PHD\GitFiles\diffractsim\BOEproject\test0329\etchphoto\image2.5_3.png"  # 你的图片路径
sag = SparkleSimulator.jetimg2sag(image_path = image_path, v_min=-0.696*um, v_max=1.401*um)

sag = SparkleSimulator.mosaic_array(sag,6)
plt.figure();plt.imshow(sag, cmap='jet');plt.colorbar();plt.title('Sag map from jet image');plt.show()

# 定义刻蚀层参数
AGetchparams = {
    'width': 277.*um*6,
    'height':207.*um*6,
    'sag': sag*1,
    'xyorigin': [0.0*um, 0.0*um]
}
# 创建 SparkleSimulator 实例， 得到闪点图的仿真结果
sim = SparkleSimulator(wavelength=500 * nm, extent_x=1000*um, extent_y=1000*um,Nx=1024, Ny=1024 )
Isparkle = sim.get_Sparkle_origin(spectrum='G', layer_Structure=layerstructure, AGetch_params=AGetchparams,intensity=10000 )
sim.plot_intensity(Isparkle, square_root= False, units = um, text = 'Sparkle Pattern at RGB layer(unfilter)',colormap = 'gray')

# #####################Visual photo###################
Ivisual = sim.VisualPhoto(sigma= 62*um)
sim.plot_intensity(Ivisual, square_root= False, units = um, text = 'Visual Photo of Sparkle Pattern',colormap = 'gray')
#

#################### SIM method ######################
sim.SIM()
#
# #################### DIM test ########################
AGetchparams['xyorigin'] = [100*um, 100*um]
# 创建 SparkleSimulator 实例， 得到闪点图的仿真结果
sim = SparkleSimulator(wavelength=500 * nm, extent_x=1000*um, extent_y=1000*um,Nx=1024, Ny=1024 )
Isparkle1 = sim.get_Sparkle_origin(spectrum='G', layer_Structure=layerstructure, AGetch_params=AGetchparams,intensity=10000 )
sim.plot_intensity(Isparkle1, square_root= False, units = um, text = 'Second Sparkle Pattern at RGB layer(unfilter)',colormap = 'gray')
sim.DIM(Isparkle)
