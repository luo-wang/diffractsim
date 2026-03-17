from PIL import Image
from matplotlib import pyplot as plt
import diffractsim
diffractsim.set_backend("CPU") #Change the string to "CUDA" to use GPU acceleration
from diffractsim.util.backend_functions import backend as bd
from diffractsim import  cm, mm, um, nm, cf
# 测试 sparkle_sim 类
from diffractsim.sparkle_simulator import SparkleSimulator
from diffractsim.colorpattern_simulator import ColorpatternSimulator


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
image_path = r"F:\PHD\GitFiles\diffractsim\BOEproject\test\etchphoto\lower1_3.png"  # 你的图片路径
sag = SparkleSimulator.jetimg2sag(image_path = image_path, v_min=-2.204*um, v_max=3.043*um)
sag = sag*1
plt.figure();plt.imshow(sag, cmap='jet');plt.colorbar();plt.title('Sag map from jet image');plt.show()

# 定义刻蚀层参数
AGetchparams = {
    'width': 705.282*um*3,
    'height':528.789*um*3,
    'sag': sag,
    'xyorigin': [0.0*um, 0.0*um]
}
# 创建 SparkleSimulator 实例， 得到闪点图的仿真结果
sim = SparkleSimulator(wavelength=500 * nm, extent_x=1500*um, extent_y=1500*um,Nx=1000, Ny=1000 )
Isparkle = sim.get_Sparkle_origin(spectrum='G', layer_Structure=layerstructure, AGetch_params=AGetchparams,intensity=10000 )
sim.plot_intensity(Isparkle, square_root= False, units = um, text = 'Sparkle Pattern at RGB layer(unfilter)',colormap = 'gray')

#####################Visual photo###################
Ivisual = sim.VisualPhoto(sigma= 62*um)
sim.plot_intensity(Ivisual, square_root= False, units = um, text = 'Visual Photo of Sparkle Pattern',colormap = 'gray')


#################### SIM method ######################
sim.SIM()

# #################### DIM test ########################
AGetchparams['xyorigin'] = [100*um, 10*um]
# 创建 SparkleSimulator 实例， 得到闪点图的仿真结果
sim = SparkleSimulator(wavelength=500 * nm, extent_x=1500*um, extent_y=1500*um,Nx=1000, Ny=1000 )
Isparkle1 = sim.get_Sparkle_origin(spectrum='G', layer_Structure=layerstructure, AGetch_params=AGetchparams,intensity=10000 )
sim.plot_intensity(Isparkle1, square_root= False, units = um, text = 'Second Sparkle Pattern at RGB layer(unfilter)',colormap = 'gray')
sim.DIM(Isparkle)
