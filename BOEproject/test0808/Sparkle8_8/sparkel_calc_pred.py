from PIL import Image
from matplotlib import pyplot as plt
import diffractsim
diffractsim.set_backend("CPU") #Change the string to "CUDA" to use GPU acceleration
from diffractsim.util.backend_functions import backend as bd
from diffractsim import  cm, mm, um, nm, cf
# 测试 sparkle_sim 类
from diffractsim.sparkle_simulator import SparkleSimulator
module = 8.8 ###模组名称
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
# image_path = r"F:\PHD\GitFiles\diffractsim\BOEproject\test0629\imagefiles\image1.png"  # 你的图片路径
# sag = SparkleSimulator.jetimg2sag(image_path = image_path, v_min=-0.561*um, v_max=1.202*um)
# image_path = r"F:\PHD\GitFiles\diffractsim\BOEproject\test0629\imagefiles\image2.png"  # 你的图片路径
# sag = SparkleSimulator.jetimg2sag(image_path = image_path, v_min=-0.811*um, v_max=1.914*um)
# image_path = r"F:\PHD\GitFiles\diffractsim\BOEproject\test0629\imagefiles\image3.png"  # 你的图片路径
# sag = SparkleSimulator.jetimg2sag(image_path = image_path, v_min=-0.641*um, v_max=1.103*um)
# image_path = r"F:\PHD\GitFiles\diffractsim\BOEproject\test0629\imagefiles\image4.png"  # 你的图片路径
# sag = SparkleSimulator.jetimg2sag(image_path = image_path, v_min=-0.592*um, v_max=0.925*um)
# image_path = r"F:\PHD\GitFiles\diffractsim\BOEproject\test0629\imagefiles\image5.png"  # 你的图片路径
# sag = SparkleSimulator.jetimg2sag(image_path = image_path, v_min=-0.574*um, v_max=0.881*um)
image_path = r"F:\PHD\GitFiles\diffractsim\BOEproject\test0629\imagefiles\image6.png"  # 你的图片路径
sag = SparkleSimulator.jetimg2sag(image_path = image_path, v_min=-0.634*um, v_max=1.036*um)

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

Isparkle = sim.get_Sparkle_origin(spectrum='G', layer_Structure=layerstructure, AGetch_params=AGetchparams,intensity=5, Module = module )
sim.plot_intensity(Isparkle, square_root= False, units = um, text = 'Sparkle Pattern at RGB layer(unfilter)',colormap = 'gray')

# # #####################Visual photo###################
# Ivisual = sim.VisualPhoto(sigma= 62*um)
# sim.plot_intensity(Ivisual, square_root= False, units = um, text = 'Visual Photo of Sparkle Pattern',colormap = 'gray')
#
# #
# # #################### SIM method ######################
sparkle_sim = sim.SIM(method = "A")#  A:对应黄光AG B:对应普通蒙砂刻蚀AG
##########调取sim_calibration.jion中的拟合参数，进行闪点预测##########
import json

# 读取拟合参数
with open(
        "sim_calibration.json",
        "r",
        encoding="utf-8"
) as f:

    fit = json.load(f)

a = fit["a"]
b = fit["b"]

# 修正后的预测实验值
sparkle_pred = a * sparkle_sim + b

print(f"仿真SIM值: {sparkle_sim:.3f}")
print(f"预测实验SIM值: {sparkle_pred:.3f}")
