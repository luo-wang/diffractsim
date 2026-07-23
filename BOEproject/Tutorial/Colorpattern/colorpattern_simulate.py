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

# image_path = r"F:\PHD\GitFiles\diffractsim\BOEproject\Tutorial\imagefiles\image1.png"  # 你的图片路径
# sag = SparkleSimulator.jetimg2sag(image_path = image_path, v_min=-0.339*um, v_max=0.78*um)
# image_path = r"F:\PHD\GitFiles\diffractsim\BOEproject\Tutorial\imagefiles\image2.png"  # 你的图片路径
# sag = SparkleSimulator.jetimg2sag(image_path = image_path, v_min=-0.503*um, v_max=1.095*um)
# image_path = r"F:\PHD\GitFiles\diffractsim\BOEproject\Tutorial\imagefiles/image3.png"  # 你的图片路径
# sag = SparkleSimulator.jetimg2sag(image_path = image_path, v_min=-0.635*um, v_max=1.301*um)
# image_path = r"F:\PHD\GitFiles\diffractsim\BOEproject\Tutorial\imagefiles\image4.png"  # 你的图片路径
# sag = SparkleSimulator.jetimg2sag(image_path = image_path, v_min=-0.925*um, v_max=0.705*um)
image_path = r"F:\PHD\GitFiles\diffractsim\BOEproject\Tutorial\imagefiles\image5.png"  # 你的图片路径
sag = SparkleSimulator.jetimg2sag(image_path = image_path, v_min=-0.731*um, v_max=0.69*um)

sag = SparkleSimulator.mosaic_array(sag,1)
plt.figure();plt.imshow(sag, cmap='jet');plt.colorbar();plt.title('Sag map from jet image');plt.show()

# 定义刻蚀层参数
AGetchparams = {
    'width': 277.*um*1,
    'height':207.*um*1,
    'sag': sag*1,
    'xyorigin': [0.0*um, 0.0*um]
}
#
# ##########################colorpattern simulate##############
#
colorsim = ColorpatternSimulator(spectrum= cf.illuminant_d65,
    extent_x=0.2 * mm,
    extent_y=0.2 * mm,
    Nx=128*1,
    Ny=128*1
    )
##白光彩纹模拟结果
colorsim.DoSimulate(30*cm,AGetchparams)
##单波长衍射结果
colorsim.DoSimulate2(30*cm,AGetchparams)