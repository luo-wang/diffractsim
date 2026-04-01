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
#### 0闪点
# image_path = r"F:\PHD\GitFiles\diffractsim\BOEproject\test0329\etchphoto\image0_1.png"  # 你的图片路径
# sag = SparkleSimulator.jetimg2sag(image_path = image_path, v_min=-0.611*um, v_max=1.27*um)
# image_path = r"F:\PHD\GitFiles\diffractsim\BOEproject\test0329\etchphoto\image0_2.png"  # 你的图片路径
# sag = SparkleSimulator.jetimg2sag(image_path = image_path, v_min=-0.553*um, v_max=1.251*um)
# image_path = r"F:\PHD\GitFiles\diffractsim\BOEproject\test0329\etchphoto\image0_3.png"  # 你的图片路径
# sag = SparkleSimulator.jetimg2sag(image_path = image_path, v_min=-0.525*um, v_max=1.089*um)


#### 1.8闪点
image_path = r"F:\PHD\GitFiles\diffractsim\BOEproject\test0329\etchphoto\image1.81_1.png"  # 你的图片路径
sag = SparkleSimulator.jetimg2sag(image_path = image_path, v_min=-0.577*um, v_max=1.29*um)
# image_path = r"F:\PHD\GitFiles\diffractsim\BOEproject\test0329\etchphoto\image1.81_2.png"  # 你的图片路径
# sag = SparkleSimulator.jetimg2sag(image_path = image_path, v_min=-0.567*um, v_max=1.154*um)
# image_path = r"F:\PHD\GitFiles\diffractsim\BOEproject\test0329\etchphoto\image1.81_3.png"  # 你的图片路径
# sag = SparkleSimulator.jetimg2sag(image_path = image_path, v_min=-0.586*um, v_max=1.169*um)

# ### 2.5闪点
# image_path = r"F:\PHD\GitFiles\diffractsim\BOEproject\test0329\etchphoto\image2.5_1.png"  # 你的图片路径
# sag = SparkleSimulator.jetimg2sag(image_path = image_path, v_min=-0.685*um, v_max=1.329*um)
# image_path = r"F:\PHD\GitFiles\diffractsim\BOEproject\test0329\etchphoto\image2.5_2.png"  # 你的图片路径
# sag = SparkleSimulator.jetimg2sag(image_path = image_path, v_min=-0.707*um, v_max=1.387*um)
# image_path = r"F:\PHD\GitFiles\diffractsim\BOEproject\test0329\etchphoto\image2.5_3.png"  # 你的图片路径
# sag = SparkleSimulator.jetimg2sag(image_path = image_path, v_min=-0.696*um, v_max=1.401*um)

sag = SparkleSimulator.mosaic_array(sag,1)
plt.figure();plt.imshow(sag, cmap='jet');plt.colorbar();plt.title('Sag map from jet image');plt.show()

# 定义刻蚀层参数
AGetchparams = {
    'width': 277.*um*1,
    'height':207.*um*1,
    'sag': sag,
    'xyorigin': [0.0*um, 0.0*um]
}
#
# ##########################colorpattern simulate##############
#
colorsim = ColorpatternSimulator(spectrum=1* cf.illuminant_d65,
    extent_x=0.5 * mm,
    extent_y=0.5 * mm,
    Nx=256*8,
    Ny=256*8
    )
colorsim.DoSimulate(10*cm,AGetchparams)

