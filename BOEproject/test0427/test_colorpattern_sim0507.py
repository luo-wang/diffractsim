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

# image_path = r"F:\PHD\GitFiles\diffractsim\BOEproject\colorpattern0507\image1.png"  # 你的图片路径
# sag = ColorpatternSimulator.jetimg2sag(image_path = image_path, v_min=-0.561*um, v_max=1.202*um)
# image_path = r"F:\PHD\GitFiles\diffractsim\BOEproject\colorpattern0507\image2.png"  # 你的图片路径
# sag = ColorpatternSimulator.jetimg2sag(image_path = image_path, v_min=-0.845*um, v_max=1.535*um)
# image_path = r"F:\PHD\GitFiles\diffractsim\BOEproject\colorpattern0507\image3.png"  # 你的图片路径
# sag = ColorpatternSimulator.jetimg2sag(image_path = image_path, v_min=-0.698*um, v_max=1.243*um)
# image_path = r"F:\PHD\GitFiles\diffractsim\BOEproject\colorpattern0507\image4.png"  # 你的图片路径
# sag = ColorpatternSimulator.jetimg2sag(image_path = image_path, v_min=-0.723*um, v_max=1.368*um)
# image_path = r"F:\PHD\GitFiles\diffractsim\BOEproject\colorpattern0507\image5.png"  # 你的图片路径
# sag = ColorpatternSimulator.jetimg2sag(image_path = image_path, v_min= 0.119*um, v_max=2.61*um)
# image_path = r"F:\PHD\GitFiles\diffractsim\BOEproject\colorpattern0507\image6.png"  # 你的图片路径
# sag = ColorpatternSimulator.jetimg2sag(image_path = image_path, v_min=-0.974*um, v_max=1.39*um)
# image_path = r"F:\PHD\GitFiles\diffractsim\BOEproject\colorpattern0507\image7.png"  # 你的图片路径
# sag = ColorpatternSimulator.jetimg2sag(image_path = image_path, v_min=-0.531*um, v_max=0.986*um)
# image_path = r"F:\PHD\GitFiles\diffractsim\BOEproject\colorpattern0507\image8.png"  # 你的图片路径
# sag = ColorpatternSimulator.jetimg2sag(image_path = image_path, v_min=-0.506*um, v_max=1.057*um)
# image_path = r"F:\PHD\GitFiles\diffractsim\BOEproject\colorpattern0507\image9.png"  # 你的图片路径
# sag = ColorpatternSimulator.jetimg2sag(image_path = image_path, v_min=-0.858*um, v_max=1.183*um)
# image_path = r"F:\PHD\GitFiles\diffractsim\BOEproject\colorpattern0507\image10.png"  # 你的图片路径
# sag = ColorpatternSimulator.jetimg2sag(image_path = image_path, v_min=-0.811*um, v_max=1.155*um)
# image_path = r"F:\PHD\GitFiles\diffractsim\BOEproject\colorpattern0507\image11.png"  # 你的图片路径
# sag = ColorpatternSimulator.jetimg2sag(image_path = image_path, v_min=-0.414*um, v_max=0.823*um)
# image_path = r"F:\PHD\GitFiles\diffractsim\BOEproject\colorpattern0507\image12.png"  # 你的图片路径
# sag = ColorpatternSimulator.jetimg2sag(image_path = image_path, v_min=-0.357*um, v_max=0.772*um)
# image_path = r"F:\PHD\GitFiles\diffractsim\BOEproject\colorpattern0507\image13.png"  # 你的图片路径
# sag = ColorpatternSimulator.jetimg2sag(image_path = image_path, v_min=-0.886*um, v_max=1.17*um)
image_path = r"F:\PHD\GitFiles\diffractsim\BOEproject\colorpattern0507\image14.png"  # 你的图片路径
sag = ColorpatternSimulator.jetimg2sag(image_path = image_path, v_min=-0.708*um, v_max=1.003*um)





# sag = SparkleSimulator.mosaic_array(sag,1)
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
colorsim = ColorpatternSimulator(spectrum= cf.illuminant_d65,
    extent_x=0.3 * mm,
    extent_y=0.3 * mm,
    Nx=256*1,
    Ny=256*1
    )
colorsim.DoSimulate(30*cm,AGetchparams)

