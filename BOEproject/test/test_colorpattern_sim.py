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
image_path = r"F:\PHD\GitFiles\diffractsim\BOEproject\test\rectangular_grating.jpg"
sag = SparkleSimulator.jetimg2sag(image_path = image_path, v_min=-2.204*um, v_max=3.043*um)
sag = sag*1
plt.figure();plt.imshow(sag, cmap='jet');plt.colorbar();plt.title('Sag map from jet image');plt.show()

# 定义刻蚀层参数
AGetchparams = {
    'width': 705.282*um,
    'height':528.789*um,
    'sag': sag,
    'xyorigin': [0.0*um, 0.0*um]
}


##########################colorpattern simulate##############


colorsim = ColorpatternSimulator(spectrum=100 * cf.illuminant_d65,
    extent_x=20.0 * mm,
    extent_y=20.0 * mm,
    Nx=256*2,
    Ny=256*2,
    )
colorsim.DoSimulate(50*cm,AGetchparams)