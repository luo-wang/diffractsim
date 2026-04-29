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



import numpy as np
import matplotlib.pyplot as plt

# ======================
# 参数设置
# ======================
wavelength = 532e-9       # 绿光 532 nm
k = 2*np.pi / wavelength

dx = 1e-6                 # 采样间隔（2 um）
N = 1024                  # 采样点数
L = N * dx

z = 1e-3                  # 传播距离 1 mm

# 光源参数
pixel_size = 25e-6        # 像素尺寸
period = 100e-6           # 周期

# ======================
# 坐标
# ======================
x = np.linspace(-L/2, L/2, N)
y = np.linspace(-L/2, L/2, N)
X, Y = np.meshgrid(x, y)

# ======================
# 构建矩阵光源（周期阵列）
# ======================
def rect(x, w):
    return np.abs(x) <= w/2

U0 = np.zeros((N, N))

# 构建周期阵列
for i in np.arange(-L/2, L/2, period):
    for j in np.arange(-L/2, L/2, period):
        U0 += rect(X - i, pixel_size) * rect(Y - j, pixel_size)

# 归一化
U0 = U0.astype(np.complex128)

# ======================




from scipy.interpolate import RegularGridInterpolator
def sag_interpolation(Z, Lx, Ly, X, Y):
    """
    将 sag 数据插值到任意坐标

    参数：
    Z  : 原始 sag (Nx, Ny)
    Lx : x方向物理长度
    Ly : y方向物理长度
    X, Y : 目标坐标网格

    返回：
    插值后的 sag
    """

    Nx, Ny = Z.shape

    x0 = np.linspace(-Lx/2, Lx/2, Nx)
    y0 = np.linspace(-Ly/2, Ly/2, Ny)

    interp_func = RegularGridInterpolator(
        (x0, y0), Z,
        bounds_error=False,
        fill_value=0
    )

    pts = np.stack([X.ravel(), Y.ravel()], axis=-1)

    return interp_func(pts).reshape(X.shape)
# 角谱传播函数
# ======================
def angular_spectrum(U, z):
    fx = np.fft.fftfreq(N, d=dx)
    fy = np.fft.fftfreq(N, d=dx)
    FX, FY = np.meshgrid(fx, fy)

    # 传递函数
    H = np.exp(1j * k * z * np.sqrt(
        1 - (wavelength*FX)**2 - (wavelength*FY)**2
    ))

    U_fft = np.fft.fft2(U)
    U_out = np.fft.ifft2(U_fft * H)

    return U_out

# ======================
# Step 1: 前向传播 1 mm
# ======================
U1 = angular_spectrum(U0, z)

# ======================
# Step 2: 相位调制面
# （这里给你一个示例：球面相位）
# ======================
# R = 2e-3  # 曲率半径 2 mm
# phase = k * (X**2 + Y**2) / (2 * R)
# phase = 0
Z_interp = sag_interpolation(sag, 6*277e-6, 6*207e-6, X, Y)


phase = (1.5-1)*Z_interp*2*np.pi/wavelength*0.1




U2 = U1 * np.exp(1j * phase)

# ======================
# Step 3: 反向传播 -1 mm
# ======================
U3 = angular_spectrum(U2, -z)

# ======================

plt.figure(figsize=(12,4))

plt.title("Sag")
plt.imshow(phase, cmap='gray')
plt.colorbar()
plt.show()

# 可视化
# ======================
plt.figure(figsize=(12,4))

plt.subplot(131)
plt.title("Source Intensity")
plt.imshow(np.abs(U0)**2, cmap='gray')
plt.colorbar()

plt.subplot(132)
plt.title("After Phase (1mm)")
plt.imshow(np.abs(U2)**2, cmap='gray')
plt.colorbar()

plt.subplot(133)
plt.title("Back Propagated")
plt.imshow(np.abs(U3)**2, cmap='gray')
plt.colorbar()

plt.tight_layout()
plt.show()





from scipy.interpolate import RegularGridInterpolator
def sag_interpolation(Z, Lx, Ly, X, Y):
    """
    将 sag 数据插值到任意坐标

    参数：
    Z  : 原始 sag (Nx, Ny)
    Lx : x方向物理长度
    Ly : y方向物理长度
    X, Y : 目标坐标网格

    返回：
    插值后的 sag
    """

    Nx, Ny = Z.shape

    x0 = np.linspace(-Lx/2, Lx/2, Nx)
    y0 = np.linspace(-Ly/2, Ly/2, Ny)

    interp_func = RegularGridInterpolator(
        (x0, y0), Z,
        bounds_error=False,
        fill_value=0
    )

    pts = np.stack([X.ravel(), Y.ravel()], axis=-1)

    return interp_func(pts).reshape(X.shape)