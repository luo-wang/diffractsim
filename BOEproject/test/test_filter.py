import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter

# =====================================================
# 1. 参数设置
# =====================================================
N = 512                    # 光场尺寸
pixel_pitch = 16           # LED 像素周期
led_size = 8               # 单个 LED 尺寸

# 低频噪声参数
lowfreq_noise_strength = 0.6
lowfreq_sigma = 20        # 越大，噪声越低频

# 方形卷积核参数（关键）
kernel_size = pixel_pitch * 1  # 必须显著大于 pixel_pitch

np.random.seed(0)

# =====================================================
# 2. 构造 LED 像素阵列光场
# =====================================================
pixel_field = np.zeros((N, N))
for i in range(0, N, pixel_pitch):
    for j in range(0, N, pixel_pitch):
        pixel_field[i:i+led_size, j:j+led_size] = 1.0

# =====================================================
# 3. 构造随机低频噪声
# =====================================================
white_noise = np.random.randn(N, N)
lowfreq_noise = gaussian_filter(white_noise, sigma=lowfreq_sigma)
lowfreq_noise /= np.std(lowfreq_noise)

# =====================================================
# 4. 初始光场
# =====================================================
field = pixel_field + lowfreq_noise_strength * lowfreq_noise

plt.imshow(lowfreq_noise, cmap='gray')
# =====================================================
# 5. 方形卷积核（积分核）
# =====================================================
kernel = np.ones((kernel_size, kernel_size))
kernel /= kernel.sum()     # 能量归一化

# =====================================================
# 6. 卷积实现均匀化
# =====================================================
uniform_field = convolve2d(
    field,
    kernel,
    mode="same",
    boundary="symm"
)

# =====================================================
# 7. 整体亮度归一化
# =====================================================
uniform_field /= np.mean(uniform_field)

# =====================================================
# 8. 可视化
# =====================================================
plt.figure(figsize=(16, 4))

plt.subplot(1, 4, 1)
plt.imshow(pixel_field, cmap="gray")
plt.title("LED Pixel Array")

plt.subplot(1, 4, 2)
plt.imshow(field, cmap="gray")
plt.title("Pixel Array + Low-frequency Noise")

plt.subplot(1, 4, 3)
plt.imshow(uniform_field, cmap="gray")
plt.title("Uniformized Field (Square Kernel)")

plt.subplot(1, 4, 4)
plt.plot(uniform_field[N // 2])
plt.title("1D Cross-section")

plt.tight_layout()
plt.show()
