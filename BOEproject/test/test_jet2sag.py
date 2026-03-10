import numpy as np
from PIL import Image
import matplotlib.cm as cm
import matplotlib.pyplot as plt


# =========================
# 1. 参数设置（你需要改的地方）
# =========================

image_path = "etchphoto/image4.png"   # 你的图片路径
v_min = 0.0                     # 已知最小值
v_max = 1.0                     # 已知最大值
N_COLORS = 256                  # jet 采样数（256 已经够用）


# =========================
# 2. 读取图片
# =========================

img = Image.open(image_path).convert("RGB")
img = np.asarray(img, dtype=np.float32) / 255.0   # (H, W, 3)
H, W, _ = img.shape

print(f"Image loaded: {H} x {W}")


# =========================
# 3. 构建 jet colormap 查找表
# =========================

# jet = cm.get_cmap("jet", N_COLORS)

jet = plt.get_cmap("jet", N_COLORS)
jet_rgb = jet(np.linspace(0.0, 1.0, N_COLORS))[:, :3]  # (N, 3)


# =========================
# 4. RGB → jet 索引 → 归一化值
# =========================

# 展平图片
img_flat = img.reshape(-1, 3)  # (H*W, 3)

# 计算 RGB 距离
# (H*W, N, 3) → (H*W, N)
diff = img_flat[:, None, :] - jet_rgb[None, :, :]
dist = np.linalg.norm(diff, axis=2)

# 找到最接近的 jet 颜色索引
idx = np.argmin(dist, axis=1)

# 归一化到 [0, 1]
value_norm = idx.astype(np.float32) / (N_COLORS - 1)

# 还原二维
value_norm = value_norm.reshape(H, W)


# =========================
# 5. 映射到真实物理范围
# =========================

value = v_min + value_norm * (v_max - v_min)


# =========================
# 6. 可视化检查（强烈建议保留）
# =========================

plt.figure(figsize=(6, 5))
plt.imshow(value, cmap="jet")
plt.colorbar(label="Recovered value")
plt.title("Recovered 2D scalar field")
plt.tight_layout()
plt.show()


# # =========================
# # 7. 保存为 numpy 数据（模型 / 仿真可直接用）
# # =========================
#
# np.save("recovered_value.npy", value)
# print("Saved recovered_value.npy")
