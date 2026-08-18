from PIL import Image
import matplotlib
import numpy as np
import diffractsim
diffractsim.set_backend("CPU") #Change the string to "CUDA" to use GPU acceleration
from diffractsim.util.backend_functions import backend as bd
from diffractsim import  cm, mm, um, nm, cf
# 测试 sparkle_sim 类
from diffractsim.sparkle_simulator import SparkleSimulator
import pandas as pd
######################### SIM calibration parameters calculate##############
module = 8.8 ###模组名称

def calc_SIM(image_path, v_min, v_max,  Module, method, spectrum='G', intensity=5,):
    '''SIM 计算函数
    Parameters
    ----------
    image_path: AG图片路径
    v_min： AG三维形貌最小值
    v_max： AG三维形貌最大值
    Module：模组：8.8， 13.2， 13.8
    method： 刻蚀方法： A:对应黄光AG B:对应普通蒙砂刻蚀AG
    spectrum: 光谱： 'R', 'G', 'B'
    intensity: 光强，默认值为5

    Returns
    -------

    '''

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
    sag = SparkleSimulator.jetimg2sag(image_path = image_path, v_min=v_min*um, v_max=v_max*um)
    sag = SparkleSimulator.mosaic_array(sag,5)
    # plt.figure();plt.imshow(sag, cmap='jet');plt.colorbar();plt.title('Sag map from jet image');plt.show()
    # 定义刻蚀层参数
    AGetchparams = {
        'width': 277.*um*5,
        'height':207.*um*5,
        'sag': sag*1,
        'xyorigin': [0.0*um, 0.0*um]
    }
    # 创建 SparkleSimulator 实例， 得到闪点图的仿真结果
    sim = SparkleSimulator(wavelength=550 * nm, extent_x=1000*um, extent_y=1000*um,Nx=512*1, Ny=512*1 )######8.8
    Isparkle = sim.get_Sparkle_origin(spectrum=spectrum, layer_Structure=layerstructure, AGetch_params=AGetchparams,intensity=intensity, Module = Module )
    sim.plot_intensity(Isparkle, square_root= False, units = um, text = 'Sparkle Pattern at RGB layer(unfilter)',colormap = 'gray')
    # # #################### SIM method ######################
    sim.SIM(method = method)#  A:对应黄光AG B:对应普通蒙砂刻蚀AG
    return sim.SparkValue

# ==========================
# 数据集计算及SIM拟合预测
# ==========================
import json
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

from scipy.stats import pearsonr

import matplotlib.pyplot as plt

# --------------------------
# 读取Excel
# --------------------------
df = pd.read_excel("sparkle_database0808.xlsx")

print("样本数量:", len(df))

# --------------------------
# 计算所有仿真SIM
# --------------------------

sim_values = []
matplotlib.use('Agg')
for idx, row in df.iterrows():

    print(
        f"Calculating {idx+1}/{len(df)} : "
        f"{row['sample_id']}"
    )

    sim_value = calc_SIM(
        image_path=row["image_path"],
        v_min=row["v_min"],
        v_max=row["v_max"],
        Module=module,
        method=row["method"]
    )

    sim_values.append(sim_value)

# 添加到表格

df["SIM_sim"] = sim_values

# --------------------------
# 线性拟合
# SIM_exp = a * SIM_sim + b
# --------------------------

X = df["SIM_sim"].values.reshape(-1, 1)

y = df["SIM_exp"].values

model = LinearRegression()

model.fit(X, y)

a = model.coef_[0]
b = model.intercept_

print("\n拟合结果:")
print(f"SIM_exp = {a:.6f} * SIM_sim + {b:.6f}")

# --------------------------
# 拟合预测
# --------------------------

y_pred = model.predict(X)

df["SIM_pred"] = y_pred

# --------------------------
# 评价指标
# --------------------------

r2 = r2_score(y, y_pred)

r, p_value = pearsonr(
    df["SIM_sim"],
    df["SIM_exp"]
)

mae = mean_absolute_error(y, y_pred)

rmse = np.sqrt(
    mean_squared_error(y, y_pred)
)

print("\n评价指标")
print("-" * 40)
print(f"Pearson r = {r:.4f}")
print(f"p-value   = {p_value:.4e}")
print(f"R²        = {r2:.4f}")
print(f"MAE       = {mae:.4f}")
print(f"RMSE      = {rmse:.4f}")

# --------------------------
# 保存结果Excel
# --------------------------

df.to_excel(
    "sparkle_fit_result.xlsx",
    index=False
)

print("\n结果已保存:")
print("sparkle_fit_result.xlsx")

# --------------------------
# 保存拟合参数
# --------------------------

fit_result = {
    "a": float(a),
    "b": float(b),
    "pearson_r": float(r),
    "p_value": float(p_value),
    "R2": float(r2),
    "MAE": float(mae),
    "RMSE": float(rmse)
}

with open(
        "sim_calibration.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        fit_result,
        f,
        indent=4,
        ensure_ascii=False
    )

print("拟合参数已保存:")
print("sim_calibration.json")

# --------------------------
# 绘制拟合图
# --------------------------

plt.figure(figsize=(7, 6))

plt.scatter(
    df["SIM_sim"],
    df["SIM_exp"],
)

x_line = np.linspace(
    df["SIM_sim"].min(),
    df["SIM_sim"].max(),
    200
)

y_line = model.predict(
    x_line.reshape(-1, 1)
)

plt.plot(
    x_line,
    y_line,
)

plt.xlabel("Simulated SIM")
plt.ylabel("Experimental SIM")

plt.title(
    f"R²={r2:.4f}, r={r:.4f}"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "SIM_calibration.png",
    dpi=300
)

plt.show()

print("拟合图已保存:")
print("SIM_calibration.png")

