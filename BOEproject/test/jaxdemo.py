import jax
import time
import jax.numpy as jnp
from jax import grad
from jax import vmap

"""
JAX 简介
JAX 是一个由 Google 开发的高性能库，专为 科学计算 和 机器学习 设计。其主要特点是：
NumPy 的灵活替代品：JAX 支持几乎所有的 NumPy 接口，同时具有更高性能。
硬件加速：自动利用 GPU 或 TPU 提高计算速度。
自动微分：内置梯度计算支持（通过 jax.grad 等工具）。
即使编译（JIT 编译）：使用 XLA（Accelerated Linear Algebra）编译动态代码为高效的机器指令。
分布式计算：支持多设备、分片计算和大型阵列分布式计算。
JAX 的主要功能
1. 自动微分 (Autograd)
    JAX 支持对任意函数进行一阶和高阶微分。
"""


def f(x):
    return x ** 2 + 3 * x + 2


df = grad(f)
print(df(2.0))

dff = grad(grad(f))

print(dff(2.0))

"""
2. JIT 编译
    使用 jax.jit 将函数加速并自动运行在 GPU 或 TPU 上。
"""


def cost(func):
    def decorator(*args):
        start = time.time()
        func(*args)
        end = time.time()
        costs = end - start
        print("costs -----> ", costs)
        return costs

    return decorator


@cost
@jax.jit
def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))


x = jnp.arange(1000000000)

y = sigmoid(x)
print(y)

"""
通过 jax.jit，代码将被编译为高效指令。
"""

"""
3. 向量化 (Vectorization)
    通过 jax.vmap 使标量运算自动扩展到批量数据。
"""


def square(x):
    return x ** 2


batched_square = vmap(square)

x = jnp.array([1, 2, 3, 4, 5])

print(batched_square(x))

"""
4. 硬件支持
GPU：只需安装 CUDA 对应的 JAX 版本，无需修改代码即可在 GPU 上运行。
TPU：同样支持 TPU 加速，适合大规模计算。
例如：
bash
# 安装支持 GPU 的 JAX
pip install jax[cuda] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
JAX 和 NumPy 的对比
功能	NumPy	JAX
自动微分	不支持	内置 grad 支持
硬件加速	CPU 支持	GPU / TPU 加速
JIT 编译	不支持	动态加速
分布式计算	不支持	原生多设备和分布式计算
API	数学计算、数组操作	接近 NumPy，简单易用
JAX 的应用场景
机器学习：
利用自动微分快速构建模型和梯度计算。
加速深度学习模型训练（如基于 Flax 的 JAX 深度学习框架）。
物理模拟：
高效地求解复杂的物理公式。
使用 GPU 并行加速仿真任务。
科学研究：
数值计算和优化（如梯度下降）。
自动求导和分布式计算。
强化学习：
配合其他框架，如 RLlib 或 Haiku，构建 RL 系统。
"""