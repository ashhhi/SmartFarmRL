import numpy as np
import matplotlib.pyplot as plt
from mpmath import sqrtm

# 假设 y_correct 是一个常数值，这里以 500 为例
y_correct = 500

# 设置 a 和 b 的范围
a = np.linspace(-100, 100, 100)  # a 从 -10 到 10，包含 100 个点
b = np.linspace(-100, 100, 100)  # b 从 -10 到 10，包含 100 个点

# 创建网格坐标
A, B = np.meshgrid(a, b)

# 计算 J 的值
J = (0.5 * A + 300 * B - y_correct)**2

# 创建二维图
plt.figure(figsize=(8, 6))

# 绘制等高线图
contour = plt.contourf(A, B, J, cmap='plasma')

# 设置标签
plt.xlabel('a')
plt.ylabel('b')
plt.title('Contour plot of J = (5a + 300b - y_correct)^2')

# 显示颜色条
plt.colorbar(contour)

# 显示图形
plt.show()