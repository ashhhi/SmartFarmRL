import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# 读取 CSV 文件
data = pd.read_csv('/Users/shijunshen/Downloads/去除best.csv')

# 假设 CSV 文件的列名分别为 'Step' 和 'Value'
x = data['Step']
y = data['Value']

# 绘制折线图
plt.plot(x, y)
plt.title('Actor Loss')
plt.xlabel('Step')
plt.ylabel('Value')

# 设置 y 轴为科学记数法
plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

# 调整 y 轴标签字体大小
plt.tick_params(axis='y', labelsize=8)  # 调整 y 轴标签字体大小为 8

# 可选：旋转 y 轴标签
plt.yticks(rotation=45)

plt.grid()
plt.show()