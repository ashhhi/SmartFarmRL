import matplotlib.pyplot as plt

# # 数据 no best


# 数据 have best
temp_pred = [27.92, 27.33, 26.77, 26.04, 25.05, 25.02, 24.57]
temp_actu = [24.27, 24.92, 24.98, 24.54, 24.59, 26.45, 27.02]

humidity_pred = [57.76, 57.02, 56.33, 55.43, 53.97, 54.32, 53.69]
humidity_actu = [54.38, 49.55, 50.44, 52.45, 50.41, 53.12, 58.23]

# 创建图形和轴
fig, ax1 = plt.subplots()

# 创建时间序列，从1开始
days = list(range(1, len(temp_pred) + 1))

# 绘制温度数据
ax1.set_xlabel('Time (Days)')
ax1.set_ylabel('Temperature (°C)', color='tab:red')
ax1.plot(days, temp_pred, 'o-', color='tab:red', label='Predicted Temp')
ax1.plot(days, temp_actu, 'o-', color='tab:red', marker='^', label='Actual Temp')
ax1.tick_params(axis='y', labelcolor='tab:red')

# 创建第二个 Y 轴
ax2 = ax1.twinx()
ax2.set_ylabel('Humidity (%)', color='tab:blue')
ax2.plot(days, humidity_pred, 's-', color='tab:blue', label='Predicted Humidity')
ax2.plot(days, humidity_actu, 's-', color='tab:blue', marker='^', label='Actual Humidity')
ax2.tick_params(axis='y', labelcolor='tab:blue')

# 添加图例
fig.tight_layout()  
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# 显示图形
plt.title('Temperature and Humidity Over Time')
plt.show()