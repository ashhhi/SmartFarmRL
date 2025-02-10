import os
import torch.nn as nn
import torch
import gym
import pygame
import numpy as np
from sentry_sdk.metrics import increment

from env import env
import matplotlib.pyplot as plt

e = env()
NUM_EPISODE = 6

a_list = torch.empty((0, len(e.act_par)))
s_list = torch.empty((0, 1))
print(a_list)
# Test phase
for episode_i in range(NUM_EPISODE):
    for index in range(len(e.act_par)):
        sensor_type = e.act_par[index].replace(" sensor","")
        if sensor_type == 'Wind speed':
            sensor_type = 'Air flow'
        if index != len(e.act_par) - 1:
            print(sensor_type, end=", ")
        else:
            print(sensor_type, end="]\n\n")
    NUM_STEP = 0
    e.reset(episode_i)
    episode_reward = 0
    state = e.s
    while True:
        if e.done:
            break

        gs = e.s[0] * e.s[1]
        gs_ = e.s_[0] * e.s_[1]
        increment = gs_ - gs
        increment = increment.unsqueeze(0)
        s_list = torch.cat((s_list, increment.unsqueeze(0)), dim=0)
        a_list = torch.cat((a_list, e.a.unsqueeze(0)), dim=0)
        e.step()
        NUM_STEP += 1

# 去除负数值
s_list = torch.clamp(s_list, min=0)

# 获取第一列的排序索引
par_index = 7

sorted_indices = torch.argsort(a_list[:, par_index], descending=True)

sorted_s_list = s_list[sorted_indices]
sorted_a_list = a_list[sorted_indices]

x = sorted_a_list[:, par_index]
y = sorted_s_list[:, 0]
print(x, y)


# 绘制线图
plt.figure(figsize=(10, 6))
plt.plot(x.numpy(), y.numpy(), color='blue', marker='o', linestyle='-', markersize=5)
plt.title('Humidity vs Growth Score')
plt.xlabel('Humidity')
plt.ylabel('Growth Score')
plt.grid(True)

# 让 X 轴自适应范围
plt.autoscale(enable=True, axis='x', tight=True)

# 显示图形
plt.show()