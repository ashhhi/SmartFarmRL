import os
import torch.nn as nn
import torch
import gym
import pygame
import numpy as np
from env import env
import torch.nn.functional as F
from TD7 import Actor, Encoder, AvgL1Norm


device = 'mps'
np.set_printoptions(linewidth=120, precision=2, suppress=True)
e = env()

STATE_DIM = len(e.state_par)
ACTION_DIM = len(e.act_par)


# actor_path = '/Users/shijunshen/Documents/Code/PycharmProjects/ReinforcementLearning/DDPG_SmartFarm/models/no_best/ddpg_actor_20250121-162718.pth'
# critic_path = '/Users/shijunshen/Documents/Code/PycharmProjects/ReinforcementLearning/DDPG_SmartFarm/models/no_best/ddpg_critic_20250121-162718.pth'

actor_path = '/Users/shijunshen/Documents/Code/PycharmProjects/ReinforcementLearning/TD7/results/actor_final.pt'
encoder_path = '/Users/shijunshen/Documents/Code/PycharmProjects/ReinforcementLearning/TD7/results/encoder_final.pt'


actor = Actor(STATE_DIM, ACTION_DIM).to(device)
actor.load_state_dict(torch.load(actor_path))
encoder = Encoder(STATE_DIM, ACTION_DIM).to(device)
encoder.load_state_dict(torch.load(encoder_path))
NUM_EPISODE = 6

# Test phase
for episode_i in range(NUM_EPISODE):
    print("====================================================================================================================================================")
    print("[", end="")
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
    flag = 0
    while True:
        if e.done:
            flag = 1
        state = e.s
        action_actual = np.array(e.a)
        zs = encoder.zs(torch.FloatTensor(state).unsqueeze(0).to(device))

        action = actor(torch.FloatTensor(state).unsqueeze(0).to(device), zs)
        action = action.detach().cpu().numpy()[0]
        np.set_printoptions(precision=4)    # 控制所有都是小数点后四位

        action_actual = e.destandardize(action_actual, e.act_par)
        action = e.destandardize(action, e.act_par)
        print(f"Day {NUM_STEP + 1} action:")
        print("prediction: " + np.array2string(action.detach().cpu().numpy(),
                                               formatter={'float_kind': lambda x: "%.2f" % x}))
        print("actual:     " + np.array2string(action_actual.detach().cpu().numpy(),
                                               formatter={'float_kind': lambda x: "%.2f" % x}))
        print()
        NUM_STEP += 1
        if flag == 1:
            break
        e.step()