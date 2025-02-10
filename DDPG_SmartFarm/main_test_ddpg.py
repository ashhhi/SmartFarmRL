import os
import torch.nn as nn
import torch
import gym
import pygame
import numpy as np
from env import env


device = 'mps'
np.set_printoptions(linewidth=120, precision=2, suppress=True)
e = env()

STATE_DIM = len(e.state_par)
ACTION_DIM = len(e.act_par)


# actor_path = '/Users/shijunshen/Documents/Code/PycharmProjects/ReinforcementLearning/DDPG_SmartFarm/models/no_best/ddpg_actor_20250121-162718.pth'
# critic_path = '/Users/shijunshen/Documents/Code/PycharmProjects/ReinforcementLearning/DDPG_SmartFarm/models/no_best/ddpg_critic_20250121-162718.pth'

actor_path = '/Users/shijunshen/Documents/Code/PycharmProjects/ReinforcementLearning/DDPG_SmartFarm/models/have_best/ddpg_actor_20250121-174612.pth'
critic_path = '/Users/shijunshen/Documents/Code/PycharmProjects/ReinforcementLearning/DDPG_SmartFarm/models/have_best/ddpg_critic_20250121-174612.pth'


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc3 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, action_dim)

        # Action range, need to customize

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)

        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc3 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


actor = Actor(STATE_DIM, ACTION_DIM).to(device)
actor.load_state_dict(torch.load(actor_path))
critic = Critic(STATE_DIM, ACTION_DIM).to(device)
critic.load_state_dict(torch.load(critic_path))

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
        action = actor(torch.FloatTensor(state).unsqueeze(0).to(device))
        q_value = critic(torch.FloatTensor(state).unsqueeze(0).to(device), action).detach().cpu().numpy()[0][0]
        action = action.detach().cpu().numpy()[0]
        np.set_printoptions(precision=4)    # 控制所有都是小数点后四位


        # next_state, reward, done, _ = env.step(action)
        # state = next_state
        # episode_reward += reward
        print(f"Day {NUM_STEP+1} action:")
        print("prediction: [" + ", ".join(f"{x:6.2f}" for x in np.round(action,2)) + "]")
        print("actual:     [" + ", ".join(f"{x:6.2f}" for x in np.round(action_actual,2)) + "]")
        print()
        NUM_STEP += 1
        if flag == 1:
            break
        e.step()