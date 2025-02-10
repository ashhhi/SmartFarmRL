import os
import torch.nn as nn
import torch
import gym
import pygame
import numpy as np
from env import env


device = 'mps'

e = env()

STATE_DIM = len(e.state_par)
ACTION_DIM = len(e.act_par)

# Load params
actor_path = '/DDPG_SmartFarm_Public/models/ddpg_actor_20241128-171053.pth'
critic_path = '/DDPG_SmartFarm_Public/models/ddpg_critic_20241128-171053.pth'


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
    print(f"Episode {episode_i}\t\t\tpredict:temp\thumidity\tCO2\t\tEC\t\tpH\t\twater\t\t\t\t\t\t\t\t\tactual")
    NUM_STEP = 0
    e.reset(episode_i)
    episode_reward = 0
    state = e.s
    while True:
        if e.done:
            break
        action = actor(torch.FloatTensor(state).unsqueeze(0).to(device))
        q_value = critic(torch.FloatTensor(state).unsqueeze(0).to(device), action).detach().cpu().numpy()[0][0]
        action = action.detach().cpu().numpy()[0]
        e.step()
        state = e.s
        next_state = e.s_
        np.set_printoptions(precision=4)    # 控制所有都是小数点后四位
        action_ = np.array(e.a)
        q_value_ = np.array(e.reward(state, next_state))
        # next_state, reward, done, _ = env.step(action)
        # state = next_state
        # episode_reward += reward
        print(f"Step {NUM_STEP} action:\t\t\t", action, "\t\t\t", action_)
        NUM_STEP += 1