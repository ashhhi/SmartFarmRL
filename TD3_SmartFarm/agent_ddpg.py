import numpy as np
import torch
import torch.nn as nn
from collections import deque
import random

LR_ACTOR = 1e-4
LR_CRITIC = 1e-3
GAMMA = 0.99
MEMORY_SIZE = 16
BATCH_SIZE = 4
TAU = 5e-3

device = torch.device("mps")
useAutomaticWeightedLoss = True

class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss

    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum

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

class ReplayMemory:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add_memo(self, state, action, reward, next_state):
        state = np.expand_dims(state, axis=0)
        next_state = np.expand_dims(next_state, axis=0)
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        state, action, reward, next_state = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state)

    def __len__(self):
        return len(self.buffer)

class TD3Agent:
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        self.actor = Actor(state_dim, action_dim, hidden_dim=hidden_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim=hidden_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=LR_ACTOR)

        self.critic1 = Critic(state_dim, action_dim, hidden_dim=hidden_dim).to(device)
        self.critic1_target = Critic(state_dim, action_dim, hidden_dim=hidden_dim).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=LR_CRITIC)

        self.critic2 = Critic(state_dim, action_dim, hidden_dim=hidden_dim).to(device)
        self.critic2_target = Critic(state_dim, action_dim, hidden_dim=hidden_dim).to(device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=LR_CRITIC)

        self.replay_buffer = ReplayMemory(MEMORY_SIZE)

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.actor(state)
        return action.detach().cpu().numpy()[0]

    def update(self, dones, t):
        if len(self.replay_buffer) < BATCH_SIZE:
            # print(str(len(self.replay_buffer)) + "?")
            states, actions, rewards, next_states = self.replay_buffer.sample(len(self.replay_buffer))
        else:
            # print(str(len(self.replay_buffer)) + "!")
            states, actions, rewards, next_states = self.replay_buffer.sample(BATCH_SIZE)

        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(np.vstack(actions)).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)

        # Updata critic
        next_actions = self.actor_target(next_states)
        noise = torch.randn_like(next_actions)  # 生成与all_a形状相同的标准正态分布噪声
        col_mean = noise.mean(dim=0, keepdim=True)  # 按列计算均值
        col_std = noise.std(dim=0, keepdim=True)  # 按列计算标准差
        normalized_noise = (noise - col_mean) / col_std
        next_actions += normalized_noise

        target_Q1 = self.critic1_target(next_states, next_actions.detach())
        target_Q2 = self.critic2_target(next_states, next_actions.detach())
        # print(min(target_Q1, target_Q2))
        target_Q = rewards + GAMMA * (1 - dones) * torch.min(target_Q1, target_Q2)

        current_Q1 = self.critic1(states, actions)
        current_Q2 = self.critic2(states, actions)

        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1)
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), max_norm=1)
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=1)

        critic1_loss = nn.MSELoss()(current_Q1, target_Q)
        critic2_loss = nn.MSELoss()(current_Q2, target_Q)
        critic_loss = critic1_loss + critic2_loss
        critic_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        # Delay update actor
        loss1 = torch.max(-self.critic1(states, actions), -self.critic2(states, actions))

        if t % 2:
            loss2 = nn.MSELoss()(actions, self.actor(states))
            if useAutomaticWeightedLoss:
                awl = AutomaticWeightedLoss(2)
                actor_loss = awl(loss1, loss2).sum()
            else:
                actor_loss = (loss1 + loss2).sum()
            self.actor_optimizer.zero_grad()

            actor_loss.backward()
            self.actor_optimizer.step()
            # print(f"critic loss: {critic_loss}, actor loss: {actor_loss.item()}")
            # Update target networks of critic and actor
            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

            for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

            for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

        return critic1_loss.item(), critic2_loss.item()