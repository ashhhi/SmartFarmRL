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

class DDPGAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        self.actor = Actor(state_dim, action_dim, hidden_dim=hidden_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim=hidden_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=LR_ACTOR)

        self.critic = Critic(state_dim, action_dim, hidden_dim=hidden_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim=hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        self.replay_buffer = ReplayMemory(MEMORY_SIZE)

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.actor(state)
        return action.detach().cpu().numpy()[0]

    def update(self, dones):
        if len(self.replay_buffer) < BATCH_SIZE:
            # print(str(len(self.replay_buffer)) + "?")
            states, actions, rewards, next_states = self.replay_buffer.sample(len(self.replay_buffer))
        else:
            # print(str(len(self.replay_buffer)) + "!")
            states, actions, rewards, next_states = self.replay_buffer.sample(BATCH_SIZE)
        # print(dones)
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(np.vstack(actions)).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)

        # Updata critic
        next_actions = self.actor_target(next_states)
        target_Q = self.critic_target(next_states, next_actions.detach())
        target_Q = rewards + GAMMA * target_Q * (1 - dones)
        current_Q = self.critic(states, actions)
        # print("==========================")
        # print(target_Q, current_Q)


        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1)

        # print(next_actions, "\n", actions)
        # print("==========================")
        critic_loss = nn.MSELoss()(current_Q, target_Q)
        # print(" ================ ")
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        # print(-self.critic(states, self.actor(states)).mean(), ", ", nn.MSELoss()(actions, self.actor(states)) * 0.01)
        actor_loss = -self.critic(states, self.actor(states)).mean() + nn.MSELoss()(actions, self.actor(states))
        # actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # print(actions, "\n", self.actor(states))
        # print(f"critic loss: {critic_loss}, actor loss: {actor_loss.item()}")
        # print(target_Q)
        # print()
        # print(actions, '\n',next_actions)
        # Update target networks of critic and actor
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

        return critic_loss.item(), actor_loss.item(), current_Q






















