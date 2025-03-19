import numpy as np
import torch
import torch.nn as nn
from collections import deque
import random
import os
from torch.utils.tensorboard import SummaryWriter
current_path = os.path.dirname(os.path.abspath(__file__))
writer = SummaryWriter(log_dir=current_path + '/logs/')

useAutomaticWeightedLoss = False

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

    def add_memo(self, state, action, next_state, reward):
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        state, action, reward, next_state = zip(*random.sample(self.buffer, batch_size))
        return state, action, reward, next_state

    def __len__(self):
        return len(self.buffer)

class Agent:
    def __init__(self, state_dim, action_dim, hp):
        self.hp = hp
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(state_dim, action_dim, hidden_dim=hp.actor_hdim).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim=hp.actor_hdim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=hp.actor_lr)

        self.critic = Critic(state_dim, action_dim, hidden_dim=hp.critic_hdim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim=hp.critic_hdim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=hp.critic_lr)

        self.replay_buffer = ReplayMemory(self.hp.buffer_size)
        self.training_steps = 0

    def update(self, dones):
        self.training_steps += 1
        if len(self.replay_buffer) < self.hp.batch_size:
            # print(str(len(self.replay_buffer)) + "?")
            states, actions, rewards, next_states = self.replay_buffer.sample(len(self.replay_buffer))
        else:
            # print(str(len(self.replay_buffer)) + "!")
            states, actions, rewards, next_states = self.replay_buffer.sample(self.hp.batch_size)
        # print(dones)
        states = torch.stack(states).to(self.device)
        actions = torch.FloatTensor(np.vstack(actions)).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.stack(next_states).to(self.device)

        # Updata critic
        next_actions = self.actor_target(next_states)
        target_Q = self.critic_target(next_states, next_actions.detach())
        target_Q = rewards + self.hp.discount * target_Q * (1 - dones)
        current_Q = self.critic(states, actions)

        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1)

        critic_loss = nn.MSELoss()(current_Q, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        Q = self.critic(states, self.actor(states)).mean()
        behavior_cloning = nn.MSELoss()(actions, self.actor(states))
        if useAutomaticWeightedLoss:
            awl = AutomaticWeightedLoss(2)
            actor_loss = awl(-Q, behavior_cloning)
        else:
            actor_loss = -Q + self.hp.lmbda * Q.abs().mean().detach() * behavior_cloning
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks of critic and actor
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.hp.tau * param.data + (1 - self.hp.tau) * target_param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.hp.tau * param.data + (1 - self.hp.tau) * target_param.data)

        writer.add_scalar('Loss/critic_loss', critic_loss, self.training_steps)
        writer.add_scalar('Loss/actor_loss', actor_loss, self.training_steps)

        torch.save(self.actor.state_dict(), os.path.join(self.hp.model_path, 'actor_final.pt'))