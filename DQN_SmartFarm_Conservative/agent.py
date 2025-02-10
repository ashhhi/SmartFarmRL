import numpy as np
import torch
import torch.nn as nn
import random


class Replaymemory:
    def __init__(self, n_s, n_a):
        self.n_s = n_s
        self.n_a = n_a
        self.MEMORY_SIZE = 100
        self.BATCH_SIZE = 64

        self.all_s = torch.empty(size=(self.MEMORY_SIZE, self.n_s), dtype=torch.float32)
        self.all_a = torch.empty(size=(self.MEMORY_SIZE, self.n_a), dtype=torch.uint8)
        self.all_r = torch.empty(self.MEMORY_SIZE, dtype=torch.float32)
        # self.all_done = np.random.randint(low=0, high=2, size=self.MEMORY_SIZE, dtype=np.uint8)
        self.all_s_ = torch.empty(size=(self.MEMORY_SIZE, self.n_s), dtype=torch.float32)

        self.t_memo = 0
        self.t_max = 0

    def fill(self, e):
        for index in range(len(e.all_s)):
            if index == len(e.all_s)-1:
                break
            self.add_memo(e.all_s[index], e.all_a[index], e.reward(e.all_s[index], e.all_s[index+1]), e.all_s[index+1])
        self.MEMORY_SIZE = len(e.all_s)
        self.t_memo = 0
        return 0

    def add_memo(self, s, a, r, s_):
        # if isinstance(s, np.ndarray):
        #     s = torch.tensor(s, dtype=torch.float32)
        #     a = torch.tensor(a, dtype=torch.uint8)
        #     r = torch.tensor(r, dtype=torch.float32)
        #     s_ = torch.tensor(s_, dtype=torch.float32)
        self.all_s[self.t_memo] = s.clone().detach().float()
        self.all_a[self.t_memo] = a.clone().detach().to(torch.uint8)
        self.all_r[self.t_memo] = r.clone().detach().float()
        # self.all_done [self.t_memo]= done
        self.all_s_[self.t_memo] = s_.clone().detach().float()
        self.t_max = max(self.t_max, self.t_memo+1)
        self.t_memo = (self.t_memo + 1) % self.MEMORY_SIZE

    def sample(self):
        if self.t_max >= self.BATCH_SIZE:
            idxes = random.sample(range(0, self.t_max), self.BATCH_SIZE)
        else:
            idxes = range(0, self.t_max)

        batch_s = []
        batch_a = []
        batch_r = []
        # batch_done = []
        batch_s_ = []
        for idx in idxes:
            batch_s.append(self.all_s[idx])
            batch_a.append(self.all_a[idx])
            batch_r.append(self.all_r[idx])
            # batch_done.append(self.all_done[idx])
            batch_s_.append(self.all_s_[idx])

        batch_s = torch.stack(batch_s)
        batch_a = torch.stack(batch_a)
        batch_r = torch.stack(batch_r).unsqueeze(1)
        batch_s_ = torch.stack(batch_s_)

        return batch_s, batch_a, batch_r, batch_s_


class Critic(nn.Module):
    def __init__(self, n_s, n_a):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_s + n_a, 24),  # 输入状态和动作的大小
            nn.ReLU(),
            nn.Linear(24, 1),  # 输出 Q 值
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


class DQN(nn.Module):
    def __init__(self, n_s, n_a):
        super().__init__()
        self.n_s = n_s
        self.n_a = n_a

        self.net = nn.Sequential(
            nn.Linear(n_s, 24),
            nn.ReLU(),
            nn.Linear(24, n_a),
            # nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)

    # def act(self, obs):
    #     # print(type(obs))
    #     obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
    #     actions = self(obs_tensor.unsqueeze(0)).squeeze(0)
    #
    #     a_tair = torch.clamp(actions[0], 0, 5)
    #     a_rhair = torch.clamp(actions[1], 0, 19)
    #     a_co2air = torch.clamp(actions[2], 0, 12)
    #     a_ec_drain = torch.clamp(actions[3], 0, 8)
    #     a_ph_drain = torch.clamp(actions[4], 0, 6)
    #     a_water_sup = torch.clamp(actions[5], 0, 10)
    #
    #     return torch.stack([a_tair, a_rhair, a_co2air, a_ec_drain, a_ph_drain, a_water_sup])

class Agent:
    def __init__(self, n_input, n_output):
        self.n_input = n_input
        self.n_output = n_output

        self.GAMMA = 0.9
        self.learning_rate = 1e-3

        self.memo = Replaymemory(n_s=n_input, n_a=n_output)

        self.online_net = DQN(n_s=n_input, n_a=n_output)
        self.target_net = DQN(n_s=n_input, n_a=n_output)
        self.critic_model = Critic(n_s=n_input, n_a=n_output)

        self.optimizer_dqn = torch.optim.Adam(self.online_net.parameters(), lr=self.learning_rate)
        self.optimizer_critic = torch.optim.Adam(self.critic_model.parameters(), lr=0.001)


