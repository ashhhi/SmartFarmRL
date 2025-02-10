import random

import numpy as np
import torch.nn as nn
from agent import Agent
import torch
from env import env
import torch.nn.functional as F

e = env()
EPSILON_DECAY = 100000
EPSILON_START = 1.0
EPSILON_END = 0.02
TARGET_UPDATE_FREQUENCY = 100

use_existing_data_ratio = 0.8

n_episode = 50000
n_time_step = 10

n_state = len(e.state_par)
n_action = len(e.act_par)

agent = Agent(n_input=n_state, n_output=n_action)

REWARD_BUFFER = np.empty(shape=n_episode)

plt_data = []

for episode_i in range(n_episode):
    e.reset(episode_i)
    s = e.s
    episode_reward = 0
    # 开始训练之前，填充满memory
    agent.memo.fill(e)

    for step_i in range(n_time_step):
        # print(step_i)
        a = agent.online_net.act(s)

        # 优先使用已有数据
        if random.random() < use_existing_data_ratio:
            # 从经验回放中随机抽取一条经验
            s, a, r, s_ = agent.memo.sample()
            index = random.choice(range(len(a)))
            s = s[index]
            a = a[index]
            r = r[index].squeeze(0)
            s_ = s_[index]
        else:
            # 使用 DynamicsModel 预测下一个状态和奖励
            state_tensor = torch.FloatTensor(s).unsqueeze(0)
            action_tensor = torch.FloatTensor(a).unsqueeze(0)

            # 预测下一个状态
            s_ = agent.dynamics_model(state_tensor, action_tensor).squeeze(0).detach()
            r = e.reward(s, s_)  # 根据某种逻辑计算奖励
            # done = check_done_condition(next_state)  # 判断新状态是否终止
        # print(r, episode_reward)
        episode_reward += r
        agent.memo.add_memo(s, a, r, s_)
        s = s_

        # 假设 agent.memo.sample() 返回的值是正确的
        batch_s, batch_a, batch_r, batch_s_ = agent.memo.sample()

        # 训练 Critic
        current_q_values = agent.critic_model(batch_s, batch_a)
        target_next_act = agent.target_net(batch_s_)
        target_next_q_values = agent.critic_model(batch_s_, target_next_act)
        target_q_values = batch_r + agent.GAMMA * target_next_q_values.detach()  # detach 以避免影响后续梯度计算

        critic_loss = F.mse_loss(current_q_values, target_q_values)
        agent.optimizer_critic.zero_grad()
        critic_loss.backward()
        agent.optimizer_critic.step()

        # 训练 DQN
        dqn_current_action = agent.online_net(batch_s)
        dqn_current_q_value = agent.critic_model(batch_s, dqn_current_action)

        # 使用 detach 确保不影响计算图
        dqn_loss = nn.functional.smooth_l1_loss(dqn_current_q_value, target_q_values.detach())
        agent.optimizer_dqn.zero_grad()
        dqn_loss.backward()
        agent.optimizer_dqn.step()

        '''训练Dynamic Model'''
        predicted_s_next = agent.dynamics_model(batch_s, batch_a)
        model_loss = nn.MSELoss()(predicted_s_next, batch_s_)

        agent.optimizer_model.zero_grad()
        model_loss.backward()
        agent.optimizer_model.step()

    REWARD_BUFFER[episode_i] = episode_reward

    plt_data.append(np.mean(REWARD_BUFFER[:episode_i]))
    if np.mean(REWARD_BUFFER[:episode_i]) >= 50:
        print('success！')
        break

    if episode_i % TARGET_UPDATE_FREQUENCY == 0:
        agent.target_net.load_state_dict(agent.online_net.state_dict())
        print(f"Episode: {episode_i}")
        print(f"Cur. Reward: {REWARD_BUFFER[episode_i]}; Avg. Reward: {np.mean(REWARD_BUFFER[:episode_i])}")


import matplotlib.pyplot as plt
plt.plot(plt_data, marker='o')  # 使用 'o' 标记点
plt.title('Reward-Episode')
plt.xlabel('Episode (x100)')
plt.ylabel('Reward')
plt.grid(True)  # 添加网格
plt.show()  # 显示图形


