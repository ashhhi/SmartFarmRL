import argparse
import os
import time
import random
import gym
import numpy as np
import torch
import TD7
from env import env
from tqdm import tqdm

# RL
seed = 0
use_checkpoints = False
# Evaluation
timesteps_before_training = 25e3
eval_freq = 5e3
eval_eps = 10
num_episodes = 5000
# File
file_name = None

epsilon = 0.5

def train(RL_agent, env):
	NUM_STEP = 0
	for episode in tqdm(range(num_episodes)):
		env.reset(episode)
		episode_reward = 0

		while True:
			if e.done:
				# print(f"Total T: {episode + 1} Reward: {episode_reward:.3f}")
				break

			random_sample = random.random()
			if episode_reward == 0 or random_sample < epsilon:
				e.step()
				state, action, next_state = e.s, e.a, e.s_
				reward = e.reward(state, next_state)  # 根据某种逻辑计算奖励
				RL_agent.replay_buffer.add_memo(state, action, next_state, reward)
			else:
				state, action, reward, next_state = RL_agent.replay_buffer.sample(1)
				reward = reward[0][0]
			episode_reward += reward
			RL_agent.update(e.done)

			NUM_STEP += 1


if __name__ == "__main__":
	if not os.path.exists("./results"):
		os.makedirs("./results")
	e = env()
	torch.manual_seed(seed)
	np.random.seed(seed)

	state_dim = len(e.state_par)
	action_dim = len(e.act_par)

	# max_action = float(env.action_space.high[0])

	RL_agent = TD7.Agent(state_dim, action_dim)

	train(RL_agent, e)