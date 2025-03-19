from tqdm import tqdm
import random
from data.env import env
import torch
import os
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Callable
import numpy as np
from models.DDPG import DDPG
from models.TD7 import TD7

@dataclass
class Hyperparameters:
	# Generic
	batch_size: int = 4
	buffer_size: int = 1000
	discount: float = 0.99
	target_update_rate: int = 250
	exploration_noise: float = 0.1
	model_path: str = ''

	# DDPG
	tau = 5e-3

	# TD3
	target_policy_noise: float = 0.2
	noise_clip: float = 0.5
	policy_freq: int = 2

	# Critic Model
	critic_hdim: int = 256
	critic_activ: Callable = F.elu
	critic_lr: float = 3e-4

	# Actor Model
	actor_hdim: int = 256
	actor_activ: Callable = F.relu
	actor_lr: float = 3e-4

	# LAP
	alpha: float = 0.4
	min_priority: float = 1

	# TD3+BC
	lmbda: float = 0.1

	# Encoder Model
	zs_dim: int = 256
	enc_hdim: int = 256
	enc_activ: Callable = F.elu
	encoder_lr: float = 3e-4



class Coach:
	def __init__(self, model):
		self.num_episodes = 5000
		self.epsilon = 0.5

		# env model
		self.env = env()
		self.state_dim = len(self.env.state_par)
		self.action_dim = len(self.env.act_par)

		self.model = model
		self.model_path = "models/" + model + "/checkpoints"
		if not os.path.exists(self.model_path):
			os.makedirs(self.model_path)
		self.hp = Hyperparameters(model_path="models/" + model + "/checkpoints")

	def train(self):
		print(f"Training model [{self.model}]")
		if self.model == 'DDPG':
			RL_agent = DDPG.Agent(self.state_dim, self.action_dim, self.hp)
		elif self.model == 'TD7':
			RL_agent = TD7.Agent(self.state_dim, self.action_dim, self.hp)
		else:
			raise NotImplementedError

		# Start Train
		NUM_STEP = 0
		for episode in tqdm(range(self.num_episodes)):
			self.env.reset(episode)
			episode_reward = 0
			while True:
				if self.env.done:
					break
				random_sample = random.random()
				if episode_reward == 0 or random_sample < self.epsilon:
					self.env.step()
					state, action, next_state = self.env.s, self.env.a, self.env.s_
					reward = self.env.reward(state, next_state)
					RL_agent.replay_buffer.add_memo(state, action, next_state, reward)
				else:
					state, action, reward, next_state = RL_agent.replay_buffer.sample(1)
					reward = reward[0]

				episode_reward += reward
				RL_agent.update(self.env.done)
				NUM_STEP += 1
	def test(self):
		pass

if __name__ == '__main__':
	torch.manual_seed(0)
	np.random.seed(0)
	coach = Coach("DDPG")
	is_train = True

	if is_train:
		coach.train()
	else:
		coach.test()