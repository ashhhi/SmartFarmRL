import argparse
import os
import time
import random
import gym
import numpy as np
import torch
from env import env
import TD7

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

def train(RL_agent, env, eval_env):
	evals = []
	start_time = time.time()
	allow_train = False
	NUM_STEP = 0
	for episode in range(num_episodes):
		state = env.reset(episode)
		episode_reward = 0

		while True:
			if e.done:
				break
			random_sample = random.random()
			if episode_reward == 0 or random_sample < epsilon:
				e.step()
				state, action, next_state = e.s, e.a, e.s_
				reward = e.reward(state, next_state)  # 根据某种逻辑计算奖励
				RL_agent.replay_buffer.add_memo(state, action, next_state, reward)
			else:
				state, action, reward, next_state = RL_agent.replay_buffer.sample(1)
				state = state[0]
				action = action[0]
				next_state = next_state[0]
				reward = reward[0][0]
			state = next_state
			print(reward, episode_reward)
			episode_reward += reward
			RL_agent.update(e.done)


			NUM_STEP += 1

			# if allow_train and not use_checkpoints:
			# 	RL_agent.train()
			#
			# if ep_finished:
			# 	print(f"Total T: {episode+1} Episode Num: {ep_num} Episode T: {ep_timesteps} Reward: {ep_total_reward:.3f}")
			#
			# 	if allow_train and use_checkpoints:
			# 		RL_agent.maybe_train_and_checkpoint(ep_timesteps, ep_total_reward)
			#
			# 	if t >= timesteps_before_training:
			# 		allow_train = True
			#
			# 	state, done = env.reset(), False
			# 	ep_total_reward, ep_timesteps = 0, 0
			# 	ep_num += 1

def maybe_evaluate_and_print(RL_agent, eval_env, evals, t, start_time, args, d4rl=False):
	if t % args.eval_freq == 0:
		print("---------------------------------------")
		print(f"Evaluation at {t} time steps")
		print(f"Total time passed: {round((time.time()-start_time)/60.,2)} min(s)")

		total_reward = np.zeros(args.eval_eps)
		for ep in range(args.eval_eps):
			state, done = eval_env.reset(), False
			while not done:
				action = RL_agent.select_action(np.array(state), args.use_checkpoints, use_exploration=False)
				state, reward, done, _ = eval_env.step(action)
				total_reward[ep] += reward

		print(f"Average total reward over {args.eval_eps} episodes: {total_reward.mean():.3f}")
		if d4rl:
			total_reward = eval_env.get_normalized_score(total_reward) * 100
			print(f"D4RL score: {total_reward.mean():.3f}")
		
		print("---------------------------------------")

		evals.append(total_reward)
		np.save(f"./results/{args.file_name}", evals)


if __name__ == "__main__":
	if file_name is None:
		file_name = f"TD7_{seed}"

	if not os.path.exists("./results"):
		os.makedirs("./results")

	e = env()
	eval_e = env()
	torch.manual_seed(seed)
	np.random.seed(seed)

	state_dim = len(e.state_par)
	action_dim = len(e.act_par)

	# max_action = float(env.action_space.high[0])

	RL_agent = TD7.Agent(state_dim, action_dim)

	train(RL_agent, e, eval_e)