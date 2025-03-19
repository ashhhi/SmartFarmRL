import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.tensorboard import SummaryWriter
from models.TD7 import buffer
current_path = os.path.dirname(os.path.abspath(__file__))
writer = SummaryWriter(log_dir=current_path + '/logs/')



def AvgL1Norm(x, eps=1e-8):
	return x/x.abs().mean(-1,keepdim=True).clamp(min=eps)
	# return x

def LAP_huber(x, min_priority=1):
	return torch.where(x < min_priority, 0.5 * x.pow(2), min_priority * x).sum(1).mean()


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256, activ=F.relu):
		super(Actor, self).__init__()

		self.activ = activ

		self.l0 = nn.Linear(state_dim, hdim)
		self.l1 = nn.Linear(zs_dim + hdim, hdim)
		self.l2 = nn.Linear(hdim, hdim)
		self.l3 = nn.Linear(hdim, action_dim)

	def forward(self, state, zs):
		a = AvgL1Norm(self.l0(state))
		a = torch.cat([a, zs], 1)
		a = self.activ(self.l1(a))
		a = self.activ(self.l2(a))
		# return torch.tanh(self.l3(a))
		return self.l3(a)

class Encoder(nn.Module):
	def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256, activ=F.elu):
		super(Encoder, self).__init__()

		self.activ = activ

		# state encoder
		self.zs1 = nn.Linear(state_dim, hdim)
		self.zs2 = nn.Linear(hdim, hdim)
		self.zs3 = nn.Linear(hdim, zs_dim)
		
		# state-action encoder
		self.zsa1 = nn.Linear(zs_dim + action_dim, hdim)
		self.zsa2 = nn.Linear(hdim, hdim)
		self.zsa3 = nn.Linear(hdim, zs_dim)


	def zs(self, state):
		zs = self.activ(self.zs1(state))
		zs = self.activ(self.zs2(zs))
		zs = AvgL1Norm(self.zs3(zs))
		return zs


	def zsa(self, zs, action):
		zsa = self.activ(self.zsa1(torch.cat([zs, action], 1)))
		zsa = self.activ(self.zsa2(zsa))
		zsa = self.zsa3(zsa)
		return zsa


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256, activ=F.elu):
		super(Critic, self).__init__()

		self.activ = activ
		
		self.q01 = nn.Linear(state_dim + action_dim, hdim)
		self.q1 = nn.Linear(2*zs_dim + hdim, hdim)
		self.q2 = nn.Linear(hdim, hdim)
		self.q3 = nn.Linear(hdim, 1)

		self.q02 = nn.Linear(state_dim + action_dim, hdim)
		self.q4 = nn.Linear(2*zs_dim + hdim, hdim)
		self.q5 = nn.Linear(hdim, hdim)
		self.q6 = nn.Linear(hdim, 1)


	def forward(self, state, action, zsa, zs):
		# Double Q-learning
		sa = torch.cat([state, action], 1)
		embeddings = torch.cat([zsa, zs], 1)

		q1 = AvgL1Norm(self.q01(sa))
		q1 = torch.cat([q1, embeddings], 1)
		q1 = self.activ(self.q1(q1))
		q1 = self.activ(self.q2(q1))
		q1 = self.q3(q1)

		q2 = AvgL1Norm(self.q02(sa))
		q2 = torch.cat([q2, embeddings], 1)
		q2 = self.activ(self.q4(q2))
		q2 = self.activ(self.q5(q2))
		q2 = self.q6(q2)
		return torch.cat([q1, q2], 1)


class Agent(object):
	def __init__(self, state_dim, action_dim, hp, offline=True):
		self.hp = hp
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.actor = Actor(state_dim, action_dim, hp.zs_dim, hp.actor_hdim, hp.actor_activ).to(self.device)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=hp.actor_lr)
		self.actor_target = copy.deepcopy(self.actor)

		self.critic = Critic(state_dim, action_dim, hp.zs_dim, hp.critic_hdim, hp.critic_activ).to(self.device)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=hp.critic_lr)
		self.critic_target = copy.deepcopy(self.critic)

		self.encoder = Encoder(state_dim, action_dim, hp.zs_dim, hp.enc_hdim, hp.enc_activ).to(self.device)
		self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=hp.encoder_lr)
		self.fixed_encoder = copy.deepcopy(self.encoder)
		self.fixed_encoder_target = copy.deepcopy(self.encoder)

		self.checkpoint_actor = copy.deepcopy(self.actor)
		self.checkpoint_encoder = copy.deepcopy(self.encoder)

		self.replay_buffer = buffer.LAP(state_dim, action_dim, self.device, hp.buffer_size, prioritized=True)

		self.offline = offline

		self.training_steps = 0

		# Value clipping tracked values
		self.max = -1e8
		self.min = 1e8
		self.max_target = 0
		self.min_target = 0

	def update(self, done):
		self.training_steps += 1
		if len(self.replay_buffer) < self.hp.batch_size:
			state, action, reward, next_state = self.replay_buffer.sample(len(self.replay_buffer))
		else:
			state, action, reward, next_state = self.replay_buffer.sample(self.hp.batch_size)

		#########################
		# Update Encoder
		#########################
		with torch.no_grad():
			next_zs = self.encoder.zs(next_state)

		zs = self.encoder.zs(state)
		pred_zs = self.encoder.zsa(zs, action)
		encoder_loss = F.mse_loss(pred_zs, next_zs)

		self.encoder_optimizer.zero_grad()
		encoder_loss.backward()
		self.encoder_optimizer.step()

		#########################
		# Update Critic
		#########################
		with torch.no_grad():
			fixed_target_zs = self.fixed_encoder_target.zs(next_state)

			noise = (torch.randn_like(action) * self.hp.target_policy_noise).clamp(-self.hp.noise_clip, self.hp.noise_clip)
			next_action = (self.actor_target(next_state, fixed_target_zs) + noise).clamp(-1,1)
			
			fixed_target_zsa = self.fixed_encoder_target.zsa(fixed_target_zs, next_action)

			Q_target = self.critic_target(next_state, next_action, fixed_target_zsa, fixed_target_zs).min(1,keepdim=True)[0]
			Q_target = reward + done * self.hp.discount * Q_target.clamp(self.min_target, self.max_target)

			self.max = max(self.max, float(Q_target.max()))
			self.min = min(self.min, float(Q_target.min()))

			fixed_zs = self.fixed_encoder.zs(state)
			fixed_zsa = self.fixed_encoder.zsa(fixed_zs, action)

		Q = self.critic(state, action, fixed_zsa, fixed_zs)

		# huber loss
		td_loss = (Q - Q_target).abs()
		critic_loss = LAP_huber(td_loss)
		# critic_loss = nn.MSELoss()(Q, Q_target)

		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()
		
		#########################
		# Update LAP
		#########################
		priority = td_loss.max(1)[0].clamp(min=self.hp.min_priority).pow(self.hp.alpha)
		self.replay_buffer.update_priority(priority)

		#########################
		# Update Actor
		#########################
		if self.training_steps % self.hp.policy_freq == 0:
			actor = self.actor(state, fixed_zs)
			fixed_zsa = self.fixed_encoder.zsa(fixed_zs, actor)
			Q = self.critic(state, actor, fixed_zsa, fixed_zs)
			actor_loss = -Q.mean()
			if self.offline:	# behavior cloning
				actor_loss = actor_loss + self.hp.lmbda * Q.abs().mean().detach() * F.mse_loss(actor, action)

			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()
			# print(critic_loss, actor_loss)
			writer.add_scalar('Loss/critic_loss', critic_loss, self.training_steps)
			writer.add_scalar('Loss/actor_loss', actor_loss, self.training_steps)

		if self.training_steps % 10000 == 0:
			torch.save(self.actor.state_dict(), os.path.join(self.hp.model_path, f'actor_{self.training_steps}.pt'))
			torch.save(self.encoder.state_dict(), os.path.join(self.hp.model_path, f'encoder_{self.training_steps}.pt'))

		#########################
		# Update Iteration
		#########################
		if self.training_steps % self.hp.target_update_rate == 0:
			self.actor_target.load_state_dict(self.actor.state_dict())
			self.critic_target.load_state_dict(self.critic.state_dict())
			self.fixed_encoder_target.load_state_dict(self.fixed_encoder.state_dict())
			self.fixed_encoder.load_state_dict(self.encoder.state_dict())
			
			self.replay_buffer.reset_max_priority()

			self.max_target = self.max
			self.min_target = self.min
		torch.save(self.actor.state_dict(), os.path.join(self.hp.model_path, 'actor_final.pt'))
		torch.save(self.encoder.state_dict(), os.path.join(self.hp.model_path, 'encoder_final.pt'))
