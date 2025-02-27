import time
from agent_ddpg import *
import os
from env import env
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

current_path = os.path.dirname(os.path.abspath(__file__))
writer = SummaryWriter(log_dir=current_path + '/logs/')

e = env()

STATE_DIM = len(e.state_par)
ACTION_DIM = len(e.act_par)

agent = TD3Agent(STATE_DIM, ACTION_DIM)    # TODO

# Hyperparameters
NUM_EPISODE = 5000
NUM_STEP = 0
epsilon = 0.5



REWARD_BUFFER = np.empty(shape=NUM_EPISODE)
l1 = []
l2 = []
for episode_i in range(NUM_EPISODE):
    e.reset(episode_i)
    episode_reward = 0
    t = 0
    while True:
        if e.done:
            break
        random_sample = random.random()
        if episode_reward == 0 or random_sample < epsilon:
            e.step()
            state, action, next_state = e.s, e.a, e.s_
            reward = e.reward(state, next_state)  # 根据某种逻辑计算奖励
            agent.replay_buffer.add_memo(state, action, reward, next_state)
        else:
            state, action, reward, next_state = agent.replay_buffer.sample(1)
            state = state[0]
            action = action[0]
            next_state = next_state[0]
            reward = reward[0]
        episode_reward += reward
        cl1, cl2 = agent.update(e.done, t)

        writer.add_scalar('Loss/critic1_loss', cl1, NUM_STEP)
        writer.add_scalar('Loss/critic2_loss', cl2, NUM_STEP)
        # writer.add_scalar('Loss/actor_loss', al, NUM_STEP)
        # writer.add_scalar('Reward/pred_q', q[0], NUM_STEP)


        t += 1
        NUM_STEP += 1

    REWARD_BUFFER[episode_i] = episode_reward
    print(f"Episode: {episode_i + 1}")
    # print(f"Episode: {episode_i+1}, Current Reward:{episode_reward}, Mean Reward:{np.mean(REWARD_BUFFER[:episode_i+1])}")


# 绘制散点图
plt.scatter(l1, l2, color='green', s=5)

# 添加标题和标签
plt.title("strategy choice")
plt.xlabel("Q-value")
plt.ylabel("MSE")

# 显示图形
plt.tight_layout()
plt.grid()
plt.show()


model = current_path + '/models/'
timestamp = time.strftime("%Y%m%d-%H%M%S")
# Save models
torch.save(agent.actor.state_dict(), model + f"ddpg_actor_{timestamp}" + '.pth')
torch.save(agent.critic1.state_dict(), model + f"ddpg_critic1_{timestamp}" + '.pth')
torch.save(agent.critic2.state_dict(), model + f"ddpg_critic2_{timestamp}" + '.pth')