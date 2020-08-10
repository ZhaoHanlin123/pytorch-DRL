import numpy as np
import gym
import matplotlib.pyplot as plt
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


env = gym.make('CartPole-v0')
state_space = env.observation_space.shape[0]   # 状态空间
action_space = env.action_space.n  # 动作空间

# hyper parameters  超参数
num_episode = 300
gamma = 0.98
batch_size = 32
learning_rate = 0.01

is_render = False
num_step = 1000
checkpoints_path =  os.path.dirname(os.path.abspath(__file__))+'/checkpoints/'
print(checkpoints_path)

class Policy(nn.Module):
    """
    计算状态s所对应的value
    """
    def __init__(self):
        super(Policy, self).__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.reward = []
        self.log_probs = []
        self.fc1 = nn.Linear(self.state_space, 64)
        self.fc2 = nn.Linear(64, self.action_space)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=-1)

        return x

def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    choose = Categorical(probs)
    action = choose.sample()
    policy.log_probs.append(choose.log_prob(action))
    return action.item()


def update_policy():
    R = 0
    reward = []
    policy_loss = []
    for r in policy.reward[::-1]:
        R = r + gamma * R
        reward.insert(0, R)

    # 归一化
    reward_mean = np.mean(reward)
    reward_std = np.std(reward)
    reward = (reward - reward_mean) / reward_std

    for reward, log_prob in zip(reward, policy.log_probs):
        policy_loss.append(-reward*log_prob)

    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.reward[:]
    del policy.log_probs[:]


policy = Policy()  # 创建动作策略
optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)


def train():
    for episode in range(num_episode):
        state = env.reset()

        for step in range(num_step):
            if is_render:  # 更新渲染
                env.render()
            action = select_action(state)  # 选择动作

            next_state, reward, done, _ = env.step(action)  # 与环境交互
            reward = 0 if done else reward  # eposide结束时回报为0
            policy.reward.append(reward)
            if done:
                break
            state = next_state

        # 更新策略
        update_policy()
        # save model
        torch.save(policy.state_dict(), checkpoints_path+'model.pth')
        print('Episode {}\tLast length: {:5d}'.format(
                episode, step))


def test(model_path=None):
    if model_path:
        policy.load_state_dict(torch.load(model_path))
    else:
        print("请给定模型参数文件")

    state = env.reset()
    rewards = []
    for step in range(num_step):
        env.render()

        action = select_action(state)
        next_state, reward, done, _ = env.step(action)
        rewards.append(reward)
        if done:
            break
        state = next_state

    plt.plot(rewards)
    plt.xlabel('step')
    plt.ylabel('reward')


if __name__ == '__main__':
    test(checkpoints_path+'model.pth')
    # train()