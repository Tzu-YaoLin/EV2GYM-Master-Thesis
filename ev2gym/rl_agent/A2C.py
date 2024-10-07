# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 19:32:56 2024

@author: River
"""
import numpy as np
from ev2gym.models.ev2gym_env import EV2Gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import math
from collections import namedtuple, deque
from itertools import count
import matplotlib.pyplot as plt
import pandas as pd
from state import V2G_profit_max
from reward import profit_maximization  # 確保 v2g_profit_reward 定義在 reward.py 中

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 超參數
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 0.005
LR = 1e-4  # 學習率

# 使用绝对路径配置文件
config_file = "C:\\Users\\River\\Desktop\\EV2Gym-main\\EV2Gym-main\\ev2gym\\example_config_files\\V2GProfitMax.yaml"

# 创建环境
env = EV2Gym(config_file,
             render_mode=True,
             seed=42,
             save_plots=False,
             save_replay=False)

# 設定狀態函數為 V2G_profit_max
env.state_function = V2G_profit_max

# 設定新的獎勵函數為 v2g_profit_reward
env.set_reward_function(profit_maximization)

# 計算所有充電站的總埠數並設置 n_actions
total_ports = sum(cs.n_ports for cs in env.charging_stations)
n_actions = total_ports  # 確保 n_actions 與總埠數相同

# Get the initial state
env.reset()

# Actor 網絡 (Policy)
class Actor(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.action_head = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return F.softmax(self.action_head(x), dim=-1)  # 輸出動作的概率分佈

# Critic 網絡 (Value function)
class Critic(nn.Module):
    def __init__(self, n_observations):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.value_head(x)  # 輸出狀態的價值

# 定義 Transition
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# Initialize networks
state, _ = env.reset()
n_observations = len(state)

actor = Actor(n_observations, n_actions).to(device)

critic = Critic(n_observations).to(device)

optimizer_actor = optim.Adam(actor.parameters(), lr=LR)
optimizer_critic = optim.Adam(critic.parameters(), lr=LR)


# Select action with noise
# 修改 select_action 函數
def select_action(state):
    state = state.unsqueeze(0)  # 保證 state 是二維的
    action_probs = actor(state)  # 獲取動作的概率分佈
    action_distribution = torch.distributions.Categorical(action_probs)
    
    # 生成一個對應於每個充電站端口的動作列表
    actions = [action_distribution.sample().item() for _ in range(n_actions)]  # n_actions 應該等於充電站端口數
    return actions





def optimize_model(state, action, reward, next_state):
    state = state.clone().detach().float().to(device) if isinstance(state, torch.Tensor) else torch.tensor(state, dtype=torch.float32, device=device)
    next_state = next_state.clone().detach().float().to(device) if next_state is not None and isinstance(next_state, torch.Tensor) else None
    reward = torch.tensor([reward], dtype=torch.float32, device=device)

    # Critic 輸出
    value = critic(state).view(-1)
    with torch.no_grad():
        next_value = critic(next_state).view(-1) if next_state is not None else torch.tensor(0.0, device=device)

    # 計算 Advantage
    advantage = reward + GAMMA * next_value - value

    # 更新 Actor
    action_probs = actor(state)
    log_prob = torch.log(action_probs.squeeze(0)[action])
    actor_loss = -(log_prob * advantage.detach()).mean()  # 修正 actor_loss 為 mean()

    optimizer_actor.zero_grad()
    actor_loss.backward()
    optimizer_actor.step()

    # 更新 Critic
    critic_loss = F.mse_loss(value, (reward + GAMMA * next_value).detach())
    optimizer_critic.zero_grad()
    critic_loss.backward()
    optimizer_critic.step()

episode_rewards = []
episode_stats = []

if torch.cuda.is_available():
    num_episodes = 200
else:
    num_episodes = 50

for i_episode in range(num_episodes):
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    episode_reward = 0

    for t in count():
        action = select_action(state)

        observation, reward, done, truncated, stats = env.step(action)
        episode_reward += reward

        next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0) if not done else None

        # 將當前狀態、動作、獎勵和下一狀態傳遞給 optimize_model 函數
        optimize_model(state, action, reward, next_state)

        state = next_state

        if done or truncated:
            # 在每個回合結束時保存 episode_reward 到列表中
            episode_rewards.append(episode_reward)
            print(f'Episode {i_episode} finished after {t + 1} timesteps, total reward: {episode_reward:.2f}')
            break


# Save results
results = pd.DataFrame(episode_rewards, columns=["Episode Reward"])
results.to_csv("training_results.csv", index=False)

# 繪製訓練結果
plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Training Results')
plt.show()
