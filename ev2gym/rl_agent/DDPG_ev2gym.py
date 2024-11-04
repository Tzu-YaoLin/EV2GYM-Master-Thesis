# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 18:07:54 2024

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
GAMMA = 0.95
TAU = 0.005
LR = 1e-3  # 學習率

# 使用绝对路径配置文件
config_file = "C:\\Users\\River\\Desktop\\EV2Gym-main\\EV2Gym-main\\ev2gym\\example_config_files\\V2GProfitMax.yaml"

# 加載家庭負載數據
household_loads_df = pd.read_csv('C:\\Users\\River\\Desktop\\EV2Gym-main\\EV2Gym-main\\ev2gym\\data\\standardlastprofil-haushalte-2023.csv', sep=',', engine='python')
household_loads = household_loads_df['SLP-Bandlastkunden HB [kWh]'].values

# 讀取電價資料 (假設您只需要 'Germany/Luxembourg [€/MWh]' 列的電價)
electricity_prices_df = pd.read_csv('C:/Users/River/Desktop/EV2Gym-main/EV2Gym-main/ev2gym/data/Day-ahead_prices_202301010000_202401010000_Quarterhour_processed.csv', sep=',', engine='python')

# 提取電價列並將其轉換為 numpy 陣列
electricity_prices = electricity_prices_df['Germany/Luxembourg [€/MWh] Calculated resolutions'].values

# 檢查並替換 NaN 值
electricity_prices = np.nan_to_num(electricity_prices, nan=0.0)

# 创建环境
env = EV2Gym(config_file,
             render_mode=True,
             seed=None,
             save_plots=False,
             save_replay=False)

# 將家庭負載數據傳遞給環境
env.household_loads = household_loads

# 加入電價資料到環境中
env.electricity_prices = electricity_prices

# 計算低電價和高電價的閾值
low_price_threshold = np.percentile(electricity_prices, 25)  # 第25百分位數
high_price_threshold = np.percentile(electricity_prices, 75)  # 第75百分位數


# 設定狀態函數為 V2G_profit_max，不再傳遞均值和標準差
env.state_function = lambda *args: V2G_profit_max(env, *args)


# 設定新的獎勵函數為 profit_maximization
env.set_reward_function(lambda env_instance, total_costs, user_satisfaction_list, *args: 
    profit_maximization(env_instance, total_costs, user_satisfaction_list, low_price_threshold, high_price_threshold))

# 計算所有充電站的總埠數並設置 n_actions
total_ports = sum(cs.n_ports for cs in env.charging_stations)
n_actions = total_ports  # 確保 n_actions 與總埠數相同

# Get the initial state
env.reset()
# Reset the environment and get the initial state
state, _ = env.reset()
print(f"Initial state: {state}")


# Actor network
class Actor(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        output = torch.tanh(self.layer3(x))  # 使用 Tanh 激活函數
        output = output.squeeze(0)  # 去掉多餘的維度
        #print(f"Actor output shape: {output.shape}")  # 打印形狀
        return output
class Critic(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(n_observations + n_actions, 256)  # 增加單元數
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 128)
        self.output_layer = nn.Linear(128, 1)  # 增加一層輸出

    def forward(self, state, action):
        if action.dim() == 1:
            action = action.unsqueeze(1)  # 確保 action 是 2D 張量

        x = torch.cat([state, action], dim=1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.output_layer(x)

# 定義 Transition
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Initialize networks
state, _ = env.reset()
n_observations = len(state)

actor = Actor(n_observations, n_actions).to(device)
target_actor = Actor(n_observations, n_actions).to(device)
target_actor.load_state_dict(actor.state_dict())

critic = Critic(n_observations, n_actions).to(device)
target_critic = Critic(n_observations, n_actions).to(device)
target_critic.load_state_dict(critic.state_dict())

optimizer_actor = optim.Adam(actor.parameters(), lr=LR)
optimizer_critic = optim.Adam(critic.parameters(), lr=LR)
memory = ReplayMemory(10000)


steps_done = 0

# 初始化 epsilon 參數
epsilon = 1.0  # 初始探索概率
epsilon_min = 0.01  # 最小 epsilon 值
epsilon_decay = 0.995  # 每次選擇行動後的 epsilon 衰減比例

# 修改 select_action 函數
def select_action(state):
    global epsilon  # 引入 epsilon 變量
    state = state.unsqueeze(0)  # 保證 state 是二維的
    
    # 使用 epsilon-greedy 策略選擇行動
    if np.random.rand() < epsilon:
        # 探索：隨機選擇動作
        action = np.random.uniform(-1, 1, n_actions)  # 隨機選擇一個在 [-1, 1] 範圍內的動作
    else:
        # 利用：從 Actor 網絡獲取當前最佳動作
        action = actor(state).cpu().data.numpy()
        action = action.squeeze(0)  # 去掉多餘的維度

    action = np.clip(action, -1, 1)  # 確保動作在 [-1, 1] 範圍內

    # 每次選擇行動後，逐漸減少 epsilon 的值
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    return action

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # 過濾掉 None 的 next_state，並調整其他批次數據
    valid_indices = [i for i, s in enumerate(batch.next_state) if s is not None]
    state_batch = torch.cat([batch.state[i].float() for i in valid_indices])
    next_state_batch = torch.cat([batch.next_state[i].float() for i in valid_indices if batch.next_state[i] is not None])

    # 確保 action 是 Tensor 格式，並保留二維結構
    action_batch = torch.stack([torch.tensor(batch.action[i], dtype=torch.float32) if isinstance(batch.action[i], np.ndarray) else batch.action[i].float() for i in valid_indices])
    
    reward_batch = torch.cat([batch.reward[i].clone().detach() for i in valid_indices])
    
    # 檢查是否有 nan 值
    if torch.isnan(state_batch).any() or torch.isnan(action_batch).any():
        #print(f"Invalid values in state or action batch during optimization at step {t} in episode {i_episode}")
        #print(f"State batch range: {state_batch.min().item()} to {state_batch.max().item()}")
        #print(f"Action batch range: {action_batch.min().item()} to {action_batch.max().item()}")
        return


    # 檢查拼接後的維度
    #print(f"State batch shape: {state_batch.shape}")
    #print(f"Action batch shape: {action_batch.shape}")
    #print(f"Next state batch shape: {next_state_batch.shape}")

    # Update Critic
    with torch.no_grad():
        next_actions = target_actor(next_state_batch)
        next_state_values = target_critic(next_state_batch, next_actions).view(-1, 1)  # 確保形狀正確
        expected_state_action_values = (reward_batch.view(-1, 1) + GAMMA * next_state_values).float()  # 確保類型正確

    state_action_values = critic(state_batch, action_batch)
    critic_loss = F.mse_loss(state_action_values, expected_state_action_values)

    optimizer_critic.zero_grad()
    critic_loss.backward()
    # 限制 Critic 梯度的大小
    torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
    optimizer_critic.step()

    # Update Actor
    actor_loss = -critic(state_batch, actor(state_batch)).mean()
    optimizer_actor.zero_grad()
    actor_loss.backward()
    # 限制 Actor 梯度的大小
    torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
    optimizer_actor.step()
    
    #print(f'State-Action values shape: {state_action_values.shape}')
    #print(f'Expected State-Action values shape: {expected_state_action_values.shape}')
    #print(f"State batch range: {state_batch.min().item()} to {state_batch.max().item()}")
    #print(f"Action batch range: {action_batch.min().item()} to {action_batch.max().item()}")
    #print(f"Episode {i_episode}, Step {t}, Actor Loss: {actor_loss.item()}, Critic Loss: {critic_loss.item()}")

    # Soft update of target networks
    for target_param, param in zip(target_critic.parameters(), critic.parameters()):
        target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)

    for target_param, param in zip(target_actor.parameters(), actor.parameters()):
        target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)

episode_rewards = []
episode_stats = []

if torch.cuda.is_available():
    num_episodes = 200
else:
    num_episodes = 50

for i_episode in range(num_episodes):
    # Initialize the environment and get its state
    state, _ = env.reset()
    if torch.isnan(torch.tensor(state)).any():
        print("Invalid state after reset:", state)
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    
    episode_reward = 0
    for t in count():
        action = select_action(state)  # 使用 Epsilon-Greedy 選擇行動

        # 檢查 action 的大小
        if len(action) != n_actions:
            raise ValueError(f"Expected action length {n_actions}, but got {len(action)}.")

        observation, reward, done, truncated, stats = env.step(action)
        episode_reward += reward
        reward = torch.tensor([reward], dtype=torch.float32, device=device)
        
        # 檢查 reward 是否為 NaN 或無限大
        if math.isnan(reward.item()) or math.isinf(reward.item()):
            print(f"Invalid reward detected: {reward.item()} at step {t} in episode {i_episode}")
        
        if done:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        memory.push(state, action, next_state, reward)
        state = next_state

        optimize_model()  # 優化模型
        
        # 結束 episode 時記錄
        if done or truncated:
            episode_rewards.append(episode_reward)
            episode_stats.append(stats)
            print(f'Episode {i_episode} finished after {t + 1} timesteps, total reward: {episode_reward:.2f}')
            print(f'Epsilon after episode {i_episode}: {epsilon:.4f}')  # 輸出當前的 epsilon 值
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
