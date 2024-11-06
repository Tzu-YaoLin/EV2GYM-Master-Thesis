# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 16:26:37 2024

@author: River
"""

from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch  # 添加這一行來導入 torch
from ev2gym.models.ev2gym_env import EV2Gym
from state import V2G_profit_max
from reward import profit_maximization

# 設置裝置
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 加載家庭負載數據
household_loads_df = pd.read_csv(
    '../data/standardlastprofil-haushalte-2023.csv', 
    sep=',', 
    engine='python'
)
household_loads = household_loads_df['SLP-Bandlastkunden HB [kWh]'].values

# 讀取電價資料
electricity_prices_df = pd.read_csv(
    '../data/Day-ahead_prices_202301010000_202401010000_Quarterhour_processed.csv', 
    sep=',', 
    engine='python'
)
electricity_prices = electricity_prices_df['Germany/Luxembourg [€/MWh] Calculated resolutions'].values
electricity_prices = np.nan_to_num(electricity_prices, nan=0.0)  # 檢查並替換 NaN 值

# 創建自定義環境
class CustomEV2Gym(EV2Gym):
    def __init__(self, config_file, household_loads, electricity_prices, *args, **kwargs):
        super().__init__(config_file, *args, **kwargs)
        self.household_loads = household_loads
        self.electricity_prices = electricity_prices
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(28,), dtype=np.float32)

config_file = "../example_config_files/V2GProfitMax.yaml"
env = CustomEV2Gym(config_file, household_loads, electricity_prices, render_mode=True)

# 設置狀態函數和獎勵函數
low_price_threshold = np.percentile(electricity_prices, 25)
high_price_threshold = np.percentile(electricity_prices, 75)

env.state_function = lambda *args: V2G_profit_max(env, *args)
env.set_reward_function(lambda env_instance, total_costs, user_satisfaction_list, *args: 
    profit_maximization(env_instance, total_costs, user_satisfaction_list, low_price_threshold, high_price_threshold))

vec_env = DummyVecEnv([lambda: env])  # 使用 DummyVecEnv 包裝環境

# 定義超參數
BATCH_SIZE = 64
GAMMA = 0.95
TAU = 0.005
LR = 1e-3  # 學習率

# 定義演算法
algorithms = {
    "A2C": A2C("MlpPolicy", vec_env, learning_rate=LR, gamma=GAMMA, verbose=1),
    "DDPG": DDPG("MlpPolicy", vec_env, learning_rate=LR, batch_size=BATCH_SIZE, gamma=GAMMA, verbose=1),
    "PPO": PPO("MlpPolicy", vec_env, learning_rate=LR, gamma=GAMMA, verbose=1),
    "SAC": SAC("MlpPolicy", vec_env, learning_rate=LR, batch_size=BATCH_SIZE, gamma=GAMMA, verbose=1),
    "TD3": TD3("MlpPolicy", vec_env, learning_rate=LR, batch_size=BATCH_SIZE, gamma=GAMMA, verbose=1)
}

# 訓練並測試每個演算法
results = {}
total_timesteps = 100000  # 訓練步數
for algo_name, model in algorithms.items():
    print(f"Training {algo_name}...")
    model.learn(total_timesteps=total_timesteps)
    
    # 測試模型
    obs = vec_env.reset()
    episode_rewards = []
    for _ in range(50):  # 測試 50 個 episode
        obs = vec_env.reset()
        episode_reward = 0
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            if isinstance(reward, torch.Tensor):
                reward = reward.item()  # 將 Tensor 轉為數值
            if isinstance(obs, torch.Tensor):
                obs = obs.numpy()
            episode_reward += reward
        episode_rewards.append(episode_reward)
    
    # 儲存結果
    results[algo_name] = episode_rewards

# 繪製結果比較圖
plt.figure(figsize=(10, 6))
for algo_name, rewards in results.items():
    plt.plot(rewards, label=algo_name)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Comparison of RL Algorithms')
plt.legend()
plt.show()

# 繪製移動平均圖
window = 10
plt.figure(figsize=(10, 6))
for algo_name, rewards in results.items():
    moving_avg_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
    plt.plot(moving_avg_rewards, label=algo_name)
plt.xlabel('Episode')
plt.ylabel('Moving Average Reward')
plt.title('Moving Average of Total Reward (Window Size 10)')
plt.legend()
plt.show()
