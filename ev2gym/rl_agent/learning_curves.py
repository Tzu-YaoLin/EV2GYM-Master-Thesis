import sys
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import DDPG, PPO, A2C, TD3, SAC
from sb3_contrib import RecurrentPPO  # RPPO

# Append custom path (根據實際路徑調整)
sys.path.append("C:/Users/River/Desktop/EV2Gym-main/EV2Gym-main")
from ev2gym.rl_agent.reward import profit_maximization
from ev2gym.rl_agent.state import V2G_profit_max

# 環境設定
config_file = "../example_config_files/V2GProfitMax.yaml"
reward_function = profit_maximization
state_function = V2G_profit_max

env = gym.make('EV2Gym-v1',
               config_file=config_file,
               reward_function=reward_function,
               state_function=state_function)

# 定義演算法模型與對應檔案路徑 (6個演算法)
algo_dict = {
    "DDPG": (DDPG, "../models/ddpg_model (2).zip"),
    "PPO": (PPO, "../models/ppo_model (2).zip"),
    "A2C": (A2C, "../models/a2c_model (2).zip"),
    "TD3": (TD3, "../models/td3_model (2).zip"),
    "SAC": (SAC, "../models/sac_model (2).zip"),
    "RPPO": (RecurrentPPO, "../models/rppo_model (2).zip"),
}

device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
eval_episodes = 10  # 設定每個模型評估的 episode 數
evaluation_results = {}  # 用來存放每個演算法的每個 episode reward

for algo_name, (AlgoClass, model_path) in algo_dict.items():
    print(f"Evaluating {algo_name} model from {model_path}...")
    # 載入模型並設置環境
    model = AlgoClass.load(model_path, device=device)
    model.set_env(env)
    
    episode_rewards = []
    for ep in range(eval_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        episode_rewards.append(ep_reward)
        print(f"{algo_name} - Episode {ep+1} reward: {ep_reward}")
    
    evaluation_results[algo_name] = episode_rewards

# 繪製學習曲線 (每個 episode 的 reward)
plt.figure(figsize=(10, 6))
for algo_name, rewards in evaluation_results.items():
    plt.plot(range(1, eval_episodes+1), rewards, marker='o', label=algo_name)
plt.xlabel("Episode")
plt.ylabel("Episode Reward")
plt.title("Evaluation Learning Curves for Different Algorithms")
plt.legend()
plt.show()
