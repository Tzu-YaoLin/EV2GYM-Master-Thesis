# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 16:33:02 2024

@author: River
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import wandb  # Import WandB

# Environment setup
sys.path.append("C:/Users/River/Desktop/EV2Gym-main/EV2Gym-main")
import gymnasium as gym
from stable_baselines3 import DDPG
from ev2gym.rl_agent.reward import profit_maximization, ProfitMax_TrPenalty_UserIncentives
from ev2gym.rl_agent.state import V2G_profit_max
from ev2gym.visuals.plots import ev_city_plot, visualize_step
from gymnasium.wrappers.monitoring import video_recorder
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

import warnings
warnings.filterwarnings('ignore')

# Initialize WandB for hyperparameter tuning with DDPG 的超參數
wandb.init(project="ev2gym_ddpg_tuning", config={
    "policy_type": "MlpPolicy",
    "total_timesteps": 105120,
    "buffer_size": 35040,
    "learning_rate": 0.000673250381088552,
    "batch_size": 8760,
    "gamma": 0.9912641475395462,
    "gradient_steps": 5,
    "noise_std": 0.3150978088574419,
    "tau": 0.031774730414353455,
    "theta": 0.18937426912457295,
    "train_freq": 48,
})
config = wandb.config

# Define reward and state function
reward_function = profit_maximization
state_function = V2G_profit_max

# Path to the config file
config_file = "../example_config_files/V2GProfitMax.yaml"

# Initialize the environment
env = gym.make('EV2Gym-v1',
               config_file=config_file,
               reward_function=reward_function,
               state_function=state_function)

# 定義 DDPG 需要的 action noise
n_actions = env.action_space.shape[-1]
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions),
                                            sigma=config.noise_std * np.ones(n_actions),
                                            theta=config.theta)

# 建立 DDPG 模型，更新參數為 DDPG 的超參數
model = DDPG(
    config.policy_type,
    env,
    learning_rate=config.learning_rate,
    buffer_size=config.buffer_size,
    batch_size=config.batch_size,
    tau=config.tau,
    gamma=config.gamma,
    train_freq=config.train_freq,
    gradient_steps=config.gradient_steps,
    action_noise=action_noise,
    verbose=1
)

# 定義提前停止的 callback（可選）
def early_stopping_callback(_locals, _globals):
    reward_threshold = 100  # 設置提前停止的閾值
    if 'rewards' in _locals and len(_locals['rewards']) > 0 and _locals['rewards'][-1] > reward_threshold:
        print("Stopping early: Reward threshold reached")
        return False
    return True

# Train the model
model.learn(total_timesteps=config.total_timesteps, progress_bar=True, callback=early_stopping_callback)

# Evaluate the model
obs, info = env.reset()  # `reset` returns (obs, info)
rewards = []             # 儲存每一步的 reward
actions = []             # 儲存每一步的 action
cumulative_rewards = []  # 儲存累計 reward

# 記錄各 reward component 的貢獻
reward_details = {
    "total_reward": [],
    "degradation_reward": [],
    "user_satisfaction_reward": [],
    "transformer_load_reward": [],
    "total_costs": [],
    "soc_reward": [],
}

# Evaluation loop
cumulative_reward = 0
for i in range(35040):
    # 從模型獲取 action
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    rewards.append(reward)
    actions.append(action)
    cumulative_reward += reward
    cumulative_rewards.append(cumulative_reward)

    # 擷取 info 中 reward 的各個部分
    reward_contributions = info["reward_contributions"]
    reward_details["total_reward"].append(reward)
    reward_details["degradation_reward"].append(reward_contributions.get("degradation_reward", 0))
    reward_details["user_satisfaction_reward"].append(reward_contributions.get("user_satisfaction_reward", 0))
    reward_details["transformer_load_reward"].append(reward_contributions.get("transformer_load_reward", 0))
    reward_details["total_costs"].append(reward_contributions.get("total_costs", 0))
    reward_details["soc_reward"].append(reward_contributions.get("soc_reward", 0))

    # Log 資料到 WandB
    wandb.log({
        "step": i,
        "reward": reward,
        "cumulative_reward": cumulative_reward,
        "action": action,
        "degradation_reward": reward_contributions.get("degradation_reward", 0),
        "user_satisfaction_reward": reward_contributions.get("user_satisfaction_reward", 0),
        "transformer_load_reward": reward_contributions.get("transformer_load_reward", 0),
        "total_costs": reward_contributions.get("total_costs", 0),
        "soc_reward": reward_contributions.get("soc_reward", 0)
    })

    if done:
        obs, info = env.reset()

# 結束 WandB run
wandb.finish()
