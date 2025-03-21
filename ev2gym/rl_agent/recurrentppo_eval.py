# -*- coding: utf-8 -*-
"""
RecurrentPPO Evaluation with Saved Model (Arrival/Departure Satisfaction Distribution)
"""
import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from sb3_contrib import RecurrentPPO  # Import Recurrent PPO

# Environment setup
sys.path.append("C:/Users/River/Desktop/EV2Gym-main/EV2Gym-main")
from ev2gym.rl_agent.reward import profit_maximization
from ev2gym.rl_agent.state import V2G_profit_max

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Environment setup
config_file = "../example_config_files/V2GProfitMax.yaml"
reward_function = profit_maximization
state_function = V2G_profit_max

env = gym.make(
    'EV2Gym-v1',
    config_file=config_file,
    # load_from_replay_path="replay/spring.pkl",
    save_replay=True,
    reward_function=reward_function,
    state_function=state_function
)

# Function to unwrap the environment
def unwrap_env(env):
    while hasattr(env, 'env'):
        env = env.env
    return env

unwrapped_env = unwrap_env(env)
if not hasattr(unwrapped_env, 'EVs'):
    raise AttributeError("The unwrapped environment does not have an attribute 'EVs'")

# Load the saved model
model_save_path = "../models/rppo_model (2).zip"
if not os.path.exists(model_save_path):
    raise FileNotFoundError(f"Model file not found at {model_save_path}")

try:
    custom_objects = {
        "observation_space": env.observation_space,
        "action_space": env.action_space,
    }
    model = RecurrentPPO.load(
        model_save_path,
        env=env,
        device=device,
        custom_objects=custom_objects
    )
    print(f"Model loaded successfully from {model_save_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# Evaluate the model
obs, info = env.reset()

# ------------------ Data containers ------------------
steps_list = []
cumulative_rewards = []
cumulative_costs = []
total_degradation_list = []
overload_list = []
price_list = []
action_list = []

# 新增兩個清單：分別紀錄「到達時刻」及「離站時刻」的使用者滿意度
arrival_satisfaction_list = []
departure_satisfaction_list = []

# New lists for tracking charged/discharged actions
charged_actions = []
discharged_actions = []

cumulative_reward = 0
cumulative_cost = 0
num_steps = 35040

for i in range(num_steps):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    # 取第 0 維動作 (若多維則只關心第一個)
    if isinstance(action, (list, tuple, np.ndarray)):
        action_val = action[0]
    else:
        action_val = action

    # 避免 step=0 時的 -1 索引
    current_step = max(0, env.current_step - 1)
    if 0 <= current_step < env.discharge_prices.shape[1]:
        price_val = env.discharge_prices[0, current_step]
    else:
        price_val = 0.0

    price_list.append(price_val)
    action_list.append(action_val)

    # 累計 reward / cost
    cumulative_reward += reward
    reward_contributions = info.get("reward_contributions", {})
    total_costs = reward_contributions.get("total_costs", 0)
    cumulative_cost += total_costs

    # 如果環境中有 EV
    if hasattr(env, 'EVs') and env.EVs:
        ev_degrade_sum = 0
        for ev in env.EVs:
            ev_degrade = getattr(ev, "total_degradation", 0)
            ev_degrade_sum += ev_degrade

            # 檢查「到達時刻」與「離站時刻」
            # 1) 如果 current_step == ev.time_of_arrival
            if env.current_step - 1 == ev.time_of_arrival:
                arrival_satisfaction_list.append(ev.get_user_satisfaction())

            # 2) 如果 current_step == ev.time_of_departure
            if env.current_step - 1 == ev.time_of_departure:
                departure_satisfaction_list.append(ev.get_user_satisfaction())
        
        total_degradation = ev_degrade_sum
    else:
        total_degradation = 0

    # 取得變壓器負載
    if hasattr(env, 'transformers') and env.transformers:
        overload = env.transformers[0].get_how_overloaded()
    else:
        overload = 0

    # 記錄其餘指標
    steps_list.append(i)
    cumulative_rewards.append(cumulative_reward)
    cumulative_costs.append(cumulative_cost)
    total_degradation_list.append(total_degradation)
    overload_list.append(overload)

    if done:
        obs, info = env.reset()  # Reset the environment

# --------------- Plot 1: Cumulative Reward vs Steps ---------------
plt.figure(figsize=(10, 6))
plt.plot(steps_list, cumulative_rewards, label="Cumulative Reward")
plt.xlabel("Step")
plt.ylabel("Cumulative Reward")
plt.title("Cumulative Reward Over Steps")
plt.legend()
plt.show()

# --------------- Plot 2: Cumulative Cost vs Steps ---------------
plt.figure(figsize=(10, 6))
plt.plot(steps_list, cumulative_costs, label="Cumulative Cost", color="orange")
plt.xlabel("Step")
plt.ylabel("Cumulative Cost")
plt.title("Cumulative Cost Over Steps")
plt.legend()
plt.show()

# --------------- Plot 3: Total Degradation vs Steps ---------------
plt.figure(figsize=(10, 6))
plt.plot(steps_list, total_degradation_list, label="Total Degradation", color="red")
plt.xlabel("Step")
plt.ylabel("Total Degradation")
plt.title("Total Degradation Over Steps")
plt.legend()
plt.show()

# --------------- Plot 4: Overload vs Steps ---------------
plt.figure(figsize=(10, 6))
plt.plot(steps_list, overload_list, label="Overload", color="purple")
plt.xlabel("Step")
plt.ylabel("Overload")
plt.title("Transformer Overload Over Steps")
plt.legend()
plt.show()

# --------------- Plot 5: Action vs Price (scatter) ---------------
plt.figure(figsize=(8,6))
plt.scatter(price_list, action_list, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Electricity Price")
plt.ylabel("Action (Charge/Discharge)")
plt.title("Action vs. Price")
plt.show()

# ------------------ Compute total charged and discharged energy ------------------
# Assume each step is 15 minutes, i.e. 0.25 hours
dt_hours = 15.0 / 60.0

total_charged_energy = sum(charged_actions) * dt_hours
total_discharged_energy = sum(discharged_actions) * dt_hours

print(f"Total Charged Energy (kWh): {total_charged_energy:.2f}")
print(f"Total Discharged Energy (kWh): {total_discharged_energy:.2f}")

# Plot bar chart for charged vs discharged energy
plt.figure(figsize=(8, 6))
plt.bar(["Charged", "Discharged"],
        [total_charged_energy, total_discharged_energy],
        color=["orange", "blue"])
plt.xlabel("Action Type")
plt.ylabel("Energy (kWh)")
plt.title("Total Charged vs. Discharged Energy")
plt.show()

# --------------- NEW: 2 Histograms for Arrival/Departure user satisfaction ---------------
plt.figure(figsize=(8,6))
plt.hist(arrival_satisfaction_list, bins=20, alpha=0.7, color='green')
plt.xlabel("User Satisfaction Score")
plt.ylabel("Frequency")
plt.title("Arrival Satisfaction Distribution")
plt.show()

plt.figure(figsize=(8,6))
plt.hist(departure_satisfaction_list, bins=20, alpha=0.7, color='blue')
plt.xlabel("User Satisfaction Score")
plt.ylabel("Frequency")
plt.title("Departure Satisfaction Distribution")
plt.show()
