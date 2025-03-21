# -*- coding: utf-8 -*-
"""
<<<<<<< HEAD
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
=======
RecurrentPPO Evaluation with Saved Model
"""

# Install required packages
%pip install wandb stable-baselines3 gymnasium matplotlib torch sb3-contrib

import numpy as np
import wandb
import os
import torch
import gymnasium as gym
from sb3_contrib import RecurrentPPO  # Import Recurrent PPO
>>>>>>> da436234dff60d7313809a1f10cd28ef6f685df1
from ev2gym.rl_agent.reward import profit_maximization
from ev2gym.rl_agent.state import V2G_profit_max

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

<<<<<<< HEAD
=======
# WandB Initialization
wandb.login(key='e09f1372cfe6ed2cd13773350fe920cb084c6341')
wandb.init(project="ev2gym_recurrentppo_evaluation")

>>>>>>> da436234dff60d7313809a1f10cd28ef6f685df1
# Environment setup
config_file = "../example_config_files/V2GProfitMax.yaml"
reward_function = profit_maximization
state_function = V2G_profit_max

<<<<<<< HEAD
env = gym.make(
    'EV2Gym-v1',
    config_file=config_file,
    # load_from_replay_path="replay/spring.pkl",
    save_replay=True,
    reward_function=reward_function,
    state_function=state_function
)
=======
env = gym.make('EV2Gym-v1',
               config_file=config_file,
               load_from_replay_path="replay/autumn.pkl",
               reward_function=reward_function,
               state_function=state_function)
>>>>>>> da436234dff60d7313809a1f10cd28ef6f685df1

# Function to unwrap the environment
def unwrap_env(env):
    while hasattr(env, 'env'):
        env = env.env
    return env

<<<<<<< HEAD
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
=======
# Unwrap the environment to access custom attributes
unwrapped_env = unwrap_env(env)

# Access the EVs attribute
if hasattr(unwrapped_env, 'EVs'):
    EVs = unwrapped_env.EVs
else:
    raise AttributeError("The unwrapped environment does not have an attribute 'EVs'")

# Load the saved model
model_save_path = "../models/recurrentppo_model.zip"

# Load the saved model with custom objects
if os.path.exists(model_save_path):
    try:
        # Replace action_space and observation_space with the current environment's spaces
        custom_objects = {
            "observation_space": env.observation_space,
            "action_space": env.action_space,
        }
        model = RecurrentPPO.load(model_save_path, env=env, device=device, custom_objects=custom_objects)
        print(f"Model loaded successfully from {model_save_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
else:
    raise FileNotFoundError(f"Model file not found at {model_save_path}")

# Evaluate the model
obs, info = env.reset()  # `reset` returns (obs, info)
rewards = []  # Used to store rewards at each step
actions = []  # Used to store actions at each step
cumulative_rewards = []  # Used to store cumulative rewards
cumulative_costs = []  # Used to store cumulative costs

# Track SOC for averaging
average_soc = []

# Record contributions of each reward component
reward_details = {
    "total_reward": [],
    "degradation_reward": [],
    #"charge_price_reward": [],
    #"discharge_price_reward": [],
    "user_satisfaction_reward": [],
    "transformer_load_reward": [],
    #"invalid_action_penalty": [],
    "total_costs": [],  # To track costs separately
    "soc_reward": [],
}

# Evaluation loop
cumulative_reward = 0
cumulative_cost = 0
for i in range(1000):
    # Unpack step() return values
    action, _states = model.predict(obs, deterministic=True)  # Pass pure observation obs
    obs, reward, terminated, truncated, info = env.step(action)  # Unpack 5 return values
    done = terminated or truncated  # Combine terminated and truncated

    # Record total reward and action
    rewards.append(reward)
    actions.append(action)
    cumulative_reward += reward
    cumulative_rewards.append(cumulative_reward)

    # Compute average SOC for active EVs
    soc_values = [
        ev.historic_soc[-1] for ev in unwrapped_env.EVs 
        if hasattr(ev, 'historic_soc') and len(ev.historic_soc) > 0
    ]
    avg_soc = np.mean(soc_values) if soc_values else 0
    average_soc.append(avg_soc)

    # Extract reward contributions from info
    reward_contributions = info.get("reward_contributions", {})
    total_costs = reward_contributions.get("total_costs", 0)  # Extract total cost
    cumulative_cost += total_costs  # Update cumulative cost
    cumulative_costs.append(cumulative_cost)  # Record cumulative cost

    # Record each reward component in the dictionary
    reward_details["total_reward"].append(reward)
    reward_details["degradation_reward"].append(reward_contributions.get("degradation_reward", 0))
    reward_details["user_satisfaction_reward"].append(reward_contributions.get("user_satisfaction_reward", 0))
    reward_details["transformer_load_reward"].append(reward_contributions.get("transformer_load_reward", 0))
    #reward_details["invalid_action_penalty"].append(reward_contributions.get("invalid_action_penalty", 0))
    reward_details["total_costs"].append(reward_contributions.get("total_costs", 0))  # Log total cost
    reward_details["soc_reward"].append(reward_contributions.get("soc_reward", 0))

    # Record charging or discharging reward based on action
    #if action > 0:  # If charging action
    #    reward_details["charge_price_reward"].append(reward_contributions.get("charge_price_reward", 0))
    #    reward_details["discharge_price_reward"].append(0)  # Discharging part is 0
    #elif action < 0:  # If discharging action
    #    reward_details["charge_price_reward"].append(0)  # Charging part is 0
    #    reward_details["discharge_price_reward"].append(reward_contributions.get("discharge_price_reward", 0))
    #else:
    #    # If no charging or discharging action
    #   reward_details["charge_price_reward"].append(0)
    #    reward_details["discharge_price_reward"].append(0)

    # Log to WandB
    wandb.log({
        "step": i,
        "reward": reward,
        "cumulative_reward": cumulative_reward,
        "action": action,
        "average_soc": avg_soc,
        "degradation_reward": reward_contributions.get("degradation_reward", 0),
        "user_satisfaction_reward": reward_contributions.get("user_satisfaction_reward", 0),
        "transformer_load_reward": reward_contributions.get("transformer_load_reward", 0),
    #    "charge_price_reward": reward_contributions.get("charge_price_reward", 0),
    #    "discharge_price_reward": reward_contributions.get("discharge_price_reward", 0),
    #    "invalid_action_penalty": reward_contributions.get("invalid_action_penalty", 0),
        "total_costs": reward_contributions.get("total_costs", 0),
        "cumulative_cost": cumulative_cost,  # Log cumulative cost
        "soc_reward": reward_contributions.get("soc_reward", 0)
    })

    if done:
        obs, info = env.reset()  # Reset the environment and unpack return values
        cumulative_reward = 0  # Reset cumulative reward
        cumulative_cost = 0

# Finish WandB run
wandb.finish()
>>>>>>> da436234dff60d7313809a1f10cd28ef6f685df1
