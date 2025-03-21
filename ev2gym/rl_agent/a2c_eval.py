# -*- coding: utf-8 -*-
"""
A2C Evaluation with Saved Model
"""

# Install required packages
%pip install wandb stable-baselines3 gymnasium matplotlib torch

import numpy as np
import wandb
import os
import torch
import gymnasium as gym
from stable_baselines3 import A2C  # Import A2C
from ev2gym.rl_agent.reward import profit_maximization
from ev2gym.rl_agent.state import V2G_profit_max
from ev2gym.models.ev2gym_env import EV2Gym

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# WandB Initialization
wandb.login(key='e09f1372cfe6ed2cd13773350fe920cb084c6341')
wandb.init(project="ev2gym_a2c_evaluation")

# Environment setup
config_file = "../example_config_files/V2GProfitMax.yaml"
reward_function = profit_maximization
state_function = V2G_profit_max

env = gym.make('EV2Gym-v1',
               config_file=config_file,
               reward_function=reward_function,
               state_function=state_function,
               save_plots=True  # Enable saving plots
               )

# Function to unwrap the environment
def unwrap_env(env):
    while hasattr(env, 'env'):
        env = env.env
    return env

# Unwrap the environment to access custom attributes
unwrapped_env = unwrap_env(env)

# Access the EVs attribute
if hasattr(unwrapped_env, 'EVs'):
    EVs = unwrapped_env.EVs
else:
    raise AttributeError("The unwrapped environment does not have an attribute 'EVs'")

# Load the saved model
model_save_path = "../models/a2c_model.zip"

# Load the saved model with custom objects
if os.path.exists(model_save_path):
    try:
        # Replace action_space and observation_space with the current environment's spaces
        custom_objects = {
            "observation_space": env.observation_space,
            "action_space": env.action_space,
        }
        model = A2C.load(model_save_path, env=env, device=device, custom_objects=custom_objects)
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
for i in range(35040):
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