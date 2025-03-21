# -*- coding: utf-8 -*-
"""
Evaluate six different RL algorithms (DDPG, TD3, SAC, A2C, PPO, RPPO)
on the same environment, each for 35040 steps, and print the final reward.
"""

import sys
import os
import torch
import numpy as np
import gymnasium as gym

# Import SB3 or SB3-contrib classes
from stable_baselines3 import DDPG, TD3, SAC, A2C, PPO
from sb3_contrib import RecurrentPPO

# Environment setup
sys.path.append("C:/Users/River/Desktop/EV2Gym-main/EV2GYM-main")
from ev2gym.rl_agent.reward import profit_maximization
from ev2gym.rl_agent.state import V2G_profit_max

# List of algorithms to evaluate
algo_list = ["ddpg", "td3", "sac", "a2c", "ppo", "rppo"]

# Dictionary mapping each algo to its model class (for stable_baselines3 or sb3_contrib)
model_classes = {
    "ddpg":  DDPG,
    "td3":   TD3,
    "sac":   SAC,
    "a2c":   A2C,
    "ppo":   PPO,
    "rppo":  RecurrentPPO,   # from sb3_contrib
}

# Path template for saved models, e.g., "../models/ddpg_model_1.zip"
model_path_template = "../models/{}_model_1.zip"

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Number of simulation steps for each run
num_steps = 35040

# Common environment configuration
config_file = "../example_config_files/V2GProfitMax.yaml"
reward_function = profit_maximization
state_function = V2G_profit_max

def evaluate_model(algo_name: str, steps: int = 35040) -> float:
    """
    Creates the environment, loads the model for 'algo_name', runs it for 'steps',
    and returns the final cumulative reward.
    """
    # Create the environment
    env = gym.make(
        'EV2Gym-v1',
        config_file=config_file,
        save_replay=False,
        reward_function=reward_function,
        state_function=state_function
    )
    
    # Build the model path and check existence
    model_save_path = model_path_template.format(algo_name)
    if not os.path.exists(model_save_path):
        raise FileNotFoundError(f"Model file not found at {model_save_path}")

    # Load the model
    model_class = model_classes[algo_name]
    try:
        custom_objects = {
            "observation_space": env.observation_space,
            "action_space": env.action_space,
        }
        model = model_class.load(
            model_save_path,
            env=env,
            device=device,
            custom_objects=custom_objects
        )
        print(f"[{algo_name.upper()}] Model loaded successfully from {model_save_path}")
    except Exception as e:
        print(f"Error loading {algo_name} model: {e}")
        raise

    # Evaluate the model
    obs, info = env.reset()
    cumulative_reward = 0.0

    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        cumulative_reward += reward

        if terminated or truncated:
            obs, info = env.reset()

    return cumulative_reward

# Main script: run all algorithms and print final rewards
if __name__ == "__main__":
    for algo in algo_list:
        final_reward = evaluate_model(algo, num_steps)
        print(f"Algorithm: {algo.upper()} | Final Total Reward after {num_steps} steps: {final_reward:.2f}")
