%python
# -*- coding: utf-8 -*-
"""
A2C Training with Best Hyperparameters
"""

# Install required packages
%pip install wandb stable-baselines3 gymnasium matplotlib torch

import numpy as np
import wandb
import os
import torch
import gymnasium as gym
from stable_baselines3 import A2C
from ev2gym.rl_agent.reward import profit_maximization
from ev2gym.rl_agent.state import V2G_profit_max

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Login to WandB
wandb.login(key='e09f1372cfe6ed2cd13773350fe920cb084c6341')

# Initialize WandB with A2C best hyperparameters
wandb.init(project="ev2gym_a2c_training", config={
    "ent_coef": 0.008456324831647006,
    "gae_lambda": 0.9519341773675116,
    "gamma": 0.9783118600694408,
    "learning_rate": 0.0009174222905013246,
    "max_grad_norm": 3.2486456116327564,
    "n_steps": 128,
    "policy_type": "MlpPolicy",
    "total_timesteps": 2102400,
    "vf_coef": 0.8013127892335519
})
config = wandb.config

# Environment setup
config_file = "../example_config_files/V2GProfitMax.yaml"
reward_function = profit_maximization
state_function = V2G_profit_max

env = gym.make('EV2Gym-v1',
               config_file=config_file,
               reward_function=reward_function,
               state_function=state_function)

# Initialize A2C model with best hyperparameters
model = A2C(
    policy=config.policy_type,
    env=env,
    learning_rate=config.learning_rate,
    gamma=config.gamma,
    gae_lambda=config.gae_lambda,
    ent_coef=config.ent_coef,
    vf_coef=config.vf_coef,
    max_grad_norm=config.max_grad_norm,
    n_steps=config.n_steps,
    verbose=1,
    device=device
)

# Define Early Stopping Callback
class EarlyStoppingCallback:
    def __init__(self, patience: int = 10000, delta: float = 1e-3):
        """
        Early Stopping parameters.
        :param patience: Number of steps to wait for improvement.
        :param delta: Minimum improvement threshold.
        """
        self.patience = patience
        self.delta = delta
        self.best_mean_reward = -np.inf
        self.no_improvement_steps = 0

    def __call__(self, _locals, _globals):
        """
        Callback logic for monitoring mean reward.
        """
        self.model = _locals['self']
        timesteps = self.model.num_timesteps
        rewards = getattr(self.model, "episode_rewards", [])

        # Calculate mean reward
        if len(rewards) >= 5:
            mean_reward = np.mean(rewards[-5:])
            wandb.log({
                "timesteps": timesteps,
                "mean_reward_last_5_episodes": mean_reward,
            })

            # Early Stopping logic
            if mean_reward > self.best_mean_reward + self.delta:
                self.best_mean_reward = mean_reward
                self.no_improvement_steps = 0
            else:
                self.no_improvement_steps += 1
                if self.no_improvement_steps >= self.patience:
                    print(f"Early stopping triggered at timestep {timesteps}")
                    return False  # Stop training

        return True
    
# Initialize EarlyStoppingCallback
early_stopping = EarlyStoppingCallback(patience=175200, delta=1e-1)

# Train the model
model.learn(
    total_timesteps=config.total_timesteps,
    progress_bar=True,
    callback=early_stopping
)

# Create directory for saving models
model_save_path = "../models"
os.makedirs(model_save_path, exist_ok=True)
model_path = os.path.join(model_save_path, "a2c_model.zip")
model.save(model_path)
print(f"Model saved to {model_path}")

# Evaluate the model
obs, info = env.reset()  # `reset` returns (obs, info)
rewards = []  # Used to store rewards at each step
actions = []  # Used to store actions at each step
cumulative_rewards = []  # Used to store cumulative rewards

# Record contributions of each reward component
reward_details = {
    "total_reward": [],
    "degradation_reward": [],
#    "charge_price_reward": [],
#    "discharge_price_reward": [],
    "user_satisfaction_reward": [],
    "transformer_load_reward": [],
    "total_costs": [],
    "soc_reward": [],
#    "invalid_action_penalty": [],
}

# Evaluation loop
cumulative_reward = 0
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

    # Extract reward contributions from info
    reward_contributions = info["reward_contributions"]

    # Record each reward component in the dictionary
    reward_details["total_reward"].append(reward)
    reward_details["degradation_reward"].append(reward_contributions.get("degradation_reward", 0))
    reward_details["user_satisfaction_reward"].append(reward_contributions.get("user_satisfaction_reward", 0))
    reward_details["transformer_load_reward"].append(reward_contributions.get("transformer_load_reward", 0))
    reward_details["total_costs"].append(reward_contributions.get("total_costs", 0))
    reward_details["soc_reward"].append(reward_contributions.get("soc_reward", 0))
    #reward_details["invalid_action_penalty"].append(reward_contributions.get("invalid_action_penalty", 0))

    # Record charging or discharging reward based on action
    #if action > 0:  # If charging action
    #    reward_details["charge_price_reward"].append(reward_contributions.get("charge_price_reward", 0))
    #    reward_details["discharge_price_reward"].append(0)  # Discharging part is 0
    #elif action < 0:  # If discharging action
    #    reward_details["charge_price_reward"].append(0)  # Charging part is 0
    #    reward_details["discharge_price_reward"].append(reward_contributions.get("discharge_price_reward", 0))
    #else:
    #    # If no charging or discharging action
    #    reward_details["charge_price_reward"].append(0)
    #    reward_details["discharge_price_reward"].append(0)

    # Log to WandB
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
    #    "charge_price_reward": reward_contributions.get("charge_price_reward", 0),
    #    "discharge_price_reward": reward_contributions.get("discharge_price_reward", 0),
    #    "invalid_action_penalty": reward_contributions.get("invalid_action_penalty", 0)
    })

    if done:
        obs, info = env.reset()  # Reset the environment and unpack return values

# Finish WandB run
wandb.finish()