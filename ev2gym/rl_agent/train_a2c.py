# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 16:33:02 2024

@author: River
"""
import numpy as np
import matplotlib.pyplot as plt
import wandb  # Import WandB

# Environment setup
import sys
sys.path.append("/Workspace/Users/extern.tzu.yao.lin@cariad.technology/EV2GYM-Master-Thesis")

import gymnasium as gym
from stable_baselines3 import A2C
from ev2gym.rl_agent.reward import profit_maximization
from ev2gym.rl_agent.state import V2G_profit_max
from ev2gym.visuals.plots import ev_city_plot, visualize_step

# Login to WandB
wandb.login(key='e09f1372cfe6ed2cd13773350fe920cb084c6341')

# Initialize WandB for hyperparameter tuning
wandb.init(sync_tensorboard=True, project="ev2gym_a2c_tuning", config={
    "policy_type": "MlpPolicy",
    "total_timesteps": 350400,
    "learning_rate": 1e-4,
    "n_steps": 5,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5
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

# Initialize DDPG model with WandB configuration
model = A2C(
    config.policy_type,
    env,
    device="cpu",
    learning_rate=config.learning_rate,
    n_steps=config.n_steps,
    gamma=config.gamma,
    gae_lambda=config.gae_lambda,
    ent_coef=config.ent_coef,
    vf_coef=config.vf_coef,
    max_grad_norm=config.max_grad_norm,
    verbose=1  # Optional: Display more training details
)

def log_callback(_locals, _globals):
    wandb.log({"timesteps": _locals['self'].num_timesteps})
    return True 

# Train the model and log progress to WandB
model.learn(
    total_timesteps=config.total_timesteps,
    progress_bar=True,
    callback=log_callback
)

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
for i in range(70080):
    # Unpack step() return values
    action, _states = model.predict(obs, deterministic=True)  # Pass pure observation obs
    obs, reward, terminated, truncated, info = env.step(action)  # Unpack 5 return values
    done = terminated or truncated  # Combine terminated and truncated

    # Record total reward and action
    rewards.append(reward)
    actions.append(action)
    cumulative_reward += reward
    cumulative_rewards.append(cumulative_reward)

    # Visualize the current step's details
    # visualize_step(env)

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
        # If no charging or discharging action
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
        "soc_reward": reward_contributions.get("soc_reward", 0),
    #    "charge_price_reward": reward_contributions.get("charge_price_reward", 0),
    #    "discharge_price_reward": reward_contributions.get("discharge_price_reward", 0),
    #    "invalid_action_penalty": reward_contributions.get("invalid_action_penalty", 0)
    })

    # # Generate city-level data visualization every 1000 steps
    # if i % 10000 == 0 and i > 0:
    #     try:
    #         ev_city_plot(env)
    #     except Exception as e:
    #         print(f"Warning: Failed to generate city plot at step {i}: {e}")

    if done:
        obs, info = env.reset()  # Reset the environment and unpack return values

# # Plot total reward
# plt.figure(figsize=(8, 5))
# plt.plot(rewards, label='Reward per Step')
# plt.xlabel("Step")
# plt.ylabel("Reward")
# plt.title("Reward per Step")
# plt.legend()
# plt.show()

# # Plot cumulative reward
# plt.figure(figsize=(8, 5))
# plt.plot(cumulative_rewards, label='Cumulative Reward')
# plt.xlabel("Step")
# plt.ylabel("Cumulative Reward")
# plt.title("Cumulative Reward")
# plt.legend()
# plt.show()

# # Plot contributions of each reward component
# plt.figure(figsize=(10, 6))
# for key, values in reward_details.items():
#     if key != "total_reward":  # Skip total reward
#         plt.plot(values, label=key)

# plt.xlabel("Step")
# plt.ylabel("Reward Contribution")
# plt.title("Reward Contributions Over Steps")
# plt.legend()
# plt.show()

# # Plot action distribution
# plt.figure(figsize=(8, 5))
# plt.hist(actions, bins=20, alpha=0.7, label='Action Distribution')
# plt.xlabel("Action Value")
# plt.ylabel("Frequency")
# plt.title("Action Distribution")
# plt.legend()
# plt.show()

# Finish WandB run
wandb.finish()
