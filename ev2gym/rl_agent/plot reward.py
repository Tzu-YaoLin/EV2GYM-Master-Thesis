import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns  # If Seaborn is not installed, install with: pip install seaborn

# Set plotting range and variables
steps = 100
charge_prices = np.linspace(-0.5, 0.52427, steps)  # Assume range for charge prices

discharge_prices = charge_prices * 1  # Discharge price is 1 times the charge price

# Filter non-negative charge prices and calculate quartiles
positive_charge_prices = charge_prices[charge_prices >= 0]
charge_q1, charge_q3 = np.percentile(positive_charge_prices, [25, 75])  # Calculate quartiles from 0 to max value

# Filter non-negative discharge prices and calculate quartiles
positive_discharge_prices = discharge_prices[discharge_prices >= 0]
discharge_q1, discharge_q3 = np.percentile(positive_discharge_prices, [25, 75])  # Calculate quartiles

user_satisfaction = np.linspace(0, 1, steps)  # Range for user satisfaction
charging_cycles = np.linspace(0, 30, steps)  # Number of battery charging cycles
load_ratios = np.linspace(0, 1.5, steps)  # Transformer load ratio relative to max load

# 1. Impact of charge price on reward
def calculate_reward_for_charge_price(price):
    reward = 0
    if price < 0:  # Provide high reward for negative prices
        return 100
    elif price < charge_q1:  # Reward for prices below the first quartile
        return 50
    elif price > charge_q3:  # Penalty for prices above the third quartile
        return -50
    else:  # Reward is 0 in other cases
        return reward

charge_rewards = [calculate_reward_for_charge_price(p) for p in charge_prices]
# Plotting the impact
plt.plot(charge_prices, charge_rewards, label='Charge Price Reward')
plt.axvline(x=charge_q1, color='green', linestyle='--', label='Q1 (25th percentile)')
plt.axvline(x=charge_q3, color='red', linestyle='--', label='Q3 (75th percentile)')
plt.xlabel('Charge Price')
plt.ylabel('Reward Impact')
plt.title('Impact of Charge Price on Reward (Updated)')
plt.grid(True)
plt.legend()
plt.show()

# 2. Impact of discharge price on reward
def calculate_reward_for_discharge_price(price):
    reward = 0
    if price < 0:  # Large penalty for negative prices
        return -100
    elif price > discharge_q3:  # Reward for prices above the third quartile
        return 50
    elif price < discharge_q1:  # Penalty for prices below the first quartile
        return -50
    else:  # Reward is 0 in other cases
        return reward

discharge_rewards = [calculate_reward_for_discharge_price(p) for p in discharge_prices]
# Plotting the impact
plt.plot(discharge_prices, discharge_rewards, label='Discharge Price Reward', color='orange')
plt.axvline(x=discharge_q1, color='green', linestyle='--', label='Q1 (25th percentile)')
plt.axvline(x=discharge_q3, color='red', linestyle='--', label='Q3 (75th percentile)')
plt.xlabel('Discharge Price')
plt.ylabel('Reward Impact')
plt.title('Impact of Discharge Price on Reward (Updated)')
plt.grid(True)
plt.legend()
plt.show()

# 3. Impact of user satisfaction on reward
def calculate_reward_for_user_satisfaction(score):
    reward = -10 * math.exp(-5 * (score - 0.7))
    return reward

user_satisfaction_rewards = [calculate_reward_for_user_satisfaction(s) for s in user_satisfaction]
plt.plot(user_satisfaction, user_satisfaction_rewards)
plt.xlabel('User Satisfaction')
plt.ylabel('Reward Impact')
plt.title('Impact of User Satisfaction on Reward')
plt.grid(True)
plt.show()

# 4. Impact of battery degradation on reward
def calculate_reward_for_degradation(charging_cycle, degradation_level):
    degradation_cost_factor = 0.3
    if degradation_level < 0.1:
        reward = -charging_cycle * degradation_cost_factor
    elif degradation_level < 0.2:
        reward = -charging_cycle * degradation_cost_factor * 4
    else:
        reward = -charging_cycle * degradation_cost_factor * 6
    return reward

# Plot the reward impact at different degradation levels
degradation_levels = [0.05, 0.15, 0.4]  # Three degradation levels

plt.figure(figsize=(10, 6))
for degradation_level in degradation_levels:
    degradation_rewards = [calculate_reward_for_degradation(c, degradation_level) for c in charging_cycles]
    plt.plot(charging_cycles, degradation_rewards, label=f'Degradation Level = {degradation_level}')

plt.xlabel('Charging Cycles')
plt.ylabel('Reward Impact')
plt.title('Impact of Battery Degradation on Reward (Different Degradation Levels)')
plt.grid(True)
plt.legend()
plt.show()

# 5. Impact of transformer load on reward
def calculate_reward_for_load(load_ratio):
    load_threshold = 0.7
    reward = 0
    if load_ratio > load_threshold:
        reward -= 30 * (load_ratio - load_threshold)
        reward += 25 if load_ratio > 1.0 else 0  # Additional reward when discharging under high load conditions
    else:
        reward += 30 * (load_threshold - load_ratio)
    return reward

load_rewards = [calculate_reward_for_load(l) for l in load_ratios]
plt.plot(load_ratios, load_rewards)
plt.xlabel('Load Ratio (Current Load / Max Load)')
plt.ylabel('Reward Impact')
plt.title('Impact of Transformer Load on Reward')
plt.grid(True)
plt.show()

# Calculate total rewards by summing up individual components
total_rewards = []
degradation_level = 0.1  # Set a fixed degradation level
for i in range(steps):
    reward = (calculate_reward_for_charge_price(charge_prices[i]) +
              calculate_reward_for_discharge_price(discharge_prices[i]) +
              calculate_reward_for_user_satisfaction(user_satisfaction[i]) +
              calculate_reward_for_degradation(charging_cycles[i], degradation_level) +
              calculate_reward_for_load(load_ratios[i]))
    total_rewards.append(reward)

plt.plot(np.arange(steps), total_rewards)
plt.xlabel('Step')
plt.ylabel('Total Reward Impact')
plt.title('Cumulative Reward Impact of Multiple Factors')
plt.grid(True)
plt.show()

# Generate a reward matrix for visualization
reward_matrix = np.zeros((steps, steps))
for i, p in enumerate(charge_prices):
    for j, s in enumerate(user_satisfaction):
        reward_matrix[i, j] = (calculate_reward_for_charge_price(p) +
                               calculate_reward_for_user_satisfaction(s))

# Visualize the reward matrix with a heatmap
sns.heatmap(reward_matrix, xticklabels=False, yticklabels=False, cmap="coolwarm")
plt.xlabel('User Satisfaction Index')
plt.ylabel('Charge Price Index')
plt.title('Combined Impact of Charge Price and User Satisfaction on Reward')
plt.show()
