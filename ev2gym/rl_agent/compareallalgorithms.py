import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

# Import algorithm classes
from stable_baselines3 import PPO, A2C, SAC, TD3
from stable_baselines3.ddpg import DDPG
from sb3_contrib import RecurrentPPO  # RecurrentPPO from sb3_contrib

# Append custom path (adjust according to your project structure)
sys.path.append("C:/Users/River/Desktop/EV2Gym-main/EV2GYM-main")
from ev2gym.rl_agent.reward import profit_maximization
from ev2gym.rl_agent.state import V2G_profit_max
from ev2gym.baselines.heuristics import ChargeAsFastAsPossibleWrapper
from ev2gym.baselines.gurobi_models.profit_max import V2GProfitMaxOracleGBWrapper

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Common environment and model configuration
config_file = "../example_config_files/V2GProfitMax.yaml"
reward_function = profit_maximization
state_function = V2G_profit_max

# Define the replay file (now using only one)
replay_files = ["replay/replay_sim_2025_03_09_780136.pkl"]

# Number of steps to simulate
num_steps = 35040

# Define algorithm classes
algo_classes = {
    "ddpg": DDPG,
    "td3": TD3,
    "ppo": PPO,
    "a2c": A2C,
    "sac": SAC,
    "rppo": RecurrentPPO,
    "dumbcharging": ChargeAsFastAsPossibleWrapper,
    "milp": V2GProfitMaxOracleGBWrapper,
}

# Pre-define colors for each algorithm to ensure consistency across plots
algorithm_colors = {
    "DDPG": "blue",
    "TD3": "green",
    "PPO": "red",
    "A2C": "purple",
    "SAC": "orange",
    "RPPO": "magenta",
    "DUMBCHARGING": "brown",
    "MILP": "cyan",
}

# Dictionary to store results for each algorithm
results = {}  # Structure: results[algo] = { "steps": [...], "cumulative_rewards": [...], ... }

def evaluate_model(model, replay_file, config_file, reward_function, state_function, num_steps):
    """
    Evaluate a single model on the given replay file.
    """
    print(f"Processing replay file: {replay_file}")
    # Create a new environment based on the replay file
    env = gym.make(
        'EV2Gym-v1',
        config_file=config_file,
        load_from_replay_path=replay_file,
        reward_function=reward_function,
        state_function=state_function
    )
    
    # For MILP models, reinitialize a new MILP instance for the replay file
    if hasattr(model, "algo_name") and model.algo_name == "Optimal (Offline)":
        current_model = V2GProfitMaxOracleGBWrapper.load(replay_path=replay_file, device=device)
        current_model.set_env(env)
    else:
        model.set_env(env)
        current_model = model

    obs, info = env.reset()

    # Local data containers for current replay file
    steps_list = []
    cumulative_rewards = []
    cumulative_costs = []
    total_degradation_list = []
    overload_list = []
    price_list = []
    action_list = []
    arrival_satisfaction_list = []
    departure_satisfaction_list = []
    soc_list = []
    
    cumulative_reward = 0
    cumulative_cost = 0

    for i in range(num_steps):
        action, _states = current_model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # If action is multidimensional, take the first element for logging
        if isinstance(action, (list, tuple, np.ndarray)):
            action_val = action[0]
        else:
            action_val = action

        # Current step (note: env.current_step is within the replay file)
        current_step = max(0, env.current_step - 1)
        if hasattr(env, 'discharge_prices') and current_step < env.discharge_prices.shape[1]:
            price_val = env.discharge_prices[0, current_step]
        else:
            price_val = 0.0

        price_list.append(price_val)
        action_list.append(action_val)

        cumulative_reward += reward
        reward_contributions = info.get("reward_contributions", {})
        total_costs = reward_contributions.get("total_costs", 0)
        cumulative_cost -= total_costs

        total_charged = 0
        total_discharged = 0
    
        if hasattr(env, 'charging_stations'):
            for cs in env.charging_stations:
                total_charged = env.charging_stations[0].total_energy_charged
                total_discharged = env.charging_stations[0].total_energy_discharged
        else:
            print("No charging_stations found in environment!")
        
        # 記錄 EV 的 SoC
        if hasattr(env, 'EVs') and env.EVs:
            # 假設有多個 EV，這裡我們取平均值
            current_soc = np.mean([ev.get_soc() for ev in env.EVs])
        else:
            current_soc = 0
        soc_list.append(current_soc)

        # Degradation tracking
        if hasattr(env, 'EVs') and env.EVs:
            ev_degrade_sum = 0
            for ev in env.EVs:
                ev_degrade = getattr(ev, "total_degradation", 0)
                ev_degrade_sum += ev_degrade
                if env.current_step == ev.time_of_arrival:
                    arrival_satisfaction_list.append(ev.get_user_satisfaction())
                if env.current_step == ev.time_of_departure:
                    departure_satisfaction_list.append(ev.get_user_satisfaction())
            total_degradation = ev_degrade_sum
        else:
            total_degradation = 0

        # Transformer overload
        if hasattr(env, 'transformers') and env.transformers:
            overload = env.transformers[0].get_how_overloaded()
        else:
            overload = 0

        steps_list.append(i)
        cumulative_rewards.append(cumulative_reward)
        cumulative_costs.append(cumulative_cost)
        total_degradation_list.append(total_degradation)
        overload_list.append(overload)

        if done:
            obs, info = env.reset()

    return {
        "steps": steps_list,
        "cumulative_rewards": cumulative_rewards,
        "cumulative_costs": cumulative_costs,
        "total_degradation": total_degradation_list,
        "overload": overload_list,
        "price": price_list,
        "action": action_list,
        "arrival_satisfaction": arrival_satisfaction_list,
        "departure_satisfaction": departure_satisfaction_list,
        "soc": soc_list,  # 將 SoC 資料加入回傳字典
        "cs_charged": total_charged,
        "cs_discharged": total_discharged,
    }

# Evaluate each algorithm and store the results
for algo, AlgoClass in algo_classes.items():
    algo_upper = algo.upper()
    if algo in ["dumbcharging"]:
        print(f"Evaluating heuristic/offline algorithm: {algo_upper}")
        model = AlgoClass.load(model_path=None, device=device)
    elif algo == "milp":
        print(f"Evaluating offline MILP algorithm: {algo_upper}")
        model = AlgoClass.load(replay_path=replay_files[0], device=device)
    else:
        model_path = f"../models/{algo}_model (2).zip"
        if not os.path.exists(model_path):
            print(f"Model file for {algo_upper} not found at {model_path}, skipping.")
            continue
        print(f"Evaluating algorithm: {algo_upper}")
        model = AlgoClass.load(model_path, device=device)
    
    # Since we only have one replay file, pass replay_files[0]
    results[algo_upper] = evaluate_model(model, replay_files[0], config_file, reward_function, state_function, num_steps=num_steps)

# ------------------ Overlay Plots ------------------

# 1. Cumulative Reward vs Steps (Overlay)
plt.figure(figsize=(10, 6))
for algo_upper, data in results.items():
    plt.plot(data["steps"], data["cumulative_rewards"],
             color=algorithm_colors.get(algo_upper, 'black'),
             label=algo_upper)
plt.xlabel("Step")
plt.ylabel("Cumulative Reward")  # Example unit if reward is cost-based
plt.title("Cumulative Reward Over Steps (Overlay of Different Algorithms)")
plt.legend()
plt.show()

# 2. Cumulative Cost vs Steps (Overlay)
plt.figure(figsize=(10, 6))
for algo_upper, data in results.items():
    plt.plot(data["steps"], data["cumulative_costs"],
             color=algorithm_colors.get(algo_upper, 'black'),
             label=algo_upper)
plt.xlabel("Step")
plt.ylabel("Cumulative Cost (EUR)")
plt.title("Cumulative Cost Over Steps (Overlay of Different Algorithms)")
plt.legend()
plt.show()

# 3. Total Degradation vs Steps (Overlay)
plt.figure(figsize=(10, 6))
for algo_upper, data in results.items():
    plt.plot(data["steps"], data["total_degradation"],
             color=algorithm_colors.get(algo_upper, 'black'),
             label=algo_upper)
plt.xlabel("Step")
plt.ylabel("Total Degradation")
plt.title("Total Degradation Over Steps (Overlay of Different Algorithms)")
plt.legend()
plt.show()

# 4. Transformer Overload vs Steps (Overlay)
plt.figure(figsize=(10, 6))
for algo_upper, data in results.items():
    plt.plot(data["steps"], data["overload"],
             color=algorithm_colors.get(algo_upper, 'black'),
             label=algo_upper)
plt.xlabel("Step")
plt.ylabel("Transformer Overload (kWh)")
plt.title("Transformer Overload Over Steps (Overlay of Different Algorithms)")
plt.legend()
plt.show()

# Calculate total transformer overload for each algorithm and plot as a bar chart
total_overload_by_algo = {}
for algo_upper, data in results.items():
    total_overload_by_algo[algo_upper] = sum(data["overload"])

plt.figure(figsize=(10, 6))
plt.bar(total_overload_by_algo.keys(), total_overload_by_algo.values(),
        color=[algorithm_colors.get(k, 'black') for k in total_overload_by_algo.keys()])
plt.xlabel("Algorithm")
plt.ylabel("Total Transformer Overload (kWh)")
plt.title("Total Transformer Overload for Each Algorithm")
plt.show()

# ------------------ Non-Overlay Plots ------------------

# 5. Action vs Price: Plot separately for each algorithm
for algo_upper, data in results.items():
    plt.figure(figsize=(8, 6))
    plt.scatter(data["price"], data["action"], alpha=0.5,
                color=algorithm_colors.get(algo_upper, 'black'))
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel("Electricity Price (EUR/kWh)")
    plt.ylabel("Action (Charge/Discharge)")
    plt.title(f"Action vs. Price for {algo_upper}")
    plt.show()

# 6. User Satisfaction Distributions: Plot separately for arrival and departure for each algorithm
for algo_upper, data in results.items():
    plt.figure(figsize=(8, 6))
    plt.hist(data["arrival_satisfaction"], bins=20, alpha=0.7,
             color=algorithm_colors.get(algo_upper, 'green'))
    plt.xlabel("State of Charge (%)")
    plt.ylabel("Frequency")
    plt.title("Arrival SoC Distribution")
    plt.show()
    
    plt.figure(figsize=(8, 6))
    plt.hist(data["departure_satisfaction"], bins=20, alpha=0.7,
             color=algorithm_colors.get(algo_upper, 'blue'))
    plt.xlabel("State of Charge (%)")
    plt.ylabel("Frequency")
    plt.title(f"Departure SoC Distribution for {algo_upper}")
    plt.show()

# ------------------ Additional Plots (using EV_Charger data) ------------------

cs_charged_energy_by_algo = {}
cs_discharged_energy_by_algo = {}

for algo_upper, data in results.items():
    cs_charged_energy_by_algo[algo_upper] = data["cs_charged"]
    cs_discharged_energy_by_algo[algo_upper] = data["cs_discharged"]

# Plot bar chart for total charged energy (kWh) from EV_Charger
plt.figure(figsize=(10, 6))
plt.bar(cs_charged_energy_by_algo.keys(),
        cs_charged_energy_by_algo.values(),
        color=[algorithm_colors.get(k, 'orange') for k in cs_charged_energy_by_algo.keys()])
plt.xlabel("Algorithm")
plt.ylabel("Total Charged Energy (kWh)")
plt.title("Total Charged Energy for Each Algorithm")
plt.show()

# Plot bar chart for total discharged energy (kWh) from EV_Charger
plt.figure(figsize=(10, 6))
plt.bar(cs_discharged_energy_by_algo.keys(),
        cs_discharged_energy_by_algo.values(),
        color=[algorithm_colors.get(k, 'blue') for k in cs_discharged_energy_by_algo.keys()])
plt.xlabel("Algorithm")
plt.ylabel("Total Discharged Energy (kWh)")
plt.title("Total Discharged Energy for Each Algorithm")
plt.show()

# ------------------ Print Final Metrics Table ------------------
print("\nFinal Metrics Summary (Per Algorithm)\n")
print("{:<12} {:>15} {:>15} {:>15} {:>15} {:>15} {:>15}".format(
    "Algorithm", 
    "Cumul.Reward", 
    "Cumul.Cost(EUR)", 
    "Overload(kWh)", 
    "Degrad.Final", 
    "Charged(kWh)", 
    "Disch.(kWh)"
))

for algo_label, data in results.items():
    # final cumulative reward is the last element
    final_cumul_reward = data["cumulative_rewards"][-1] if data["cumulative_rewards"] else 0
    # final cumulative cost is the last element
    final_cumul_cost = data["cumulative_costs"][-1] if data["cumulative_costs"] else 0
    # total overload is sum of the overload array
    sum_overload = sum(data["overload"])
    # final degrade is the last element
    final_degrade = data["total_degradation"][-1] if data["total_degradation"] else 0
    # total charged and discharged from the dictionaries
    charger_in = data["cs_charged"]      # total_energy_charged
    charger_out = data["cs_discharged"]  # total_energy_discharged

    print("{:<12} {:>15.2f} {:>15.2f} {:>15.2f} {:>15.4f} {:>15.2f} {:>15.2f}".format(
        algo_label, 
        final_cumul_reward, 
        final_cumul_cost, 
        sum_overload, 
        final_degrade, 
        charger_in,
        charger_out
    ))
    
# 只畫前100步的 EV SoC 趨勢圖 (僅針對 RPPO 與 MILP)
plt.figure(figsize=(10, 6))
for algo_upper, data in results.items():
    if algo_upper in ["RPPO", "MILP"]:
        plt.plot(data["steps"][250:350], data["soc"][250:350],
                 color=algorithm_colors.get(algo_upper, 'black'),
                 label=algo_upper)
plt.xlabel("Timestep")
plt.ylabel("EV SoC (%)")
plt.title("EV SoC Over 100 Timesteps for RPPO and MILP")
plt.legend()
plt.show()
