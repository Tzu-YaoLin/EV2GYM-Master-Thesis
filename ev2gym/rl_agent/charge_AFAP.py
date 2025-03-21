import sys
import wandb
import numpy as np

# Environment setup
sys.path.append("C:/Users/River/Desktop/EV2Gym-main/EV2Gym-main")

from ev2gym.models.ev2gym_env import EV2Gym
from ev2gym.baselines.heuristics import ChargeAsFastAsPossible
from ev2gym.rl_agent.reward import profit_maximization
from ev2gym.baselines.mpc.V2GProfitMax import V2GProfitMaxOracle

# Initialize wandb
wandb.init(
    project="Comparison",
    config={
        "algorithm": "ChargeAsFastAsPossible",
        "config_file": "../example_config_files/V2GProfitMax.yaml",
        "simulation_length": 35040,
    },
)

config = wandb.config

# Load configuration
config_file = config["config_file"]

# Initialize environment
env = EV2Gym(config_file=config_file, save_replay=True, save_plots=True)
state, _ = env.reset()

agent = ChargeAsFastAsPossible()  # Heuristic agent
# agent = V2GProfitMaxOracle(env, verbose=True)  # optimal solution

# Initialize tracking variables
reward_details = {
    "total_reward": [],
    "degradation_reward": [],
    "user_satisfaction_reward": [],
    "transformer_load_reward": [],
    "total_costs": [],
    "soc_reward": [],
}

cumulative_reward = 0
cumulative_cost = 0
average_soc = []

# Simulation loop
for t in range(config["simulation_length"]):
    # Get agent's action
    actions = agent.get_action(env)
    new_state, _, done, _, stats = env.step(actions)

    # Extract reward contributions from stats
    reward_contributions = stats.get("reward_contributions", {})
    user_satisfaction_reward = reward_contributions.get("user_satisfaction_reward", 0)
    degradation_reward = reward_contributions.get("degradation_reward", 0)
    transformer_load_reward = reward_contributions.get("transformer_load_reward", 0)
    total_costs = reward_contributions.get("total_costs", 0)
    soc_reward = reward_contributions.get("soc_reward", 0)

    soc_values = [
        ev.historic_soc[-1] for ev in env.EVs
        if hasattr(ev, 'historic_soc') and len(ev.historic_soc) > 0
    ]
    avg_soc = np.mean(soc_values) if soc_values else 0
    average_soc.append(avg_soc)
    # Calculate cumulative reward
    total_reward = sum(reward_contributions.values())
    cumulative_reward += total_reward
    cumulative_cost += reward_contributions.get("total_costs", 0)

    # Log to WandB
    wandb.log({
        "step": env.current_step,
        "user_satisfaction_reward": user_satisfaction_reward,
        "degradation_reward": degradation_reward,
        "transformer_load_reward": transformer_load_reward,
        "total_costs": total_costs,
        "soc_reward": soc_reward,
        "total_reward": total_reward,
        "cumulative_reward": cumulative_reward,
        "cumulative_cost": cumulative_cost,
        "average_soc": avg_soc,
    })

    # Reset environment if done
    if done:
        # print(f"Simulation completed at step {t}. Resetting environment.")
        state, _ = env.reset()
        for ev in env.EVs:
            ev.reset()
        cumulative_reward = 0

# Finish WandB logging
wandb.finish()