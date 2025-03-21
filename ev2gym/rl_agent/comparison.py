# -*- coding: utf-8 -*-
"""
RecurrentPPO Evaluation with Saved Model on Multiple Replay Files
(Combining results from spring, summer, autumn, winter)
"""

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from sb3_contrib import RecurrentPPO  # Import Recurrent PPO

# 添加自定义路径（根据你的项目结构调整）
sys.path.append("C:/Users/River/Desktop/EV2Gym-main/EV2Gym-main")
from ev2gym.rl_agent.reward import profit_maximization
from ev2gym.rl_agent.state import V2G_profit_max

# 设备设置
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# 环境和模型配置
config_file = "../example_config_files/V2GProfitMax.yaml"
reward_function = profit_maximization
state_function = V2G_profit_max

model_save_path = "../models/recurrentppo_model.zip"
if not os.path.exists(model_save_path):
    raise FileNotFoundError(f"Model file not found at {model_save_path}")

# 加载已训练的模型（注意：这里暂时不传入 env 参数，后续在每个 replay 中单独创建环境后再设置）
model = RecurrentPPO.load(
    model_save_path,
    device=device
)

# 定义所有 replay 文件列表
replay_files = [
    "replay/winter.pkl",
    "replay/spring.pkl",
    "replay/summer.pkl",
    "replay/autumn.pkl"
]

# 定义全局退化量偏移变量（累计退化量）
global_degradation_offset = 0

# 用于存储所有 replay 运行结果的全局容器
all_steps_list = []
all_cumulative_rewards = []
all_cumulative_costs = []
all_total_degradation_list = []
all_overload_list = []
all_price_list = []
all_action_list = []
all_arrival_satisfaction_list = []
all_departure_satisfaction_list = []

num_steps = 960           # 每个 replay 运行的步数
global_step_offset = 0    # 用于连接各个 replay 的步数

# 针对每个 replay 文件进行循环
for replay_file in replay_files:
    print(f"Processing replay file: {replay_file}")
    
    # 创建环境时指定不同的 replay 文件
    env = gym.make(
        'EV2Gym-v1',
        config_file=config_file,
        load_from_replay_path=replay_file,
        reward_function=reward_function,
        state_function=state_function
    )
    
    # 若需要，可更新模型的环境（如果模型内部需要调用 env 的属性）
    model.set_env(env)
    
    # 重置环境，获取初始观测
    obs, info = env.reset()
    
    # 初始化当前 replay 的数据容器
    steps_list = []
    cumulative_rewards = []
    cumulative_costs = []
    total_degradation_list = []
    overload_list = []
    price_list = []
    action_list = []
    arrival_satisfaction_list = []
    departure_satisfaction_list = []
    
    cumulative_reward = 0
    cumulative_cost = 0
    
    for i in range(num_steps):
        # 使用模型进行预测
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # 如果动作为多维，则取第 0 个
        if isinstance(action, (list, tuple, np.ndarray)):
            action_val = action[0]
        else:
            action_val = action

        # 注意：env.current_step 是每个 replay 内的步数
        current_step = max(0, env.current_step - 1)
        if hasattr(env, 'discharge_prices') and current_step < env.discharge_prices.shape[1]:
            price_val = env.discharge_prices[0, current_step]
        else:
            price_val = 0.0

        price_list.append(price_val)
        action_list.append(action_val)

        # 累计 reward 与 cost
        cumulative_reward += reward
        reward_contributions = info.get("reward_contributions", {})
        total_costs = reward_contributions.get("total_costs", 0)
        cumulative_cost += total_costs

        # 处理 EV 相关指标（假设 env.EVs 存在）
        if hasattr(env, 'EVs') and env.EVs:
            ev_degrade_sum = 0
            for ev in env.EVs:
                ev_degrade = getattr(ev, "total_degradation", 0)
                ev_degrade_sum += ev_degrade

                # 检查到达与离站时刻（注意：此处比较 env.current_step-1 与 EV 的时间参数）
                if env.current_step - 1 == ev.time_of_arrival:
                    arrival_satisfaction_list.append(ev.get_user_satisfaction())
                if env.current_step - 1 == ev.time_of_departure:
                    departure_satisfaction_list.append(ev.get_user_satisfaction())
            total_degradation = ev_degrade_sum + global_degradation_offset
        else:
            total_degradation = global_degradation_offset

        # 处理变压器负载（若存在）
        if hasattr(env, 'transformers') and env.transformers:
            overload = env.transformers[0].get_how_overloaded()
        else:
            overload = 0

        # 将当前步（加上偏移量）及指标记录下来
        steps_list.append(i + global_step_offset)
        cumulative_rewards.append(cumulative_reward)
        cumulative_costs.append(cumulative_cost)
        total_degradation_list.append(total_degradation)
        overload_list.append(overload)

        if done:
            obs, info = env.reset()  # 如果当前 replay 提前结束，则重置环境

    # 更新全局步数偏移
    global_step_offset += num_steps
    
    # 更新全局退化量偏移值，以当前 replay 最后的累计退化量为基准
    if total_degradation_list:
        global_degradation_offset = total_degradation_list[-1]

    # 将当前 replay 的数据合并到全局容器中
    all_steps_list.extend(steps_list)
    all_cumulative_rewards.extend(cumulative_rewards)
    all_cumulative_costs.extend(cumulative_costs)
    all_total_degradation_list.extend(total_degradation_list)
    all_overload_list.extend(overload_list)
    all_price_list.extend(price_list)
    all_action_list.extend(action_list)
    all_arrival_satisfaction_list.extend(arrival_satisfaction_list)
    all_departure_satisfaction_list.extend(departure_satisfaction_list)

# ------------------ 绘制合并后的图形 ------------------

# Plot 1: Cumulative Reward vs Steps
plt.figure(figsize=(10, 6))
plt.plot(all_steps_list, all_cumulative_rewards, label="Cumulative Reward")
plt.xlabel("Step")
plt.ylabel("Cumulative Reward")
plt.title("Cumulative Reward Over Steps (All Replays)")
plt.legend()
plt.show()

# Plot 2: Cumulative Cost vs Steps
plt.figure(figsize=(10, 6))
plt.plot(all_steps_list, all_cumulative_costs, label="Cumulative Cost", color="orange")
plt.xlabel("Step")
plt.ylabel("Cumulative Cost")
plt.title("Cumulative Cost Over Steps (All Replays)")
plt.legend()
plt.show()

# Plot 3: Total Degradation vs Steps
plt.figure(figsize=(10, 6))
plt.plot(all_steps_list, all_total_degradation_list, label="Total Degradation", color="red")
plt.xlabel("Step")
plt.ylabel("Total Degradation")
plt.title("Total Degradation Over Steps (All Replays)")
plt.legend()
plt.show()

# Plot 4: Overload vs Steps
plt.figure(figsize=(10, 6))
plt.plot(all_steps_list, all_overload_list, label="Overload", color="purple")
plt.xlabel("Step")
plt.ylabel("Overload")
plt.title("Transformer Overload Over Steps (All Replays)")
plt.legend()
plt.show()

# Plot 5: Action vs Price (scatter)
plt.figure(figsize=(8, 6))
plt.scatter(all_price_list, all_action_list, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Electricity Price")
plt.ylabel("Action (Charge/Discharge)")
plt.title("Action vs. Price (All Replays)")
plt.show()

# Plot 6: Histogram - Arrival Satisfaction Distribution
plt.figure(figsize=(8, 6))
plt.hist(all_arrival_satisfaction_list, bins=20, alpha=0.7, color='green')
plt.xlabel("User Satisfaction Score")
plt.ylabel("Frequency")
plt.title("Arrival Satisfaction Distribution (All Replays)")
plt.show()

# Plot 7: Histogram - Departure Satisfaction Distribution
plt.figure(figsize=(8, 6))
plt.hist(all_departure_satisfaction_list, bins=20, alpha=0.7, color='blue')
plt.xlabel("User Satisfaction Score")
plt.ylabel("Frequency")
plt.title("Departure Satisfaction Distribution (All Replays)")
plt.show()
