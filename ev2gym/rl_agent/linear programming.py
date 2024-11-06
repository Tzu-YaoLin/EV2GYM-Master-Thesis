import numpy as np
import pandas as pd
from pulp import LpMaximize, LpProblem, LpVariable, lpSum
import matplotlib.pyplot as plt
import random
import math

def profit_maximizationLP(
    total_costs, low_price_threshold, high_price_threshold, 
    current_step, household_loads, electricity_prices
):
    ''' This simplified reward function considers only charging cost, discharging reward, and adjustments based on load and price '''

    reward = total_costs  # 充電成本累加到總成本

    # 當前的家庭負載和電價
    current_load = household_loads[current_step]
    current_price = electricity_prices[current_step]

    # 基於電價的調整
    price_penalty_weight = 0.5  # 電價懲罰/獎勵的權重
    if current_price < low_price_threshold:
        # 當電價低於低電價閾值時增加獎勵
        reward += price_penalty_weight * 100 * (low_price_threshold - current_price) / low_price_threshold
    elif current_price > high_price_threshold:
        # 當電價高於高電價閾值時增加懲罰
        reward -= price_penalty_weight * 100 * (current_price - high_price_threshold) / high_price_threshold

    # 基於家庭負載的調整
    load_penalty_weight = 0.3  # 負載懲罰/獎勵的權重
    if current_load > high_price_threshold:
        reward -= load_penalty_weight * 50  # 當負載高時，適度減少獎勵
    elif current_load < low_price_threshold:
        reward += load_penalty_weight * 50  # 當負載低時，適度增加獎勵

    return reward


# 加載家庭負載數據和電價數據
household_loads_df = pd.read_csv(
    'C:\\Users\\River\\Desktop\\EV2Gym-main\\EV2Gym-main\\ev2gym\\data\\standardlastprofil-haushalte-2023.csv', 
    sep=',', 
    engine='python'
)
household_loads = household_loads_df['SLP-Bandlastkunden HB [kWh]'].values

electricity_prices_df = pd.read_csv(
    'C:/Users/River/Desktop/EV2Gym-main/EV2Gym-main/ev2gym/data/Day-ahead_prices_202301010000_202401010000_Quarterhour_processed.csv', 
    sep=',', 
    engine='python'
)
electricity_prices = electricity_prices_df['Germany/Luxembourg [€/MWh] Calculated resolutions'].values

# 替換 NaN 值
electricity_prices = np.nan_to_num(electricity_prices, nan=0.0)

# 設定每一天的時間步數（96 個時間步，每 15 分鐘一次）
time_steps_per_day = 96

# 電價和家庭負載的閾值設置
low_price_threshold = np.percentile(electricity_prices, 25)  # 低電價閾值
high_price_threshold = np.percentile(electricity_prices, 75)  # 高電價閾值
high_load_threshold = np.percentile(household_loads, 75)      # 高負載閾值

# 隨機選擇 1 天的數據
num_days = 10
total_days = len(electricity_prices) // time_steps_per_day
random_days = random.sample(range(total_days), num_days)

# 選擇對應的時間步數數據
selected_indices = [i * time_steps_per_day + j for i in random_days for j in range(time_steps_per_day)]
selected_electricity_prices = electricity_prices[selected_indices]
selected_household_loads = household_loads[selected_indices]

# 定義參數
transformer_capacity = 200  # 變壓器的最大功率為 200 kW
battery_capacity = 50       # 電池容量為 50 kWh
min_soc = 0.1 * battery_capacity  # 10% 容量
max_soc = 0.9 * battery_capacity  # 90% 容量
min_charge_soc = 0.3 * battery_capacity  # 當 SOC 低於 30% 時，強制充電

# 定義線性規劃問題 - 收益最大化
model_lp = LpProblem(name="profit-maximization", sense=LpMaximize)

# 定義決策變數，分別表示充電和放電
charge = {t: LpVariable(name=f"charge_{t}", lowBound=0, upBound=1) for t in range(len(selected_electricity_prices))}
discharge = {t: LpVariable(name=f"discharge_{t}", lowBound=0, upBound=1) for t in range(len(selected_electricity_prices))}

# 電價和家庭負載的閾值設置
low_price_threshold = np.percentile(selected_electricity_prices, 25)
high_price_threshold = np.percentile(selected_electricity_prices, 75)

# 定義較小的放電獎勵
base_discharge_reward = 0.2  # 基礎放電獎勵
extra_discharge_reward_factor = 0.02  # 動態放電獎勵的比例

# 簡化目標函數：僅考慮充放電收益
model_lp += lpSum(
    charge[t] * selected_electricity_prices[t] - discharge[t] * selected_electricity_prices[t]
    for t in range(len(selected_electricity_prices))
), "Simplified_Profit"

# 添加約束：每一步只能充電或放電
for t in range(len(selected_electricity_prices)):
    model_lp += charge[t] + discharge[t] <= 1, f"charging_or_discharging_{t}"

# 添加 SOC 約束：設定 SOC 的上下限為 10% 和 90%
battery_capacity = 50  # 電池容量
soc_cumulative = 0.5 * battery_capacity  # 用於追蹤 SOC 累積變量
min_soc = 0.1 * battery_capacity  # 10% 容量
max_soc = 0.9 * battery_capacity  # 90% 容量
min_charge_soc = 0.3 * battery_capacity  # 當 SOC 低於 30% 時，強制充電

# 設置 SOC 變化的基本約束
for t in range(len(selected_electricity_prices)):
    soc_cumulative += charge[t] - discharge[t]
    model_lp += (min_soc <= soc_cumulative <= max_soc), f"soc_constraint_{t}"

# 解決問題
model_lp.solve()

# 獲取線性規劃解決方案的充放電行為
lp_solution_charge = [charge[t].varValue for t in range(len(selected_electricity_prices))]
lp_solution_discharge = [discharge[t].varValue for t in range(len(selected_electricity_prices))]

# 計算使用 RL reward function 的總 reward
lp_total_costs = 0  # 用於累積成本
lp_user_satisfaction_list = []
lp_rewards = []

# 使用 profit_maximization 評估每個時間步的 reward
for t in range(len(selected_electricity_prices)):
    # 計算成本
    cost = (lp_solution_charge[t] * selected_electricity_prices[t]) - (lp_solution_discharge[t] * selected_electricity_prices[t])
    lp_total_costs += cost  # 累積總成本

    # 假設每個時間步的用戶滿意度
    user_satisfaction = 0.9  # 假設一個範例滿意度值
    lp_user_satisfaction_list.append(user_satisfaction)

    # 計算該步的 reward
    reward = profit_maximizationLP(
        total_costs=lp_total_costs,
        low_price_threshold=low_price_threshold,
        high_price_threshold=high_price_threshold,
        current_step=t,
        household_loads=selected_household_loads,
        electricity_prices=selected_electricity_prices
        )

    # 存儲該步的 reward
    lp_rewards.append(reward)

# 計算總 reward
total_lp_reward = sum(lp_rewards)

print("Total Reward using RL reward function on Linear Programming solution:", total_lp_reward)

# 可視化充放電決策
plt.plot(lp_solution_charge, label='Charging Decision (LP)')
plt.plot(lp_solution_discharge, label='Discharging Decision (LP)')
plt.xlabel('Time Step')
plt.ylabel('Action')
plt.title('Linear Programming Solution with Selected Days')
plt.legend()
plt.show()
