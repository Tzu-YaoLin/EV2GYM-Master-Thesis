# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 15:17:32 2024

@author: River
"""

import numpy as np
import pandas as pd
from pulp import LpMaximize, LpProblem, LpVariable, lpSum
import matplotlib.pyplot as plt

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

# 定義線性規劃問題 - 收益最大化
model_lp = LpProblem(name="profit-maximization", sense=LpMaximize)

# 定義決策變數
charging_decision = {t: LpVariable(name=f"charging_{t}", lowBound=-1, upBound=1) for t in range(len(electricity_prices))}

# 加入收益最大化的目標函數，包含放電加成獎勵
extra_discharge_reward = 5  # 放電獎勵
model_lp += lpSum(
    charging_decision[t] * electricity_prices[t] + 
    (extra_discharge_reward * -charging_decision[t] if charging_decision[t] < 0 else 0)
    for t in range(len(electricity_prices))
), "Total_Profit_With_Discharge_Reward"

# 添加約束：例如電池的充電限制
battery_capacity = 50  # 假設電池容量為 50 kWh
soc = 0  # 初始電池充電量
for t in range(len(electricity_prices)):
    soc += charging_decision[t]
    model_lp += (0 <= soc <= battery_capacity), f"soc_constraint_{t}"  # 保證 SOC 在 0 和電池容量之間

# 解決問題
model_lp.solve()

# 輸出結果
lp_solution = [charging_decision[t].varValue for t in range(len(electricity_prices))]
total_profit = sum(lp_solution[t] * electricity_prices[t] for t in range(len(electricity_prices)))

print("Linear Programming Solution: Total Profit =", total_profit)

# 將結果繪製出來
plt.plot(lp_solution, label='Charging/Discharging Decision (LP)')
plt.xlabel('Time Step')
plt.ylabel('Action')
plt.title('Linear Programming Solution')
plt.legend()
plt.show()
