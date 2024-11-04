# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 11:37:55 2024

@author: River
"""

import pandas as pd
import numpy as np

# 讀取家庭負載數據
file_path = "C:\\Users\\River\\Desktop\\EV2Gym-main\\EV2Gym-main\\ev2gym\\data\\standardlastprofil-haushalte-2023.csv"
household_loads_df = pd.read_csv(file_path)

# 提取家庭負載數據的列
household_loads = household_loads_df['SLP-Bandlastkunden HB [kWh]'].values

# 計算均值和標準差
load_mean = np.mean(household_loads)
load_std = np.std(household_loads)

# 確保標準差不是零，以避免除以零的情況
if load_std == 0:
    load_std = 1

# 進行標準化
household_loads_standardized = (household_loads - load_mean) / load_std

# 將標準化數據替換回 DataFrame 中
household_loads_df['SLP-Bandlastkunden HB [kWh]'] = household_loads_standardized

# 將標準化數據存回同一個檔案，覆蓋原檔案
household_loads_df.to_csv(file_path, index=False)

print("家庭負載數據已成功標準化並覆蓋原檔案。")
