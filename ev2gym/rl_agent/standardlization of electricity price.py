# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 11:45:39 2024

@author: River
"""

import pandas as pd
import numpy as np

# 檔案路徑
file_path = "C:/Users/River/Desktop/EV2Gym-main/EV2Gym-main/ev2gym/data/Day-ahead_prices_202301010000_202401011700_Quarterhour.csv"

# 讀取 CSV 檔案
df_prices = pd.read_csv(file_path, sep=';', encoding='utf-8-sig',engine='python')

# 提取德國/盧森堡的電價數據
df_prices_germany = df_prices[['Start date', 'End date', 'Germany/Luxembourg [€/MWh] Calculated resolutions']].copy()

# 將 €/MWh 轉換成 €/kWh
df_prices_germany['Germany/Luxembourg [€/MWh] Calculated resolutions'] = df_prices_germany['Germany/Luxembourg [€/MWh] Calculated resolutions'] / 1000

# 保存標準化後的資料
output_file_path = "C:/Users/River/Desktop/EV2Gym-main/EV2Gym-main/ev2gym/data/Day-ahead_prices_202301010000_202401011700_Quarterhour_processed.csv"
df_prices_germany.to_csv(output_file_path, index=False)

print("電價數據已成功標準化並另存為新檔案:", output_file_path)
