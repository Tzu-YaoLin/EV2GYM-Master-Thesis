'''This file contains various example reward functions for the RL agent. Users can create their own reward function here or in their own file using the same structure as below
'''
import numpy as np
import math
#import pandas as pd

# 讀取電價資料 (假設您只需要 'Germany/Luxembourg [€/MWh]' 列的電價)
#electricity_prices_df = pd.read_csv('C:/Users/River/Desktop/EV2Gym-main/EV2Gym-main/ev2gym/data/Day-ahead_prices_202301010000_202401010000_Quarterhour.csv', sep=';', engine='python')

# 提取電價列並將其轉換為 numpy 陣列
#electricity_prices = electricity_prices_df['Germany/Luxembourg [€/MWh] Calculated resolutions'].values

# 計算低電價和高電價的閾值
#low_price_threshold = np.percentile(electricity_prices, 25)  # 第25百分位數
#high_price_threshold = np.percentile(electricity_prices, 75)  # 第75百分位數

def SquaredTrackingErrorReward(env,*args):
    '''This reward function is the squared tracking error that uses the minimum of the power setpoints and the charge power potential
    The reward is negative'''
    
    reward = - (min(env.power_setpoints[env.current_step-1], env.charge_power_potential[env.current_step-1]) -
        env.current_power_usage[env.current_step-1])**2
        
    return reward

def SqTrError_TrPenalty_UserIncentives(env, _, user_satisfaction_list, *args):
    ''' This reward function is the squared tracking error that uses the minimum of the power setpoints and the charge power potential
    It penalizes transofrmers that are overloaded    
    The reward is negative'''
    
    tr_max_limit = env.transformers[0].max_power[env.current_step-1]
    
    reward = - (min(env.power_setpoints[env.current_step-1], env.charge_power_potential[env.current_step-1],tr_max_limit) -
        env.current_power_usage[env.current_step-1])**2
            
    for tr in env.transformers:
        reward -= 100 * tr.get_how_overloaded()
        
    for score in user_satisfaction_list:
        reward -= 1000 * (1 - score)
                    
    return reward

def ProfitMax_TrPenalty_UserIncentives(env, total_costs, user_satisfaction_list, *args):
    
    reward = total_costs
    
    for tr in env.transformers:
        reward -= 100 * tr.get_how_overloaded()                        
    
    for score in user_satisfaction_list:        
        reward -= 100 * math.exp(-10*score)
        
    return reward

def SquaredTrackingErrorRewardWithPenalty(env,*args):
    ''' This reward function is the squared tracking error that uses the minimum of the power setpoints and the charge power potential
    The reward is negative
    If the EV is not charging, the reward is penalized
    '''
    if env.current_power_usage[env.current_step-1] == 0 and env.charge_power_potential[env.current_step-2] != 0:
        reward = - (min(env.power_setpoints[env.current_step-1], env.charge_power_potential[env.current_step-1]) -
            env.current_power_usage[env.current_step-1])**2 - 100
    else:
        reward = - (min(env.power_setpoints[env.current_step-1], env.charge_power_potential[env.current_step-1]) -
            env.current_power_usage[env.current_step-1])**2
    
    return reward

def SimpleReward(env,*args):
    '''This reward function does not consider the charge power potential'''
    
    reward = - (env.power_setpoints[env.current_step-1] - env.current_power_usage[env.current_step-1])**2
    
    return reward

def MinimizeTrackerSurplusWithChargeRewards(env,*args):
    ''' This reward function minimizes the tracker surplus and gives a reward for charging '''
    
    reward = 0
    if env.power_setpoints[env.current_step-1] < env.current_power_usage[env.current_step-1]:
            reward -= (env.current_power_usage[env.current_step-1]-env.power_setpoints[env.current_step-1])**2

    reward += env.current_power_usage[env.current_step-1] #/75
    
    return reward

def profit_maximization(env, total_costs, user_satisfaction_list, low_price_threshold, high_price_threshold, *args):
    ''' This reward function is used for the profit maximization case '''
    
    # 確保 total_costs 是數值型
    if isinstance(total_costs, torch.Tensor):
        total_costs = total_costs.item()  # 將 Tensor 轉換為 Python 數值

    if not isinstance(total_costs, (int, float)):
        raise TypeError(f"Expected total_costs to be int or float, but got {type(total_costs)}")


   #reward = max(0, float(total_costs))  # 確保 total_costs 是正數浮點數
    reward = total_costs
    # 檢查 current_step 是否超出變壓器最大功率數組範圍
    if env.current_step < len(env.transformers[0].max_power):
        transformer_capacity = env.transformers[0].max_power[env.current_step]
    else:
        transformer_capacity = env.transformers[0].max_power[-1]

    # 如果 user_satisfaction_list 是整數，將其轉換為列表
    if isinstance(user_satisfaction_list, int):
        user_satisfaction_list = [user_satisfaction_list]

    # 用戶滿意度處罰
    user_satisfaction_weight = 0.8  # 調整這裡的權重
    for score in user_satisfaction_list:
        score = max(-10, score)  # 限制最低分數，避免過度懲罰
        reward -= user_satisfaction_weight * 400 * score  # 減少懲罰的幅度，並使用線性懲罰
        
    # 加入基於家庭負載的處罰
    current_load = env.household_loads[env.current_step]  # 當前家庭負載
    available_capacity = transformer_capacity - current_load
    transformer_penalty_weight = 0.4  # 調整這裡的權重

    # 如果家庭負載超過變壓器容量，根據超出容量的比例懲罰
    if available_capacity < 0:
        reward -= transformer_penalty_weight * 100 * abs(available_capacity) / transformer_capacity  # 減少懲罰的強度

    # 加入電價影響，使用實時電價
    current_price = env.electricity_prices[env.current_step]  # 當前電價（€/MWh）
    price_penalty_weight = 0.7  # 調整這裡的權重

    # 根據電價差距動態調整獎勵或懲罰
    if current_price < low_price_threshold:
        reward += price_penalty_weight * 500 * (low_price_threshold - current_price) / low_price_threshold
    elif current_price > high_price_threshold:
        reward -= price_penalty_weight * 500 * (current_price - high_price_threshold) / high_price_threshold

    # 檢查是否出現 NaN 或無窮大情況
    if math.isnan(reward) or math.isinf(reward):
        print(f"Invalid reward detected: {reward} at step {env.current_step}")
        reward = -1000  # or set to another large negative value to discourage this scenario

    return reward


# Previous reward functions for testing
#############################################################################################################
        # reward = total_costs  # - 0.5
        # print(f'total_costs: {total_costs}')
        # print(f'user_satisfaction_list: {user_satisfaction_list}')
        # for score in user_satisfaction_list:
        #     reward -= 100 * (1 - score)

        # Punish invalid actions (actions that try to charge or discharge when there is no EV connected)
        # reward -= 2 * (invalid_action_punishment/self.number_of_ports)

        # reward = min(2, 1 * 4 * self.cs / (0.00001 + (
        #     self.power_setpoints[self.current_step-1] - self.current_power_usage[self.current_step-1])**2))

        # this is the new reward function
        # reward = min(2, 1/((min(self.power_setpoints[self.current_step-1], self.charge_power_potential[self.current_step-1]) -
        #           self.current_power_usage[self.current_step-1])**2 + 0.000001))

        # new_*10*charging
        # if self.power_setpoints[self.current_step-1] < self.current_power_usage[self.current_step-1]:
        #     reward -= (min(self.power_setpoints[self.current_step-1], self.charge_power_potential[self.current_step-1]) -
        #             self.current_power_usage[self.current_step-1])
        # else:
        #     reward -= (min(self.power_setpoints[self.current_step-1], self.charge_power_potential[self.current_step-1]) -
        #             self.current_power_usage[self.current_step-1])*10

        # new_1_equal
        # if self.power_setpoints[self.current_step-1] < self.current_power_usage[self.current_step-1]:
        #     reward -= (min(self.power_setpoints[self.current_step-1], self.charge_power_potential[self.current_step-1]) -
        #             self.current_power_usage[self.current_step-1])
        # else:
        #     reward -= (min(self.power_setpoints[self.current_step-1], self.charge_power_potential[self.current_step-1]) -
        #             self.current_power_usage[self.current_step-1])

        # new_0.1
        # if self.power_setpoints[self.current_step-1] < self.current_power_usage[self.current_step-1]:
        #     reward -= (min(self.power_setpoints[self.current_step-1], self.charge_power_potential[self.current_step-1]) -
        #             self.current_power_usage[self.current_step-1])**2
        # else:
        #     reward -= (min(self.power_setpoints[self.current_step-1], self.charge_power_potential[self.current_step-1]) -
        #             self.current_power_usage[self.current_step-1])*0.1

        # new_reward squared
        # if self.power_setpoints[self.current_step-1] < self.current_power_usage[self.current_step-1]:
        #     reward -= (min(self.power_setpoints[self.current_step-1], self.charge_power_potential[self.current_step-1]) -
        #           self.current_power_usage[self.current_step-1])**2
        # else:
        #     reward -= (min(self.power_setpoints[self.current_step-1], self.charge_power_potential[self.current_step-1]) -
        #           self.current_power_usage[self.current_step-1])

        # for score in user_satisfaction_list:
        #     reward -= 100 * (1 - score)

        # for tr in self.transformers:
        #     if tr.current_amps > tr.max_current:
        #         reward -= 1000 * abs(tr.current_amps - tr.max_current)
        #     elif tr.current_amps < tr.min_current:
        #         reward -= 1000 * abs(tr.current_amps - tr.min_current)

        # reward -= 100 * (tr.current_amps < tr.min_amps)
        #######################################################################################################
        # squared tracking error
        # reward -= (min(self.power_setpoints[self.current_step-1], self.charge_power_potential[self.current_step-1]) -
        #            self.current_power_usage[self.current_step-1])**2

        # best reward so far
        ############################################################################################################
        # if self.power_setpoints[self.current_step-1] < self.current_power_usage[self.current_step-1]:
        #     reward -= (self.current_power_usage[self.current_step-1]-self.power_setpoints[self.current_step-1])

        # reward += self.current_power_usage[self.current_step-1]/75
        ############################################################################################################
        # normalize reward to -1 1
        # reward = reward/1000
        # reward = (100 +reward) / 1000
        # print(f'reward: {reward}')

        # reward -= 2 * (invalid_action_punishment/self.number_of_ports)
        # reward /= 100
        # reward = (100 +reward) / 1000
        # print(f'current_power_usage: {self.current_power_usage[self.current_step-1]}')

        # return reward
