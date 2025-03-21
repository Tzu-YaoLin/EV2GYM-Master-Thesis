'''This file contains various example reward functions for the RL agent. Users can create their own reward function here or in their own file using the same structure as below
'''
import numpy as np
import math
import torch
import wandb

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

def profit_maximization(env, total_costs, user_satisfaction_list, actions, invalid_action_punishment, *args):
    reward = total_costs   # Initial reward based on total costs
    
    # Set parameters 
    degradation_cost_factor = 0.01 
    # charge_price_reward_factor = 0.3
    soc_reward_factor = 9 
    
    # User satisfaction reward/penalty
    user_satisfaction_reward = 0
    transformer_load_reward = 0
    degradation_reward = 0
    soc_reward = 0
    
    # positive_charge_prices = env.charge_prices[env.charge_prices < 0]
    # positive_discharge_prices = env.discharge_prices[env.discharge_prices > 0]
    # current_charge_price = env.charge_prices[0, env.current_step - 1]
    # current_discharge_price = env.discharge_prices[0, env.current_step - 1]
    # avg_charge_price = np.mean(positive_charge_prices)
    # avg_discharge_price = np.mean(positive_discharge_prices)
    # charge_price_percentile = np.percentile(positive_charge_prices, 70)
    # discharge_price_percentile = np.percentile(positive_discharge_prices, 70)
    
    
    for score in user_satisfaction_list:
        # user_satisfaction_reward -= 150 * math.exp(-10*score) + 1
        if score < 0.75 :
            user_satisfaction_reward += 12 * (math.exp(10*(score-0.75)) -1)
            # user_satisfaction_reward -= 150 * math.exp(-10*score) + 0.0663
        else:
            user_satisfaction_reward += 10 * math.tanh(50*(score - 0.75)) 
            # user_satisfaction_reward += 70 * math.log(score + 0.25)
            
    reward += user_satisfaction_reward
    
    # Transformer load reward/penalty
    for tr in env.transformers:
        # transformer_load_reward -= 100 * tr.get_how_overloaded() 
        overload_ratio = tr.get_how_overloaded() / tr.max_power[env.current_step - 1]
        transformer_load_reward -= 10 * overload_ratio 
            
    reward += transformer_load_reward
    
    if env.current_evs_parked > 0:
        # for action in actions:
        #     if action > 0:  # Charging price reward/penalty
        #         if current_charge_price > charge_price_percentile:
        #             reward += abs(abs(action) * ((current_charge_price - charge_price_percentile) / avg_charge_price) * charge_price_reward_factor)
        #         elif current_charge_price > avg_charge_price:
        #             reward += abs(abs(action) * ((current_charge_price - avg_charge_price) / avg_charge_price) * charge_price_reward_factor * 0.5)
        #     elif action < 0:
        #         if current_discharge_price > discharge_price_percentile:
        #             reward += abs(action) * ((current_discharge_price - discharge_price_percentile) / avg_discharge_price) * charge_price_reward_factor
                # elif current_discharge_price > avg_discharge_price:
                #     reward += abs(action) * ((current_discharge_price - avg_discharge_price) / avg_discharge_price) * charge_price_reward_factor * 0.5
                    
        # Calculate degradation cost
        for ev in env.EVs:
            if len(ev.historic_soc) > 1:
                previous_soc = ev.historic_soc[-2]
            else:
                previous_soc = ev.historic_soc[-1] if len(ev.historic_soc) > 0 else ev.get_soc()
                
            if ev.get_soc() > previous_soc:
                soc_reward = soc_reward_factor * (ev.get_soc() - previous_soc)
            else:
                soc_reward = 0

            if ev.total_degradation < 0.1:
                degradation_reward = -1 * ev.charging_cycles * degradation_cost_factor 
            elif ev.total_degradation < 0.2:
                degradation_reward = -1 * ev.charging_cycles * degradation_cost_factor * 1.5
            else:
                degradation_reward = -1 * ev.charging_cycles * degradation_cost_factor * 2
            
        reward += degradation_reward + soc_reward
    
    
    # Construct reward contributions
    reward_contributions = {
        "degradation_reward": degradation_reward,
        "user_satisfaction_reward": user_satisfaction_reward,
        "transformer_load_reward": transformer_load_reward,
        "total_costs": total_costs,
        "soc_reward": soc_reward,
    }
    
    return reward, reward_contributions


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
