'''
This file contains the EV_Charger class, which is used to represent the EV chargers in the environment.
'''

import numpy as np
import math
import torch
from .ev import EV



class EV_Charger:
    '''
    Attributes:
        - id: unique identifier of the EV charger
        - connected_bus: the bus to which the EV charger is connected
        - connected_transformer: the transformer(s) to which the EV charger is connected
        - geo_location: the geographical location of the EV charger                
        - n_ports: the number of ports of the EV charger
        - charger_type: the type of the EV charger (typ1, type2, or DC)
        - bi_directional: whether the EV charger can provide power to the grid or not
        - timescale: the timescale of the simulation (useful for determining the charging speed)
        - verbose: whether to print information about the EV charger or not

    Status variables:
        - current_power_output: the current total power output of the EV charger (positive for draining energy from the grid, negative for providing energy to the grid)
        - evs_connected: the list of EVs connected to the EV charger
        - n_ev_connected: the current number of EVs connected to the EV charger
        - current_step: the current simulation timestep

    Statistics variables:
        - total_energy_charged: the total energy charged by the EV charger  
        - total_energy_discharged: the total energy discharged by the EV charger
        - total_profits: the total profit of the EV charger
        - total_evs_served: the total number of EVs served by the EV charger
        - total_user_satisfaction: the total user satisfaction of the EV charger

    Methods:
        - step: updates the EV charger status according to the actions taken by the EVs
        - reset: resets the EV charger status to the initial state      

'''

    def __init__(self,
                 id,
                 connected_bus,
                 connected_transformer,
                 geo_location=None,
                 min_charge_current=0,  # Amperes
                 max_charge_current=56,  # Amperes
                 min_discharge_current=0,  # Amperes
                 max_discharge_current=-56,  # Amperes
                 voltage=230,  # Volts
                 n_ports=2,
                 charger_type="AC",  # AC or DC
                 phases=3,
                 timescale=5,
                 verbose=False):

        self.id = id

        # EV Charger location and grid characteristics
        self.connected_bus = connected_bus
        self.connected_transformer = connected_transformer
        self.geo_location = geo_location

        # EV Charger technical characteristics
        self.n_ports = n_ports
        self.charger_type = charger_type
        self.timescale = timescale

        self.min_charge_current = min_charge_current
        self.max_charge_current = max_charge_current
        self.min_discharge_current = min_discharge_current
        self.max_discharge_current = max_discharge_current
        self.phases = phases

        self.voltage = voltage

        # EV Charger status
        self.current_power_output = 0
        self.evs_connected = [None] * n_ports
        self.n_evs_connected = 0
        self.current_step = 0
        self.current_charge_price = 0
        self.current_discharge_price = 0
        self.current_total_amps = 0
        self.current_signal = [0]*n_ports

        # EV Charger Statistics
        self.total_energy_charged = 0
        self.total_energy_discharged = 0
        self.total_profits = 0
        self.total_evs_served = 0
        self.total_user_satisfaction = 0

        self.verbose = verbose


    def step(self, actions, charge_price, discharge_price):
        # 確保 actions 是 Tensor 並進行複製
        actions = torch.tensor(actions) if not isinstance(actions, torch.Tensor) else actions.clone()
        
        # 檢查 NaN 值並報錯
        if torch.isnan(actions).any():
            raise ValueError(f"Actions contain NaN values: {actions}")
        
        profit = 0
        user_satisfaction = []
        self.current_power_output = 0
        self.current_total_amps = 0
        self.current_charge_price = charge_price
        self.current_discharge_price = discharge_price
        self.current_signal = []
        
        assert len(actions) == self.n_ports, f"Expected {self.n_ports} actions, got {len(actions)}"
        invalid_action_punishment = 0
    
        # 如果沒有 EV 連接，則將該位置的 action 設置為 0
        for i in range(len(actions)):
            if self.evs_connected[i] is None:
                actions[i] = 0
                invalid_action_punishment += 1
        
        # 計算 action 的總和，並避免除零
        total_action_sum = torch.sum(actions).item()
        
        if total_action_sum > 1:
            normalized_actions = actions / total_action_sum
        elif total_action_sum < -1:
            normalized_actions = -actions / total_action_sum
        else:
            normalized_actions = actions  # 若總和在 [-1, 1] 內，則不需要歸一化
        
        if self.verbose:
            print(f'CS {self.id} normalized actions: {normalized_actions}')
            # 接下來的代碼不變，繼續處理 normalized_actions
            
        # Update EVs connected to the EV charger and get profits/costs
        for i, action in enumerate(normalized_actions):
            actual_energy = 0
            # 使用torch.round
            action = torch.round(action * 1e5) / 1e5  # 保留小數點後五位
            # 確保所有的 action 都在範圍內
            assert torch.all((action >= -1) & (action <= 1)), f'Action {action} is not in range [-1,1]'
            amps = 0
            if torch.all(action == 0) and self.evs_connected[i] is not None:

                actual_energy, actual_amps = self.evs_connected[i].step(
                    amps, self.voltage)

            elif (action > 0).any():
                amps = action * self.max_charge_current
                if amps < self.min_charge_current-0.01:
                    amps = 0

                actual_energy, actual_amps = self.evs_connected[i].step(
                    amps,
                    self.voltage,
                    phases=self.phases,
                    type=self.charger_type)

                profit += abs(actual_energy) * charge_price
                self.total_energy_charged += abs(actual_energy)
                self.current_power_output += actual_energy * 60/self.timescale
                self.current_total_amps += actual_amps

            elif (action < 0).any():
                amps = action * abs(self.max_discharge_current)
                if amps > self.min_discharge_current-0.01:
                    amps = self.min_discharge_current

                actual_energy, actual_amps = self.evs_connected[i].step(
                    amps,
                    self.voltage,
                    phases=self.phases,
                    type=self.charger_type)

                profit += abs(actual_energy) * discharge_price
                self.total_energy_discharged += abs(actual_energy)
                self.current_power_output += actual_energy * 60/self.timescale
                self.current_total_amps += actual_amps

            # print(f'CS {self.id} port {i} action {action} amps {amps} energy {actual_energy} total_amps {self.current_total_amps}')
            self.current_signal.append(amps)

            if self.current_total_amps - 0.0001 > self.max_charge_current:
                raise Exception(
                    f'sum of amps {self.current_total_amps} is higher than max charge current {self.max_charge_current}')

        self.total_profits += profit

        departing_evs = []
        # Check if EVs are departing
        for i, ev in enumerate(self.evs_connected):
            if ev is not None:
                if ev.is_departing(self.current_step) is not None:
                    # calculate battery degradation for departing EV
                    self.evs_connected[i] = None
                    self.n_evs_connected -= 1
                    self.total_evs_served += 1
                    ev_user_satisfaction = ev.get_user_satisfaction()
                    self.total_user_satisfaction += ev_user_satisfaction
                    user_satisfaction.append(ev_user_satisfaction)
                    departing_evs.append(ev)
                    if self.verbose:
                        print(f'- EV {ev.id} is departing from CS {self.id}' +
                              f' port {i}'
                              f' with user satisfaction {ev_user_satisfaction}' +
                              f' (SoC: {ev.get_soc()*100: 6.1f}%)')

        self.current_step += 1

        return profit, user_satisfaction, invalid_action_punishment, departing_evs
# %%



    def __str__(self) -> str:

        if self.total_evs_served == 0:
            user_satisfaction_str = ' Avg. Sat.:  - '
        else:
            user_satisfaction_str = f' Avg. Sat.: {self.get_avg_user_satisfaction()*100: 3.1f}%'

        # f' ({self.current_step*self.timescale: 3d} mins)' + \
        return f'CS{self.id:3d}: ' + \
            f' Served {self.total_evs_served:4d} EVs' + \
            user_satisfaction_str + \
            f' in {self.current_step: 4d} steps' + \
            f' |{self.total_profits: 7.1f} € |' + \
            f' +{self.total_energy_charged: 5.1f} /' + \
            f' -{self.total_energy_discharged: 5.1f} kWh'

    def get_max_power(self):
        return self.max_charge_current * self.voltage * math.sqrt(self.phases) / 1000

    def get_min_charge_power(self):
        return self.min_charge_current * self.voltage * math.sqrt(self.phases) / 1000

    def get_min_power(self):
        return self.max_discharge_current * self.voltage * math.sqrt(self.phases) / 1000

    def get_avg_user_satisfaction(self):
        if self.total_evs_served == 0:
            return 0
        else:
            return self.total_user_satisfaction / self.total_evs_served

    def spawn_ev(self, ev):
        '''Adds an EV to the list of EVs connected to the EV charger
        Inputs:
            - ev: the EV to be added to the list of EVs connected to the EV charger
        '''
        assert (self.n_evs_connected < self.n_ports)

        index = self.evs_connected.index(None)
        ev.id = index
        self.evs_connected[index] = ev
        self.n_evs_connected += 1
        
        
        if self.verbose:
            print(f'+ EV connected to Charger {self.id} at port {index}' +
                  f' leaving at {ev.time_of_departure}' +
                  f' SoC {ev.get_soc()*100:.1f}%')

        return index

    def reset(self):
        '''Resets the EV charger status to the initial state'''

        # EV Charger status
        self.current_power_output = 0
        self.current_total_amps = 0
        self.evs_connected = [None] * self.n_ports
        self.n_evs_connected = 0
        self.current_step = 0

        # EV Charger Statistics
        self.total_energy_charged = 0
        self.total_energy_discharged = 0
        self.total_profits = 0
        self.total_evs_served = 0
        self.total_user_satisfaction = 0
