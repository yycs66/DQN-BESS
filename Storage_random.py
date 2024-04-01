# ------------------------------------------------------------------------
# Energy storage environment for reinforcement learning agents developed by
# Liping Li, PNNL, liping.li@pnnl.gov
# ------------------------------------------------------------------------
import random
import numpy as np

import pandas as pd 
import gym
from gym import spaces 
import math 
import os 
import sys
import json
import argparse
from dummy_offer import Agent
import datetime



class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
class Constant:
    MAX_STEP_HOURS = 24 * 30
    MAX_STEP_RT=288
    socmax = 608
    socmin = 128
    chmax = 125
    dcmax = 125
    efficiency = 0.892
    capacity = 500,# kw

    degradation=0, #euro/kw
    initial_capacity=128

class LoadData():
    def __init__(self, time_step, market_info, resource_info):
        # Data input from WEASLE
        self.step = time_step
        self.market = market_info
        self.resource = resource_info
        self.rid = resource_info['rid']
        type = self.market['uid'][:5]
        print(type)
        # Configurable options
        self.price_ceiling = 999
        self.price_floor = 0
        
        self.prev_price =self.market["previous"][type]["prices"]["EN"]
        self.prev_avg_price = self.market['history']['prices']['EN'].mean()
        self.prev_max_price = self.market['history']['prices']['EN'].max()
        self.init_soc = self.resource['status'][self.rid]['soc'] if self.resource['status'] is not None and self.rid in self.resource['status'] else 0
        self.init_temp = self.resource['status'][self.rid]['temp'] if self.resource['status'] is not None and self.rid in self.resource['status'] else 0
        self.init_degredation = resource_info['status'][self.rid]['degradation'] if resource_info['status'] is not None and self.rid in resource_info['status'] else 0
        self.schedule = resource_info['schedule'][self.rid]['EN']
        self.profit = resource_info['score'][self.rid]['current'] if resource_info['score'] is not None and self.rid in resource_info['score'] else 0

class Battery():
    '''simulate a simple battery here'''
    def __init__(self):
        self.capacity=Constant.capacity# 500kw
        self.max_soc=Constant.socmax
        self.capacity=Constant.socmax
        self.initial_capacity=Constant.initial_capacity
        self.min_soc=Constant.min_soc
        self.degradation=Constant.degradation# degradation cost 0，
        self.max_charge=Constant.chmax# max charge ability
        self.max_discharge=Constant.dcmax# max discharge ability
        self.efficiency=Constant.efficiency# charge and discharge efficiency
    #action_battery: the scalar value we pick for A
    def step(self,action_battery,resource_info):
        energy_change =  resource_info['schedule'][self.rid]['EN'] # if >0, charge, if <0, discharge
        updated_soc=max(self.min_soc,min(self.max_soc,self.current_soc+energy_change))
        self.current_soc=updated_soc# update capacity to current codition
    def _get_cost(self,energy):# calculate the cost depends on the degration
        cost=energy**2*self.degradation
        return cost  
    def SOC(self):
        return self.current_soc
    def reset(self):
        self.current_soc = np.random.uniform(128, 698) #inital SoC randomly
        self.energy_change=0
        self.total_reward = 0
  
 
class DAEnv(gym.Env):
    '''ENV descirption: 
    the agent learn to charge with low price and then discharge at high price, in this way, it could get benefits'''
    def __init__(self,time_step, market_info, resource_info,**kwargs):
        super(DAEnv,self).__init__()
        self.time_step = time_step
        self.market_info = market_info
        self.resource_info = resource_info
        #parameters 
        self.data_manager=LoadData(time_step, market_info, resource_info)
        
        self.episode_length=kwargs.get('episode_length',36)
        self.month=None
        self.day=None
        self.TRAIN=True
        self.current_time = market_info['uid'][5:] #set current time to the market time
        self.battery=Battery()
       

        # define action space 
        #action space here is = the scalar value we pick for A
        self.action_space=spaces.Box(low=-1,high=1,shape=(1,),dtype=np.float32)# seems here doesn't used 
        # state is [time_step,soc_output_last_step, profit_last_step, price_last_step] 
        self.state_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
        # set state related normalization reference
        self.Length_max=36
        self.SOC_max=self.battery.max_soc
        
        
    #prepares the environment for a new episode of the learning process by randomly selecting a month and day,
    # resetting the time and the components of the environment, 
    #and constructing the initial state
    def reset(self):
        '''reset is used for initialize the environment, decide the day of month.'''
        self.data_manager = LoadData(self.time_step, self.market_info, self.resource_info)

        if self.TRAIN:
            self.day=np.random.randint(1,21)
        else:
            #max_day = Constant.MONTHS_LEN[self.month-1]
            max_day = 30
            self.day=np.random.randint(21, max_day + 1)  # Ensure day is within valid range
        self.current_time=0
        self.battery.reset()
        self.total_reward = 0
        return self._build_state()
    
    def _build_state(self):
        #obs=(time_step,price,soc, profit)
        #also can transfer it into normalized state
        soc=self.battery.SOC() # get the soc of the battery 
        time_step=self.current_time
        price=self.data_manager.prev_pricee # get the prediction price
        profit=self.data_manager.profit # get the profit
        obs=np.concatenate((np.float32(time_step),np.float32(price),np.float32(soc), np.float32(profit)),axis=None)
        return obs
    def scale_cost_offers(self, action, market_info, Agent):
        # Scale the cost offers from the dummy algorithm based on the action
        # The action is a scaling factor in the range [0.1, 10]
        scaling_factor = 0.1 + 9.9 * (action.item() + 1) / 2

        # Get the original cost offers from the dummy algorithm
        market_type = market_info['type']
        if 'DAM' in market_type:
            original_charge_cost = Agent._day_ahead_offer()['block_ch_mc']
            original_discharge_cost = Agent._day_ahead_offer()['block_dc_mc']
            # Scale the cost offers
            charge_cost = original_charge_cost * scaling_factor
            discharge_cost = original_discharge_cost * scaling_factor
            soc_oc = None

        else:
            original_soc_oc = Agent._real_time_offer()['block_soc_oc']
            soc_oc = original_soc_oc * scaling_factor
            charge_cost = None
            discharge_cost = None

        return charge_cost, discharge_cost, soc_oc
    
    def step(self,action,market_info,Agent):
    # state transition here current_obs--take_action--get reward-- get_finish--next_obs
    ## here we want to put take action into each components
        # Scale the cost offers from the dummy algorithm
        current_obs=self._build_state()
        self.battery.step(action[0])
        # here execute the state-transition part, battery.current_soc also changed
        
        current_output=np.array((-self.battery.energy_change))#truely corresonding to the result
        self.current_output=current_output
        
        price=current_obs[1]
        charge_cost, discharge_cost, soc_oc = self.scale_cost_offers(action, market_info, Agent)    
        reward=0
        
        battery_cost=self.battery._get_cost(self.battery.energy_change)# we set it as 0 this time 
        
        if self.battery.energy_change < 0: # discharge
            reward = -(battery_cost + price*self.battery.energy_change*self.battery.efficiency)
        else:
            reward = -(battery_cost + price*self.battery.energy_change)
        self.total_reward += reward

        '''here we also need to store the final step outputs for the final steps including, soc, output of units for seeing the final states'''
        final_step_outputs=[self.battery.current_soc]
        if 'DAM' in market_info['type']:
            self.current_time = (datetime.datetime.strptime(self.current_time, '%Y%m%d%H%M') + datetime.timedelta(hours=1)).strftime('%Y%m%d%H%M')
        else:
            self.current_time = (datetime.datetime.strptime(self.current_time, '%Y%m%d%H%M') + datetime.timedelta(minutes=5)).strftime('%Y%m%d%H%M')
        finish=(self.current_time==self.episode_length)
        if finish:
            self.final_step_outputs=final_step_outputs
            self.current_time=0
            next_obs=self.reset()
            
        else:
            next_obs=self._build_state()
        return current_obs,next_obs,float(reward),finish
    def render(self, current_obs, next_obs, reward, finish):
        print('day={},hour={:2d}, state={}, next_state={}, reward={:.4f}, terminal={}\n'.format(self.day,self.current_time, current_obs, next_obs, reward, finish))
        
    ## test environment
if __name__ == '__main__':
    # Add argument parser for three required input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('time_step', type=int, help='Integer time step tracking the progress of the\
                        simulated market.')
    parser.add_argument('market_file', help='path to json formatted dictionary with market \
                        information.')
    parser.add_argument('resource_file', help='path to json formatted dictionary with resource \
                        information.')

    args = parser.parse_args()

    # Parse json inputs into python dictionaries
    time_step = args.time_step
    with open(args.market_file, 'r') as f:
        market_info = json.load(f)
    with open(args.resource_file, 'r') as f:
        resource_info = json.load(f)

    # Read in information from the market
    uid = market_info["uid"]
    market_type = market_info["market_type"]
    rid = resource_info["rid"]
    

    env = DAEnv(time_step, market_info, resource_info)
    env.TRAIN = False
    rewards = []
    env.reset()
    env.day=27
    tem_action=[0.1]
    finish = False
    while not finish:
        print(f'current month is {env.month}, current day is {env.day}, current time is {env.current_time}')
        current_obs, next_obs, reward, finish = env.step(tem_action,market_info,Agent)
        env.render(current_obs, next_obs, reward, finish)
        current_obs = next_obs
        rewards.append(reward)
        if finish:
            break
        
    print(f'total reward{sum(rewards)}')
