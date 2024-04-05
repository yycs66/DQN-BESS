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
    def __init__(self, next_time_step, cleared_market, cleared_rersource):
        # Data input from WEASLE
        self.step = next_time_step
        self.market = cleared_market
        self.resource = cleared_rersource
        self.rid = cleared_rersource['rid']
        type = self.market['uid'][:5]
        print(type)
        # Configurable options
        self.price_ceiling = 999
        self.price_floor = 0
        
        self.prev_price = self.market["previous"][type]["prices"]["EN"]
        self.prev_avg_price = self.market['history'][type]['prices']['EN'].mean()
        self.prev_max_price = self.market['history'][type]['prices']['EN'].max()
        self.init_soc = self.resource['status'][self.rid]['soc'] if self.resource['status'] is not None and self.rid in self.resource['status'] else 0
        self.init_temp = self.resource['status'][self.rid]['temp'] if self.resource['status'] is not None and self.rid in self.resource['status'] else 0
        self.init_degredation = cleared_rersource['status'][self.rid]['degradation'] if cleared_rersource['status'] is not None and self.rid in cleared_rersource['status'] else 0
        self.schedule = cleared_rersource['schedule'][self.rid]['EN']
        self.profit = cleared_rersource['score'][self.rid]['current'] if cleared_rersource['score'] is not None and self.rid in cleared_rersource['score'] else 0

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
    def step(self,action_battery,cleared_rersource):
        energy_change =  cleared_rersource['schedule'][self.rid]['EN'] # if >0, charge, if <0, discharge
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
    def __init__(self,next_time_step, cleared_market, cleared_rersource,**kwargs):
        super(DAEnv,self).__init__()
        self.next_time_step = next_time_step
        self.cleared_market = cleared_market
        self.cleared_rersource = cleared_rersource
        #parameters 
        self.data_manager=LoadData(next_time_step, cleared_market, cleared_rersource)
        
        self.episode_length=kwargs.get('episode_length',36)
        self.month=None
        self.day=None
        self.TRAIN=True
        self.current_time = cleared_market['uid'][5:] #set current time to the market time
        self.battery=Battery()
       

        # define action space 
        #action space here is = the scalar value we pick for A
        self.action_space=spaces.Box(low=-1,high=1,shape=(1,),dtype=np.float32)# seems here doesn't used 
        # state is [next_time_step,soc_output_last_step, profit_last_step, price_last_step] 
        self.state_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
        # set state related normalization reference
        self.Length_max=36
        self.SOC_max=self.battery.max_soc
        
        
    #prepares the environment for a new episode of the learning process by randomly selecting a month and day,
    # resetting the time and the components of the environment, 
    #and constructing the initial state
    def reset(self):
        '''reset is used for initialize the environment, decide the day of month.'''
        self.data_manager = LoadData(self.next_time_step, self.cleared_market, self.cleared_rersource)

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
        #obs=(next_time_step,price,soc, profit)
        #also can transfer it into normalized state
        soc=self.battery.SOC() # get the soc of the battery 
        next_time_step=self.current_time
        price=self.data_manager.prev_price # get the prediction price
        profit=self.data_manager.profit # get the profit
        obs=np.concatenate((np.float32(next_time_step),np.float32(price),np.float32(soc), np.float32(profit)),axis=None)
        return obs
    def scale_cost_offers(self, action, Agent):
        # Scale the cost offers from the dummy algorithm based on the action
        # The action is a scaling factor in the range [0.1, 10]
        scaling_factor = 0.1 + 9.9 * (action.item() + 1) / 2

        # Get the original cost offers from the dummy algorithm
        market_type = cleared_market['type']
        dummy_offer = Agent.make_me_an_offer()

        if 'DAM' in market_type:
            keys_factor ={'blaoc_ch_mc':scaling_factor,'block_dc_oc':scaling_factor}
            for key, factor in keys_factor.items():
                dummy_offer[key] = dummy_offer[key] * factor
            with open('dummy_offer.json', 'w') as f: #todo: dummy offer's name is needed
                json.dump(dummy_offer, f, cls=NpEncoder)

        else:
            keys_factor ={'blaoc_soc_mc':scaling_factor,'block_ch_oc':scaling_factor}
            for key, factor in keys_factor.items():
                dummy_offer[key] = dummy_offer[key] * factor
            with open('dummy_offer.json', 'w') as f: #todo: dummy offer's name is needed
                json.dump(dummy_offer, f, cls=NpEncoder)

        return dummy_offer # todo: a full offer needed
    
    def step(self,action,cleared_resource):#todo: need track time_step
    # state transition here current_obs--take_action--get reward-- get_finish--next_obs
    ## here we want to put take action into each components
        # Scale the cost offers from the dummy algorithm
        current_obs=self._build_state()
        self.battery.step(action[0])
        # here execute the state-transition part, battery.current_soc also changed
        
        reward= cleared_resource['score']["current"]

        '''here we also need to store the final step outputs for the final steps including, soc, output of units for seeing the final states'''
        final_step_outputs=[self.battery.current_soc]
        if 'DAM' in cleared_market['type']:
            self.current_time = (datetime.datetime.strptime(self.current_time, '%Y%m%d%H%M') + datetime.timedelta(hours=1)).strftime('%Y%m%d%H%M')
        else:
            self.current_time = (datetime.datetime.strptime(self.current_time, '%Y%m%d%H%M') + datetime.timedelta(minutes=5)).strftime('%Y%m%d%H%M')
        finish=(self.current_time==self.episode_length) #todo: maybe time_step is better
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
    parser.add_argument('next_time_step', type=int, help='Integer time step tracking the progress of the\
                        simulated market.')
    parser.add_argument('market_file', help='path to json formatted dictionary with market \
                        information.')
    parser.add_argument('resource_file', help='path to json formatted dictionary with resource \
                        information.')

    args = parser.parse_args()

    # Parse json inputs into python dictionaries
    next_time_step = args.next_time_step
    with open(args.market_file, 'r') as f:
        cleared_market = json.load(f)
    with open(args.resource_file, 'r') as f:
        cleared_rersource = json.load(f)

    # Read in information from the market
    uid = cleared_market["uid"]
    market_type = cleared_market["market_type"]
    rid = cleared_rersource["rid"]
    

    env = DAEnv(next_time_step, cleared_market, cleared_rersource)
    env.TRAIN = False
    rewards = []
    env.reset()
    env.day=27
    tem_action=[0.1]
    finish = False
    while not finish:
        print(f'current month is {env.month}, current day is {env.day}, current time is {env.current_time}')
        current_obs, next_obs, reward, finish = env.step(tem_action,cleared_market,Agent)
        env.render(current_obs, next_obs, reward, finish)
        current_obs = next_obs
        rewards.append(reward)
        if finish:
            break
        
    print(f'total reward{sum(rewards)}')
