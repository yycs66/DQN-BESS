<<<<<<< HEAD
# ------------------------------------------------------------------------

# A copy of BESS_DQN.py which implemnts the ddpg for bess
# ------------------------------------------------------------------------
import pickle
import torch

import os
import numpy as np
import numpy.random as rd
import pandas as pd
from typing import Tuple
import pyomo.environ as pyo
import pyomo.kernel as pmo
from omlt import OmltBlock
import json
import offer_utils as ou
import argparse
from dummy_offer import Agent


import tempfile 
# for creating temporary files and directories
import torch.onnx 
#uitiities for converting torch.nn module to onnx
import torch.nn as nn
from copy import deepcopy
import wandb
#Weights & Biases, a tool for tracking and visualizing machine learning experiments
from Storage_random import DAEnv
#torch.cuda.is_available()
=======
# This is a test dummy algorithm to get the opportunity cost curves
from ortools.linear_solver import pywraplp
import pandas as pd
import numpy as np
import argparse
import json
import datetime

>>>>>>> a572ce81a4fba6669fedb6dde2510e2f8269e39c
# Standard battery parameters
socmax = 608
socmin = 128
chmax = 125
dcmax = 125
efficiency = 0.892
<<<<<<< HEAD
=======

>>>>>>> a572ce81a4fba6669fedb6dde2510e2f8269e39c
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
<<<<<<< HEAD
    
## define net
class ReplayBuffer:
    def __init__(self, max_len, state_dim, action_dim, gpu_id=0):
        self.now_len = 0
        self.next_idx = 0
        self.if_full = False
        self.max_len = max_len
        self.data_type = torch.float32
        self.action_dim = action_dim
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id > 0)) else "cpu")

        other_dim = 1 + 1 + self.action_dim
        ##to do: check why we need to add 1+1+action_dim
        self.buf_other = torch.empty(size=(max_len, other_dim), dtype=self.data_type, device=self.device)

        if isinstance(state_dim, int):  
            self.buf_state = torch.empty((max_len, state_dim), dtype=torch.float32, device=self.device)
        elif isinstance(state_dim, tuple):
            self.buf_state = torch.empty((max_len, *state_dim), dtype=torch.uint8, device=self.device)
        else:
            raise ValueError('state_dim')

    def extend_buffer(self, state, other):  # add new state and other/action to the buffer
        size = len(other)
        next_idx = self.next_idx + size

        if next_idx > self.max_len:
            self.buf_state[self.next_idx:self.max_len] = state[:self.max_len - self.next_idx]
            self.buf_other[self.next_idx:self.max_len] = other[:self.max_len - self.next_idx]
            self.if_full = True

            next_idx = next_idx - self.max_len
            self.buf_state[0:next_idx] = state[-next_idx:]
            self.buf_other[0:next_idx] = other[-next_idx:]
        else:
            self.buf_state[self.next_idx:next_idx] = state
            self.buf_other[self.next_idx:next_idx] = other
        self.next_idx = next_idx
      

    def sample_batch(self, batch_size) -> tuple:
        indices = rd.randint(self.now_len - 1, size=batch_size)
        r_m_a = self.buf_other[indices]
        #return first three columns of r_m_a which has shape of (batch_size,1), and the state and next state
        return (r_m_a[:, 0:1],
                r_m_a[:, 1:2],
                r_m_a[:, 2:],
                self.buf_state[indices],
                self.buf_state[indices + 1])

    def update_now_len(self):
        self.now_len = self.max_len if self.if_full else self.next_idx
class Arguments:
    def __init__(self, agent=None, env=None ):

        self.agent = agent  # Deep Reinforcement Learning algorithm
        self.env = env  # the environment for training
        self.cwd = None  # current work directory. None means set automatically
        self.if_remove = False  # remove the cwd folder? (True, False, None:ask me)
        self.visible_gpu = '0,1,2,3'  # for example: os.environ['CUDA_VISIBLE_DEVICES'] = '0, 2,'
        

        self.if_per_or_gae = False  # or any other default value you want
        '''Arguments for training'''
        self.num_episode=3000
        self.gamma = 0.995  # discount factor of future rewards
        self.learning_rate = 1e-4  # 2 ** -14 ~= 6e-5
        self.soft_update_tau = 1e-2  # 2 ** -8 ~= 5e-3

        self.net_dim = 64  # the network width 256
        self.batch_size = 256  # num of transitions sampled from replay buffer.
        self.repeat_times = 2 ** 3  # repeatedly update network to keep critic's loss small
        self.target_step = 1000 # collect target_step experiences , then update network, 1024
        self.max_memo = 50000  # capacity of replay buffer
        ## arguments for controlling exploration
        self.explorate_decay=0.99
        self.explorate_min=0.3
        '''Arguments for evaluate'''
        self.random_seed_list=[1234,2234,3234,4234,5234]
        # self.random_seed_list=[2234]
        self.run_name='Battery_DQN_experiments'
        '''Arguments for save'''
        self.train=True
        self.save_network=True

    def init_before_training(self, if_main):
        if self.cwd is None:
            agent_name = self.agent.__class__.__name__
            self.cwd = f'./{agent_name}/{self.run_name}'

        if if_main:
            import shutil  # remove history according to bool(if_remove)
            # Remove the existing cwd directory
            if os.path.exists(self.cwd):
                shutil.rmtree(self.cwd, ignore_errors=True)
                print(f"| Removed existing cwd: {self.cwd}")
            # Create a new cwd directory
            os.makedirs(self.cwd, exist_ok=True)

        #np.random.seed(self.random_seed)
        #torch.manual_seed(self.random_seed)
        torch.set_default_dtype(torch.float32)

        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.visible_gpu)# control how many GPU is used 　
class Actor(nn.Module):
    def __init__(self,mid_dim,state_dim,action_dim):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(state_dim,mid_dim),nn.ReLU(),
                               nn.Linear(mid_dim,mid_dim),nn.ReLU(),
                               nn.Linear(mid_dim,mid_dim),nn.ReLU(),
                               nn.Linear(mid_dim,action_dim))
    def forward(self,state):
        return self.net(state).tanh()# make the data from -1 to 1
    def get_action(self,state,action_std):#
        action=self.net(state).tanh()
        noise=(torch.randn_like(action)*action_std).clamp(-0.5,0.5)#
        return (action+noise).clamp(-1.0,1.0)
class CriticQ(nn.Module):
    def __init__(self,mid_dim,state_dim,action_dim):
        super().__init__()
        self.net_head=nn.Sequential(nn.Linear(state_dim+action_dim,mid_dim),nn.ReLU(),
                                    nn.Linear(mid_dim,mid_dim),nn.ReLU())
        self.net_q1=nn.Sequential(nn.Linear(mid_dim,mid_dim),nn.ReLU(),
                                  nn.Linear(mid_dim,1))# we get q1 value
        self.net_q2=nn.Sequential(nn.Linear(mid_dim,mid_dim),nn.ReLU(),
                                  nn.Linear(mid_dim,1))# we get q2 value
    def forward(self,value):
        mid=self.net_head(value)
        return self.net_q1(mid)
    def get_q1_q2(self,value):
        mid=self.net_head(value)
        return self.net_q1(mid),self.net_q2(mid)
class AgentBase:
=======


class Agent():
>>>>>>> a572ce81a4fba6669fedb6dde2510e2f8269e39c
    '''
    Agent is re-initialized every time the WEASLE Platform calls market_participant.py
    Input: time_step, market_data, and resource_data are input arguments from the script call
    Additional input data must be saved to disc and reloaded each time Agent is created (e.g., to facilitate Agent persistence)
    Output:
    - make_me_an_offer() reads the market type and saves to disc a JSON file containing offer data
    '''
<<<<<<< HEAD
    def __init__(self, time_step, market_info, resource_info):
        
=======

    def __init__(self, time_step, market_info, resource_info):
>>>>>>> a572ce81a4fba6669fedb6dde2510e2f8269e39c
        # Data input from WEASLE
        self.step = time_step
        self.market = market_info
        self.resource = resource_info
        self.rid = resource_info['rid']

        # Standard battery parameters
        self.socmax = socmax
        self.socmin = socmin
        self.chmax = chmax
        self.dcmax = dcmax
        self.efficiency = efficiency

        # Configurable options
        self.price_ceiling = 999
        self.price_floor = 0

<<<<<<< HEAD
        

        #agent parameters
        self.state = None
        self.device = None
        self.action_dim = None
        self.if_off_policy = None
        self.explore_noise = None
        self.trajectory_list = None
        self.explore_rate = 1.0

        self.criterion = torch.nn.SmoothL1Loss()

    def init(self, net_dim, state_dim, action_dim, learning_rate=1e-4, _if_per_or_gae=False, gpu_id=0):
        self.device = torch.device(
            f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id > 0)) else "cpu")
        self.action_dim = action_dim

        self.cri = self.ClassCri(net_dim, state_dim, action_dim).to(self.device)
        self.act = self.ClassAct(net_dim, state_dim, action_dim).to(
            self.device) if self.ClassAct else self.cri
        self.cri_target = deepcopy(self.cri) if self.if_use_cri_target else self.cri
        self.act_target = deepcopy(self.act) if self.if_use_act_target else self.act

        self.cri_optim = torch.optim.Adam(self.cri.parameters(), learning_rate)
        self.act_optim = torch.optim.Adam(self.act.parameters(),
                                          learning_rate) if self.ClassAct else self.cri
        del self.ClassCri, self.ClassAct

    def select_action(self, state) -> np.ndarray:
        states = torch.as_tensor((state,), dtype=torch.float32, device=self.device)
        action = self.act(states)[0]
        if rd.rand()<self.explore_rate:
            action = (action + torch.randn_like(action) * self.explore_noise).clamp(-1, 1)
        return action.detach().cpu().numpy()

    def explore_env(self, env, target_step):
        trajectory = list()

        state = self.state
        for _ in range(target_step):
            action = self.select_action(state)

            state, next_state, reward, done, = env.step(action)

            trajectory.append((state, (reward, done, *action))) #trajectory is a list of tuples, each tuple contains (state,(reward,done,action))
            state = env.reset() if done else next_state
        self.state = state
        return trajectory

    @staticmethod
    def optim_update(optimizer, objective):
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()

    @staticmethod
    def soft_update(target_net, current_net, tau):
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data * tau + tar.data * (1.0 - tau))

    def save_or_load_agent(self, cwd, if_save):
        def load_torch_file(model_or_optim, _path):
            state_dict = torch.load(_path, map_location=lambda storage, loc: storage)
            model_or_optim.load_state_dict(state_dict)

        name_obj_list = [('actor', self.act), ('act_target', self.act_target), ('act_optim', self.act_optim),
                         ('critic', self.cri), ('cri_target', self.cri_target), ('cri_optim', self.cri_optim), ]
        name_obj_list = [(name, obj) for name, obj in name_obj_list if obj is not None]
        if if_save:
            for name, obj in name_obj_list:
                save_path = f"{cwd}/{name}.pth"
                torch.save(obj.state_dict(), save_path)
        else:
            for name, obj in name_obj_list:
                save_path = f"{cwd}/{name}.pth"
                load_torch_file(obj, save_path) if os.path.isfile(save_path) else None

    def _update_exploration_rate(self,explorate_decay,explore_rate_min):
        self.explore_rate = max(self.explore_rate * explorate_decay, explore_rate_min)
        '''this function is used to update the explorate probability when select action'''
class AgentDA(AgentBase):
    def __init__(self, time_step, market_info, resource_info):
        super().__init__(time_step, market_info, resource_info)
        self.explore_noise = 0.5  # standard deviation of exploration noise
        self.policy_noise = 0.2  # standard deviation of policy noise
        self.update_freq = 2  # delay update frequency
        self.if_use_cri_target = self.if_use_act_target = True
        self.ClassCri = CriticQ
        self.ClassAct = Actor

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau, state_dim) -> tuple:
        buffer.update_now_len()
        obj_critic = obj_actor = None
        for update_c in range(int(buffer.now_len / batch_size * repeat_times)):# we update too much time?
            obj_critic, state = self.get_obj_critic(buffer, state_dim, batch_size)
            self.optim_update(self.cri_optim, obj_critic)

            action_pg = self.act(state)  # policy gradient
            obj_actor = -self.cri_target(torch.cat((state, action_pg),dim=-1)).mean()  # use cri_target instead of cri for stable training
            self.optim_update(self.act_optim, obj_actor)
            if update_c % self.update_freq == 0:  # delay update
                self.soft_update(self.cri_target, self.cri, soft_update_tau)
                self.soft_update(self.act_target, self.act, soft_update_tau)
        return obj_critic.item() / 2, obj_actor.item()

    def get_obj_critic(self, buffer, state_dim,batch_size) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            batch_data = buffer.sample_batch(batch_size)
            reward, mask, action, state, next_s, market_data = (batch_data[:, 0], batch_data[:, 1], batch_data[:, 2:2+self.action_dim],
                                                                batch_data[:, 2+self.action_dim:2+self.action_dim+state_dim],
                                                                batch_data[:, 2+self.action_dim+state_dim],
                                                                batch_data[:, 2+self.action_dim+state_dim+next_s.shape[1]:],
                                                                batch_data[: -market_data.shape[1]:])
            next_a = self.act_target.get_action(next_s, self.policy_noise)  # policy noise,
            next_q = torch.min(*self.cri_target.get_q1_q2(torch.cat((next_s, next_a),dim=-1)))  # twin critics
            q_label = reward + mask * next_q

        q1, q2 = self.cri.get_q1_q2(torch.cat((state, action),dim=-1))
        obj_critic = self.criterion(q1, q_label) + self.criterion(q2, q_label)  # twin critics
        return obj_critic, state



def update_buffer(_trajectory):
    ten_state = torch.as_tensor([item[0] for item in _trajectory], dtype=torch.float32) #trajectory: (state, (reward, mask, action))
    ary_other = torch.as_tensor([item[1] for item in _trajectory]) #second column of trajectory: all other information, like reward, done flag, action
    ary_other[:, 0] = ary_other[:, 0]   # get the reward tensor
    ary_other[:, 1] = (1.0 - ary_other[:, 1]) * gamma  # ten_mask = (1.0 - ary_done) * gamma

    buffer.extend_buffer(ten_state, ary_other)

    _steps = ten_state.shape[0]
    _r_exp = ary_other[:, 0].sum()  # sum of rewards in an episode
    return _steps, _r_exp


def get_episode_return(env, act, device):
    '''get information of one episode during the training'''
    episode_return = 0.0  # sum of rewards in an episode
    state = env.reset()
    # in each episode, there are 36 iterations
    for i in range(36):
        s_tensor = torch.as_tensor((state,), device=device) # convert a current state to tensor
        a_tensor = act(s_tensor) #call the actor network to get the action tensor
        action = a_tensor.detach().cpu().numpy()[0]  # get action variable by detaching the tensor and converting to numpy
        state, next_state, reward, done,= env.step(market_info,Agent)
        state=next_state
        episode_return += reward
        
        if done:
            break
    return episode_return

def train_agent(time_step, market_info, resource_info):
    # Create an instance of the AgentDA class
    agent = AgentDA(time_step, market_info, resource_info)
    env = DAEnv(time_step, market_info, resource_info)
    agent.init(args.net_dim, env.state_space.shape[0], env.action_space.shape[0], args.learning_rate, args.if_per_or_gae)

    # Initialize the replay buffer
    buffer = ReplayBuffer(max_len=args.max_memo, state_dim=env.state_space.shape[0], action_dim=env.action_space.shape[0])

    # Collect initial data and train the agent
    agent.state = env.reset()
    trajectory = agent.explore_env(env, args.target_step)
    update_buffer(trajectory)

    # Train the agent for args.num_episode episodes
    for i_episode in range(args.num_episode):
        critic_loss, actor_loss = agent.update_net(buffer, args.batch_size, args.repeat_times, args.soft_update_tau)
        # Log or record the losses, rewards, and other metrics as needed

        with torch.no_grad():
            episode_reward = get_episode_return(env, agent.act, agent.device)
            # Log or record the episode reward, unbalance, and operation cost as needed

        print(f'current episode is {i_episode}, reward:{episode_reward}, buffer_length: {buffer.now_len}')

        if i_episode % 10 == 0:
            agent._update_exploration_rate(args.explorate_decay, args.explorate_min)
            trajectory = agent.explore_env(env, args.target_step)
            update_buffer(trajectory)

    # Save the trained agent's parameters if needed
    if args.save_network:
        act_save_path = f'{args.cwd}/actor.pth'
        cri_save_path = f'{args.cwd}/critic.pth'
        torch.save(agent.act.state_dict(), act_save_path)
        torch.save(agent.cri.state_dict(), cri_save_path)
        print('Training finished and actor and critic parameters have been saved')
=======
        self._prev_dam_file = 'prev_day_ahead_market'
        self.save_from_previous()

    def make_me_an_offer(self):
        # Read in information from the market
        market_type = self.market["market_type"]
        if market_type == 'DAM':
            offer = self._day_ahead_offer()
        elif market_type == 'RTM':
            offer = self._real_time_offer()

        #TODO: check if we need to clean up offers into maximum of 10 bins

        # Then save the result
        self._save_json(offer, f'offer_{self.step}')

    def save_from_previous(self):
        # if the current market type is DAM, then we need to save it in order to run RTM
        if 'DAM' in self.market["market_type"]:
            self._save_json(self.market['previous'], self._prev_dam_file)

    def _day_ahead_offer(self):
        # Make the offer curves and unload into arrays
        # type = self.market['market_type']
        type = self.market['uid'][:5]
        prices = self.market["previous"][type]["prices"]["EN"]
        self._calculate_offer_curve(prices)
        self._format_offer_curve()

        return self.formatted_offer

    def _format_offer_curve(self, required_times):
        # Offer parsing script below:
        required_times = [t for t in self.market['timestamps']]

        # Convert the offer curves to timestamp:offer_value dictionaries
        block_ch_mc = {}
        for i, cost in enumerate(self.charge_mc):
            block_ch_mc[required_times[i]] = float(cost)
        block_ch_mq = {}
        for i, power in enumerate(self.charge_mq):
            block_ch_mq[required_times[i]] = float(power)  # 125MW

        block_dc_mc = {}
        block_soc_mc = {}
        for i, cost in enumerate(self.discharge_mc):
            block_dc_mc[required_times[i]] = float(cost)
            block_soc_mc[required_times[i]] = 0

        block_dc_mq = {}
        block_soc_mq = {}
        for i, power in enumerate(self.discharge_mq):
            block_dc_mq[required_times[i]] = float(power)  # 125MW
            block_soc_mq[required_times[i]] = 0

        # estimate initial SoC for tomorrow's DAM
        t_init = datetime.datetime.strptime(self.market['timestamps'][0],'%Y%m%d%H%M')
        #t_now = self.market['current_time']
        t_now = t_init - datetime.timedelta(hours=15) #TODO: switch back once above in included in market_data
        t_init = t_init.strftime('%Y%m%d%H%M')
        t_now = t_now.strftime('%Y%m%d%H%M')
        schedule = self.resource['schedule'][self.rid]['EN']
        schedule_to_tomorrow = [q for t,q in schedule if t_now <= t < t_init]
        schedule_to_tomorrow = self._process_efficiency(schedule_to_tomorrow)
        soc_estimate = self.resource['status'][self.rid]['soc'] - sum(schedule_to_tomorrow)
        soc_estimate = min(self.socmax, max(soc_estimate, self.socmin))
        dispatch_estimate = self.resource['schedule'][self.rid]['EN'][t_init]

        # Package the dictionaries into an output formatted dictionary
        offer_out_dict = {self.rid: {}}
        offer_out_dict[self.rid] = {"block_ch_mc": block_ch_mc, "block_ch_mq": block_ch_mq, "block_dc_mc": block_dc_mc,
                               "block_dc_mq": block_dc_mq, "block_soc_mc": block_soc_mc, "block_soc_mq": block_soc_mq}
        offer_out_dict[self.rid].update(self._default_reserve_offer())
        offer_out_dict[self.rid].update(self._default_dispatch_capacity())
        offer_out_dict[self.rid].update(self._default_offer_constants(soc_begin=soc_estimate, init_en=dispatch_estimate))

        self.formatted_offer = offer_out_dict

    def _process_efficiency(self, data:list):
        processed_data = []
        for num in data:
            if num < 0:
                processed_data.append(num * self.efficiency)
            else:
                processed_data.append(num)
        return processed_data

    def _real_time_offer(self):
        initial_soc = resource_info["status"][self.rid]["soc"]
        soc_available = initial_soc - self.socmin
        soc_headroom = self.socmax - initial_soc
        block_dc_mc = {}
        block_dc_mq = {}
        block_ch_mc = {}
        block_ch_mq = {}
        block_soc_mc = {}
        block_soc_mq = {}

        t_end = self.market['timestamps'][-1]
        for t_now in self.market['timestamps']:
            en_ledger = {t:order for t,order in resource_info['ledger'][self.rid]['EN'] if t >= t_now}
            block_ch_mq[t_now] = []
            block_ch_mc[t_now] = []
            block_dc_mq[t_now] = []
            block_dc_mc[t_now] = []

            # add blocks for cost of current dispatch:
            for mq, mc in resource_info['ledger'][self.rid]['EN'][t_now]:
                if mq < 0:
                    soc_available += mq * self.efficiency
                    soc_headroom -= mq * self.efficiency
                    block_ch_mq[t_now].append(-mq)
                    block_ch_mc[t_now].append(mc)
                elif mq > 0:
                    soc_available -= mq
                    soc_headroom += mq
                    block_dc_mq[t_now].append(mq)
                    block_dc_mc[t_now].append(mc)

            # add blocks for soc available/headroom
            ledger_list = [tup for sublist in en_ledger.values() for tup in sublist]
            ledger_decreasing = sorted(ledger_list, key=lambda tup:tup[1], reverse=True)
            ledger_increasing = sorted(ledger_list, key=lambda tup:tup[1], reverse=False)
            # use decreasing price ledger for charging cost curve
            remaining_capacity = soc_headroom
            for mq, mc in ledger_decreasing:
                if 0 > mq >= -remaining_capacity:
                    remaining_capacity += mq * self.efficiency # remaining_capacity decreases because m<0
                    block_ch_mq[t_now].append(-mq)
                    block_ch_mc[t_now].append(mc)
                else:
                    remaining_capacity -= remaining_capacity
                    block_ch_mq[t_now].append(remaining_capacity)
                    block_ch_mc[t_now].append(mc)
                    break
            if remaining_capacity:
                block_ch_mq[t_now].append(remaining_capacity)
                block_ch_mc[t_now].append(self.price_floor)
            # use increasing price ledger for discharging cost curve
            remaining_capacity = soc_available
            for mq, mc in ledger_increasing:
                if 0 < mq <= remaining_capacity:
                    remaining_capacity -= mq
                    block_dc_mq[t_now].append(mq)
                    block_dc_mc[t_now].append(mc)
                else:
                    remaining_capacity -= remaining_capacity
                    block_dc_mq[t_now].append(remaining_capacity)
                    block_dc_mc[t_now].append(mc)
                    break
            if remaining_capacity:
                block_dc_mq[t_now].append(remaining_capacity)
                block_dc_mc[t_now].append(self.price_ceiling)

        # valuation of post-horizon SoC
        post_market_ledger = {t: order for t, order in resource_info['ledger'][self.rid]['EN'] if t > t_end}
        post_market_list = [tup for sublist in post_market_ledger.values() for tup in sublist]
        post_market_sorted = sorted(post_market_list, key=lambda tup:tup[1], reverse=True)
        block_soc_mq[t_end] = []
        block_soc_mc[t_end] = []
        remaining_capacity = soc_available
        for mq, mc in post_market_sorted:
            if 0 < mq <= remaining_capacity:
                remaining_capacity -= mq
                block_soc_mq[t_end].append(mq)
                block_soc_mc[t_end].append(mc)
            else:
                remaining_capacity -= remaining_capacity
                block_soc_mq[t_end].append(remaining_capacity)
                block_soc_mc.append(mc)
        if remaining_capacity:
            block_soc_mq[t_end].append(remaining_capacity)
            block_soc_mc[t_end].append(self.price_ceiling)
        block_soc_mq[t_end].append(soc_headroom)
        block_soc_mc[t_end].append(self.price_floor)


        # Package the dictionaries into an output formatted dictionary
        offer_out_dict = {self.rid: {}}
        offer_out_dict[self.rid] = {"block_ch_mc": block_ch_mc, "block_ch_mq": block_ch_mq, "block_dc_mc": block_dc_mc,
                               "block_dc_mq": block_dc_mq, "block_soc_mc": block_soc_mc, "block_soc_mq": block_soc_mq}
        offer_out_dict[self.rid].update(self._default_reserve_offer())
        offer_out_dict[self.rid].update(self._default_dispatch_capacity())
        offer_out_dict[self.rid].update(self._default_offer_constants(bid_soc=True))

        return offer_out_dict

    def _default_reserve_offer(self):
        reg = ['cost_rgu', 'cost_rgd', 'cost_spr', 'cost_nsp']
        res_dict = {}
        for r in reg:
            res_dict[r] = {t: 0 for t in self.market['timestamps']}
        return res_dict

    def _default_dispatch_capacity(self):
        max_dict = {}
        max_dict['chmax'] = {t: self.chmax for t in self.market['timestamps']}
        max_dict['dcmax'] = {t: self.dcmax for t in self.market['timestamps']}
        return max_dict

    def _default_offer_constants(self, **options):
        constants = {}
        constants['soc_begin'] = self.resource['status'][self.rid]['soc']
        constants['init_en'] = self.resource['status'][self.rid]['dispatch']
        constants['init_status'] = 1
        constants['ramp_dn'] = 9999
        constants['ramp_up'] = 9999
        constants['socmax'] = self.socmax
        constants['socmin'] = self.socmin
        constants['eff_ch'] = self.efficiency
        constants['eff_dc'] = 1.0
        constants['soc_end'] = self.socmin
        constants['bid_soc'] = False

        constants.update(options)

        return constants

    def _load_dam_prices_times(self):
        now = self.market['timestamps'][0]
        hour_beginning = now[:10] + '00'
        type = self.market['market_type']
        if hour_beginning in self.market['previous'][type]['timestamp']:
            prices = self.market['previous'][type]['EN']
            times = self.market['previous'][type]['timestamp']
        else:
            with open(self._prev_dam_file, "r") as file:
                prices = json.load(file)
                times = [key for key in prices.keys()]
                prices = [value for value in prices.values()]
        return prices, times

    def _save_json(self, save_dict, filename):
        # Save as json file in the current directory with name offer_{time_step}.json
        with open(f'offer_{self.step}.json', 'w') as f:
            json.dump(save_dict, f, indent=4, cls=NpEncoder)

    def _calculate_opportunity_costs(self, prices):

        self._scheduler(prices)

        # combine the charge/discharge list
        combined_list = [dis - ch for ch, dis in zip(self._charge_list, self._discharge_list)]

        # finding the index for first charge and last discharge
        t1_ch = next((index for index, value in enumerate(combined_list) if value < 0), None)
        t_last_dis = next((index for index in range(len(combined_list) - 1, -1, -1) if combined_list[index] > 0), None)

        # create two list for charging/discharging opportunity costs
        self._oc_dis_list = []
        self._oc_ch_list = []

        opportunity_costs = pd.DataFrame(None, index=range(len(prices)), columns=['Time', 'charge cost', 'disch cost'])
        soc = pd.DataFrame(None, index=range(len(prices) + 1), columns=['Time', 'SOC'])


        for index, row in opportunity_costs.iterrows():
            i = index
            row['Time'] = index

            # charging
            if combined_list[i] < 0:
                oc_ch, oc_dis = self._calc_oc_charge(combined_list, prices, i)
            # discharging
            elif combined_list[i] > 0:
                oc_ch, oc_dis = self._calc_oc_discharge(combined_list, prices, i)
            else:
                # before first charge
                if i < t1_ch:
                    oc_ch, oc_dis = self._calc_oc_before_first_charge(prices, t1_ch, i)
                # after the last discharge
                elif i > t_last_dis:
                    oc_ch, oc_dis = self._calc_oc_after_last_discharge(prices, t_last_dis, i)
                # between cycles
                else:
                    oc_ch, oc_dis = self._calc_oc_between_cycles(combined_list, prices, i)

            # save to list
            self._oc_ch_list.append(oc_ch)
            self._oc_dis_list.append(oc_dis)
            # save to dataframe
            row['charge cost'] = oc_ch
            row['disch cost'] = oc_dis

        return opportunity_costs

    def _calculate_offer_curve(self, prices):

        # marginal cost comes from opportunity cost calculation
        oc = self._calculate_opportunity_costs(prices)
        self.charge_mc = oc['charge cost'].values
        self.discharge_mc = oc['disch cost'].values

        # marginal quantities from scheduler values
        self.charge_mq = self._charge_list
        self.discharge_mq = self._discharge_list

    def _calc_oc_charge(self, combined_list, prices, idx):
        # opportunity cost during scheduled charge
        j = idx + 1 + next((index for index, value in enumerate(combined_list[idx + 1:]) if value > 0), None)
        oc_ch = min(prices[1:j], self.efficiency * prices[j]) if idx == 0 else min(np.delete(prices[0:j], idx).min(),
                                                                                 self.efficiency * prices[j])

        arr1 = prices[0] if idx == 0 else prices[0:idx].min()
        arr2 = 0 if j == idx + 1 else prices[idx + 1] if j == idx + 2 else prices[(idx + 1):j].min()
        oc_dis = oc_ch + 0.01 if idx == 0 else (-prices[idx] + arr1 + arr2) / self.efficiency

        return oc_ch, oc_dis

    def _calc_oc_discharge(self, combined_list, prices, idx):
        # opportunity cost during scheduled discharge
        j = max((index for index, value in enumerate(combined_list[:idx]) if value < 0), default=None)
        arr1 = 0 if idx == len(prices) else prices[idx + 1] if idx == len(prices) - 1 else prices[(idx + 1):].max()
        arr2 = 0 if j == idx - 1 else prices[j + 1] if j == idx - 2 else prices[(j + 1):idx].max()
        oc_ch = (-prices[idx] + arr1 + arr2) * self.efficiency
        oc_dis = max(prices[j] / self.efficiency, prices[(j + 1):].max())

        return oc_ch, oc_dis

    def _calc_oc_before_first_charge(self, prices, t1_idx, idx):
        # opportunity cost before first charge
        max_ch = 0 if idx == t1_idx - 1 else prices[idx + 1] if idx == t1_idx - 2 else prices[(idx + 1):t1_idx].max()
        oc_ch = max(max_ch * self.efficiency, prices[t1_idx])
        oc_dis = oc_ch + 0.01 if idx == 0 else prices[0] / self.efficiency if idx == 1 else prices[0:idx].min() / self.efficiency

        return oc_ch, oc_dis

    def _calc_oc_after_last_discharge(self, prices, t_last, idx):
        # opportunity cost after last discharge
        oc_ch = prices[(idx + 1):].max() * self.efficiency if idx < len(prices) - 2 else prices[idx + 1] if idx == len(
            prices) - 2 else np.min(prices)
        arr = prices[idx - 1] if idx == t_last + 2 else prices[(t_last + 1):idx] if idx > t_last + 2 else np.max(prices)
        oc_dis = min(prices[t_last], arr.min() / self.efficiency)

        return oc_ch, oc_dis

    def _calc_oc_between_cycles(self, combined_list, prices, idx):
        j_next = idx + 1 + next((index for index, value in enumerate(combined_list[idx + 1:]) if value > 0),None)
        j_prev = max((index for index, value in enumerate(combined_list[:idx]) if value < 0), default=None)
        oc_ch = 0 if idx < j_prev + 2 else prices[idx - 1] if idx == j_prev + 2 else max(
            prices[(j_prev + 1):idx].max() * self.efficiency, prices[j_prev])
        oc_dis = 0 if idx > j_next - 2 else min(prices[j_next],
                                              prices[idx + 1] / self.efficiency) if idx == j_next - 2 else min(
            prices[j_next], prices[(idx + 1):j_next].min() / self.efficiency)

        return oc_ch, oc_dis

    def _scheduler(self, prices):

        number_step =len(prices)
        # [START solver]
        # Create the linear solver with the GLOP backend.
        solver = pywraplp.Solver.CreateSolver("GLOP")
        if not solver:
            return
        # [END solver]

        #Variables: all are continous
        charge = [solver.NumVar(0.0, self.chmax, "c"+str(i)) for i in range(number_step)]
        discharge = [solver.NumVar(0, self.dcmax,  "d"+str(i)) for i in range(number_step)]
        dasoc = [solver.NumVar(0.0, self.socmax, "b"+str(i)) for i in range(number_step+1)]
        dasoc[0]=0

        #Objective function
        solver.Minimize(
            sum(prices[i]*(charge[i]-discharge[i]) for i in range(number_step)))
        for i in range(number_step):
            solver.Add(dasoc[i] + self.efficiency*charge[i] - discharge[i] == dasoc[i+1])
        solver.Solve()
        #print("Solution:")
        #print("The Storage's profit =", solver.Objective().Value())
        self._charge_list=[]
        self._discharge_list=[]
        dasoc_list=[]
        for i in range(number_step):
            self._charge_list.append(charge[i].solution_value())
            self._discharge_list.append(discharge[i].solution_value())
            #dasoc_list.append(dasoc[i].solution_value())

>>>>>>> a572ce81a4fba6669fedb6dde2510e2f8269e39c

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
<<<<<<< HEAD
    rid = resource_info["rid"]
    ##to do:check if time_step in weasle is the same as in the agent

    os.environ["WANDB_API_KEY"] = "df5048753b47e0d3fb14ffae7704c794cd0639f1"
    args = Arguments()
    args.init_before_training(if_main=True)
    train_agent(time_step, market_info, resource_info)
=======
    rid = resource_info["rid"]
>>>>>>> a572ce81a4fba6669fedb6dde2510e2f8269e39c
