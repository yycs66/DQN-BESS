# ------------------------------------------------------------------------

# MIP_DQN2 deletes Actor_MIP
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
from market_participant import Agent


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
# Standard battery parameters
socmax = 608
socmin = 128
chmax = 125
dcmax = 125
efficiency = 0.892
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
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
    '''
    Agent is re-initialized every time the WEASLE Platform calls market_participant.py
    Input: time_step, market_data, and resource_data are input arguments from the script call
    Additional input data must be saved to disc and reloaded each time Agent is created (e.g., to facilitate Agent persistence)
    Output:
    - make_me_an_offer() reads the market type and saves to disc a JSON file containing offer data
    '''
    def __init__(self, time_step, market_info, resource_info):
        
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
    ##to do:check if time_step in weasle is the same as in the agent

    os.environ["WANDB_API_KEY"] = "df5048753b47e0d3fb14ffae7704c794cd0639f1"
    args = Arguments()
    args.init_before_training(if_main=True)
    train_agent(time_step, market_info, resource_info)