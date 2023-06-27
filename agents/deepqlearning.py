import os
# import tensorflow as tf
import numpy as np
import random
from tqdm import tqdm
import itertools
import copy
import wandb
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
import math
import gymnasium as gym
from tianshou.data import Collector, PrioritizedVectorReplayBuffer, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.utils.net.common import Net
import tianshou as ts
from tianshou.trainer import offpolicy_trainer
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger



# steps_done = 0

# # what should input dimension be? (how many obsservations?)
# # target net vs. policy net?
# # gym env argument: 1D list, what should the action look like ?
# # what does alpha come from ?
# # why do state dimension change in different input of states and actions
# # why is state some times 1,1 and sometimes 1,2. what should state look like?
# # weights seem to change just little bit, is it normal?
# # draw average of Q values for training iterations

# # Output of neural net is between 0 and -1. is it normal?


# # state: [number infected students, community risk]



class DeepQAgent:
    def __init__(self, env, episodes, learning_rate, discount_factor, exploration_rate,
                 tau=1e-4, batch_size=64,tr_name='abcde'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_episodes = 15000
        self.discount = discount_factor
        self.gamma = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = 1.0 / float(episodes)
        self.TAU = tau
        self.lr = learning_rate
        self.batch_size = batch_size
        # get environment
        self.env = env
        self.state_shape = 2
        self.action_shape = np.prod(self.env.action_space.nvec)  # might not be accurate
        # self.action_shape = 1
        print(f'action shape: {self.action_shape}')
        print(f'aaa: {self.env.action_space.nvec}')
        # print(self.env.action_space.shape[0])
        self.hidden_sizes = [128, 128]
        self.training_num = 8 # can change
        # self.test_num = 100
        self.test_num = 1
        # self.buffer_size = 10000
        self.buffer_size = 10
        self.step_per_epoch = 15
        self.step_per_collect = 8
        self.update_per_step = 0.125
        self.eps_train = 0.73 # exploration_rate
        self.eps_test = 0.01
        self.train_envs = DummyVectorEnv(
            [lambda: gym.make('CampusGymEnv-v0') for _ in range(self.training_num)]
        )
        self.test_envs = DummyVectorEnv(
            [lambda: gym.make('CampusGymEnv-v0') for _ in range(self.test_num)]
        )

        # np.random.seed(100)
        # torch.manual_seed(100)
        # self.train_envs.seed(100)
        # self.test_envs.seed(100)

        self.net = Net(
            self.state_shape,
            self.action_shape,
            hidden_sizes=self.hidden_sizes,
            device=self.device,
            # softmax=True,
        ).to(self.device)
        self.optimizer = optim.AdamW(self.net.parameters(), lr=self.lr, amsgrad=True)

        self.policy = ts.policy.DQNPolicy(
            self.net,
            self.optimizer,
            self.gamma,
            3,
            target_update_freq=500
        )

        self.train_collector = Collector(
            self.policy,
            self.train_envs,
            VectorReplayBuffer(self.buffer_size, len(self.train_envs)),
            exploration_noise=True
        )
        self.test_collector = Collector(self.policy, self.test_envs, exploration_noise=True)
        self.logdir = 'log'
        log_path = os.path.join(self.logdir, 'CampusGymEnv', 'dqn')
        writer = SummaryWriter(log_path)
        self.logger = TensorboardLogger(writer)
        print('Deep Q Agent Constructor Finish')


    def train_fn(self, epoch, env_step):
        eps = max(self.eps_train * (1 - 5e-6)**env_step, self.eps_test)
        self.policy.set_eps(eps)

    def test_fn(self, epoch, env_step):
        self.policy.set_eps(self.eps_test)

    def train(self, alpha):
        self.train_collector.collect(n_step=self.batch_size*self.training_num)
        result = offpolicy_trainer(
            self.policy,
            self.train_collector,
            self.test_collector,
            self.max_episodes,
            self.step_per_epoch,
            self.step_per_collect,
            self.test_num,
            self.batch_size ,
            update_per_step=self.update_per_step,
            # stop_fn=self.stop_fn,
            train_fn=self.train_fn,
            test_fn=self.test_fn,
            # save_best_fn=save_best_fn,
            logger=self.logger
        )

    # How to append alpha to action before calling step function?


# # wandb.init(project="campus-plan", entity="leezo")
# # tf.compat.v1.disable_eager_execution()

# # def get_discrete_value(number):
# #     value = 0
# #     if number in range(0, 33):
# #         value = 0
# #     elif number in range(34, 66):
# #         value = 1
# #     elif number in range(67, 100):
# #         value = 2
# #     return value


# # # convert actions to discrete values 0,1,2
# # def action_conv_disc(action_or_state):
# #     discaction = []
# #     for i in (action_or_state):
# #         action_val = get_discrete_value(i)
# #         discaction.append(action_val)
# #     return discaction


# # convert list of discrete values to 0 to 100 range
# def disc_conv_action(discaction):
#     action = []
#     for i in range(len(discaction)):
#         action.append((int)(discaction[i] * 50))
#     return action


# Transition = namedtuple('Transition',
#                         ('state', 'action', 'next_state', 'reward'))

# # class ReplayMemory(object):

# #     def __init__(self, capacity):
# #         self.memory = deque([], maxlen=capacity)

# #     def push(self, *args):
# #         """Save a transition"""
# #         self.memory.append(Transition(*args))

# #     def sample(self, batch_size):
# #         return random.sample(self.memory, batch_size)

# #     def __len__(self):
# #         return len(self.memory)

# # dense network with 5 layer, standard parameters.
# class DeepQNetwork(nn.Module):
#     def __init__(self, n_observations, n_actions):
#         super(DeepQNetwork, self).__init__()
#         n_observations=2
#         self.layer1 = nn.Linear(n_observations, 128)
#         self.layer2 = nn.Linear(128, 128)
#         self.layer3 = nn.Linear(128, n_actions)

#     def forward(self, x):
#         x = F.relu(self.layer1(x))
#         x = F.relu(self.layer2(x))
#         return self.layer3(x)




# class DeepQAgent:

#     def __init__(self, env, episodes, learning_rate, discount_factor, exploration_rate,
#                  tau=1e-4, batch_size=10,tr_name='abcde'):
#         # set hyperparameters
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.max_episodes = 15000
#         # self.max_actions = int(args.max_actions)
#         self.discount = discount_factor
#         self.exploration_rate = exploration_rate
#         self.exploration_decay = 1.0 / float(episodes)
#         self.batch_size = batch_size
#         self.TAU = tau
#         self.lr = learning_rate
#         # get environment
#         self.env = env
#         print(f'observation space dimension: {self.env.observation_space.nvec}')
#         n_obs = np.prod(self.env.observation_space.nvec)
#         n_actions = np.prod(self.env.action_space.nvec)  # might not be accurate
#         self.target_net = DeepQNetwork(n_obs, n_actions).to(self.device)
#         self.policy_net = DeepQNetwork(n_obs, n_actions).to(self.device)
#         self.target_net.load_state_dict(self.policy_net.state_dict())
#         self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
#         self.memory = ReplayMemory(10000)
#         self.training_data = []


#     def optimize_model(self):
#         if len(self.memory) < self.batch_size:
#             return
#         transitions = self.memory.sample(self.batch_size)
#         # print(f'transitions: {transitions}')

#         batch = Transition(*zip(*transitions))
#         # print(f'batch: {batch}')
#         # Compute a mask of non-final states and concatenate the batch elements
#         # (a final state would've been the one after which simulation ended)
#         non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
#                                             batch.next_state)), device=self.device, dtype=torch.bool)
#         non_final_next_states = torch.cat([s for s in batch.next_state
#                                                     if s is not None])
#         state_batch = torch.cat(batch.state)
#         # print(f'state batch: {state_batch}')
#         action_batch = torch.cat(batch.action)
#         reward_batch = torch.cat(batch.reward)

#         # state_action_values = self.policy_net(state_batch).gather(1, action_batch)
#         state_action_values = self.policy_net(state_batch)
#         # print(f'state action vals: {state_action_values}')


#         next_state_values = torch.zeros(self.batch_size, device=self.device)
#         # print(f'non_final_next_state: {non_final_next_states}')
#         with torch.no_grad():
#             # nnOutput = self.target_net(non_final_next_states)
#             # print(f'nnOutput: {nnOutput}')
#             next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
#             # print(f'next_state_values[non_final_mask]: {next_state_values[non_final_mask]}')
#         # Compute the expected Q values
#         expected_state_action_values = (next_state_values * self.discount) + reward_batch

#         # Compute Huber loss
#         # criterion = F.mse_loss() # change loss function. reward function ?
#         # loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
#         loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

#         # Optimize the model
#         self.optimizer.zero_grad()
#         loss.backward()
#         # In-place gradient clipping
#         # torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
#         self.optimizer.step()


#     def select_action(self, state):
#         global steps_done
#         sample = random.random()
#         eps_threshold = self.exploration_rate
#         # print(f'epsilon threshold: {eps_threshold} sample: {sample}')
#         steps_done += 1
#         # print(f'state: {state}')
#         if self.exploration_rate > 0.1:
#             self.exploration_rate -= self.exploration_decay
#         if sample > eps_threshold:
#             with torch.no_grad():
#                 q = self.policy_net(state)
#                 q_flat = q.flatten()
#                 maxQ, maxQ_index = torch.max(q_flat, dim=0)
#                 action = maxQ_index
#                 # print(f'Q values: {q}, max Q: {maxQ} max Q index: {maxQ_index}')
#                 return action, maxQ
#                 # return self.policy_net(state).argmax().item()

#         else:
#             return self.env.action_space.sample(), -1



#     def train(self, alpha):
#         # hyper parameter
#         discount = self.discount
#         exploration_rate = self.exploration_rate
#         exploration_decay = self.exploration_decay
#         episode_actions = {}
#         episode_rewards = {}
#         episode_allowed = {}
#         episode_infected_students = {}
#         self.rewards = []
#         self.Q_values = []

#         for i in tqdm(range(self.max_episodes)):
#             # Initialize the environment and get it's state
#             state = self.env.reset()
#             # print(f'resetted state: {state}')
#             state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
#             done = False
#             e_infected_students = []
#             e_return = []
#             e_allowed = []
#             score = 0
#             total_Q = 0
#             j = 1
#             q_cnt = 0
#             while not done:
#                 action, Q_val = self.select_action(state)
#                 action_alpha_list = [action*50, alpha]
#                 # print(f'action_alpha_list: {action_alpha_list}')
#                 # observation, reward, terminated, info = self.env.step(action_alpha_list)
#                 observation, reward, terminated, truncated, info = self.env.step(action_alpha_list) # change to this if using gymnasium
#                 reward = torch.tensor([reward], device=self.device)
#                 score += reward.item()
#                 total_Q += Q_val
#                 if Q_val != -1:
#                     q_cnt += 1
#                 # print(f'week {j} reward: {reward}')
#                 done = terminated

#                 if terminated:
#                     next_state = None
#                 else:
#                     next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

#                 self.memory.push(state, torch.tensor(action).view(1) , next_state, reward)
#                 state = next_state
#                 # print(f'next state: {next_state}')
#                 self.optimize_model()

#                 # Soft update of the target network's weights
#                 # θ′ ← τ θ + (1 −τ )θ′
#                 target_net_state_dict = self.target_net.state_dict()
#                 # print(f'target state dict: {target_net_state_dict}')
#                 policy_net_state_dict = self.policy_net.state_dict()
#                 # print(f'policy net dict: {policy_net_state_dict}')
#                 for key in policy_net_state_dict:
#                     target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
#                 self.target_net.load_state_dict(target_net_state_dict)

#                 week_reward = reward
#                 e_return.append(week_reward)
#                 e_allowed.append(info['allowed'])
#                 e_infected_students.append(info['infected'])
#                 j+=1
#             if q_cnt != 0:
#                 self.Q_values.append(total_Q / q_cnt)
#             self.rewards.append(score)
#             episode_rewards[i] = e_return
#             episode_allowed[i] = e_allowed
#             episode_infected_students = e_infected_students
#             if (i+1) % 100 == 0:
#                 self.plot_rewards('scatter', i+1)
#         # self.plot_rewards('line')
#         # self.plot_Qvals()
#         self.training_data = [episode_rewards, episode_allowed, episode_infected_students]
#         return self.training_data

#     def plot_rewards(self, graph_type, num):
#         x = list(range(num))
#         y = self.rewards
#         plt.xlabel('Episode')
#         plt.ylabel('Reward')
#         if graph_type == 'scatter':
#             plt.scatter(x,y)
#             plt.savefig(f'reward_graph_{num}_eps_scatter-smoothl1_lowcomrisk_Q0100_COMPLEXMODEL_6_14_{self.lr}.png')
#         elif graph_type == 'line':
#             plt.plot(x,y)
#             plt.savefig(f'reward_graph_{num}_eps_line-smoothl1_lowcommrisk_Q0100_COMPLEXMODEL_6_14_{self.lr}.png')
#         plt.close()

#     def plot_Qvals(self):
#         x = list(range(self.max_episodes))
#         y = self.Q_values
#         plt.xlabel('Episode')
#         plt.ylabel('Average Q Value ')
#         plt.scatter(x,y)
#         plt.savefig(f'Qvalue_graph_{self.max_episodes}_eps_scatter_smoothl1.png')
