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

steps_done = 0

# what should input dimension be? (how many obsservations?)
# target net vs. policy net?
# gym env argument: 1D list, what should the action look like ?
# what does alpha come from ?
# why do state dimension change in different input of states and actions
# why is state some times 1,1 and sometimes 1,2. what should state look like?

# state: [number infected students, community risk]


# wandb.init(project="campus-plan", entity="leezo")
# tf.compat.v1.disable_eager_execution()

# def get_discrete_value(number):
#     value = 0
#     if number in range(0, 33):
#         value = 0
#     elif number in range(34, 66):
#         value = 1
#     elif number in range(67, 100):
#         value = 2
#     return value


# # convert actions to discrete values 0,1,2
# def action_conv_disc(action_or_state):
#     discaction = []
#     for i in (action_or_state):
#         action_val = get_discrete_value(i)
#         discaction.append(action_val)
#     return discaction


# convert list of discrete values to 0 to 100 range
def disc_conv_action(discaction):
    action = []
    for i in range(len(discaction)):
        action.append((int)(discaction[i] * 50))
    return action


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# dense network with 5 layer, standard parameters.
class DeepQNetwork(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DeepQNetwork, self).__init__()
        n_observations=2
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)




class DeepQAgent:

    def __init__(self, env, episodes, learning_rate, discount_factor, exploration_rate,
                 tau=1e-4, batch_size=128,tr_name='abcde'):
        # set hyperparameters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_episodes = episodes
        # self.max_actions = int(args.max_actions)
        self.discount = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = 1.0 / float(episodes)
        self.batch_size = batch_size
        self.TAU = tau
        self.lr = learning_rate
        # get environment
        self.env = env
        # print(f'action space sample: {self.env.action_space.sample()}')
        n_obs = np.prod(self.env.observation_space.nvec)
        n_actions = np.prod(self.env.action_space.nvec)  # might not be accurate
        # print(f'n_obs: {n_obs}, n_actions: {n_actions}')
        self.target_net = DeepQNetwork(n_obs, n_actions).to(self.device)
        self.policy_net = DeepQNetwork(n_obs, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        self.memory = ReplayMemory(10000)
        self.training_data = []


    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)

        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)


        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.discount) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()


    def select_action(self, state):
        global steps_done
        sample = random.random()
        eps_threshold = self.exploration_rate
        steps_done += 1
        # print(f'state: {state}')
        if self.exploration_rate > 0.1:
            self.exploration_rate -= self.exploration_decay
        if sample < eps_threshold:
            with torch.no_grad():
                action_values = self.policy_net(state)
                return self.policy_net(state).argmax().item()

        else:
            return self.env.action_space.sample()



    def train(self, alpha):
        # hyper parameter
        discount = self.discount
        exploration_rate = self.exploration_rate
        exploration_decay = self.exploration_decay
        episode_actions = {}
        episode_rewards = {}
        episode_allowed = {}
        episode_infected_students = {}

        if torch.cuda.is_available():
            num_episodes = 600
        else:
            num_episodes = 50

        for i in tqdm(range(num_episodes)):
            # Initialize the environment and get it's state
            state = self.env.reset()
            # print(f'resetted state: {state}')
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            done = False
            e_infected_students = []
            e_return = []
            e_allowed = []
            while not done:
                action = self.select_action(state)
                action_alpha_list = [action*50, alpha]
                print(f'action_alpha_list: {action_alpha_list}')
                observation, reward, terminated, info = self.env.step(action_alpha_list)
                # observation, reward, terminated, truncated, info = env.step(action.item()) # change to this if using gymnasium
                reward = torch.tensor([reward], device=self.device)
                print(f'reward: {reward}')
                done = terminated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

                state = next_state
                self.optimize_model()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
                self.target_net.load_state_dict(target_net_state_dict)

                week_reward = reward
                e_return.append(week_reward)
                e_allowed.append(info['allowed'])
                e_infected_students.append(info['infected'])
            episode_rewards[i] = e_return
            episode_allowed[i] = e_allowed
            episode_infected_students = e_infected_students

        self.training_data = [episode_rewards, episode_allowed, episode_infected_students]
        return self.training_data
