
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import datetime
import logging
import itertools

from tqdm import tqdm
from q_learning.utilities import load_config
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import wandb

from .visualizer import visualize_all_states, visualize_explained_variance


# ALGO LOGIC: initialize agent here:


BEST_ACTUAL_ACTION_REWARD_LOG = './random_best_actual_action_reward_log8.txt'

class RandomAgent:
    def __init__(self, env, run_name, shared_config_path, agent_config_path=None, override_config=None):
        # Load Shared Config
        print('Random Agent')
        self.shared_config = load_config(shared_config_path)
        # Load Agent Specific Config if path provided
        if agent_config_path:
            self.agent_config = load_config(agent_config_path)
        else:
            self.agent_config = {}

        # If override_config is provided, merge it with the loaded agent_config
        if override_config:
            self.agent_config.update(override_config)

        self.env = env
        self.run_name = run_name
        self.agent_type = 'cleanrl_dqn'
        # Access the results directory from the shared_config
        self.results_directory = self.shared_config['directories']['results_directory']
        self.model_directory = self.shared_config['directories']['model_directory']


        # Create a unique subdirectory for each run to avoid overwriting results
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.timestamp_day = datetime.datetime.now().strftime("%Y%m%d")
        self.model_subdirectory = os.path.join(self.model_directory, self.agent_type, self.run_name, self.timestamp)
        if not os.path.exists(self.model_subdirectory):
            os.makedirs(self.model_subdirectory, exist_ok=True)
        self.results_subdirectory = os.path.join(self.results_directory, run_name, self.timestamp)
        print('results subdir ', self.results_subdirectory)
        print('model subdir ', self.model_subdirectory)
        os.makedirs(self.results_subdirectory, exist_ok=True)

        # Set up logging to the correct directory
        log_file_path = os.path.join(self.results_subdirectory, 'agent_log.txt')
        logging.basicConfig(filename=log_file_path, level=logging.INFO)
        # Initialize agent-specific configurations and variables
        print('configs')
        print(self.agent_config['agent'])
        self.max_episodes = self.agent_config['agent']['max_episodes']
        self.learning_rate = self.agent_config['agent']['learning_rate']
        self.discount_factor = self.agent_config['agent']['discount_factor']
        self.exploration_rate = self.agent_config['agent']['exploration_rate']
        # self.min_exploration_rate = self.agent_config['agent']['min_exploration_rate']
        self.exploration_decay_rate = self.agent_config['agent']['exploration_decay_rate']
        # Parameters for adjusting learning rate over time
        self.learning_rate_decay = self.agent_config['agent']['learning_rate_decay']
        print('min epsilon exist: ', 'min_epsilon' in self.agent_config['agent'])
        self.min_epsilon = self.agent_config['agent']['min_epsilon'] if 'min_epsilon' in self.agent_config['agent'] else 0.1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.training_data = []
        self.possible_actions = [list(range(0, (k))) for k in self.env.action_space.nvec]
        self.possible_states = [list(range(0, (k))) for k in self.env.observation_space.nvec]
        self.all_actions = [str(i) for i in list(itertools.product(*self.possible_actions))]
        self.all_states = [str(i) for i in list(itertools.product(*self.possible_states))]

        self.states = list(itertools.product(*self.possible_states))

        # moving average for early stopping criteria

        self.num_actions = self.env.action_space.nvec[0]
        # print('num actions', self.num_actions)
        self.seed = 1
        self.total_timesteps = self.max_episodes
        self.tau = self.agent_config['agent']['tau'] if 'tau' in self.agent_config['agent'] else 1.0
        self.learning_starts = 20
        self.buffer_size = 1000
        self.start_epsilon = 1
        self.train_frequency = self.agent_config['agent']['train_frequency'] if 'train_frequency' in self.agent_config['agent'] else 10
        self.batch_size = 200
        self.target_network_update_frequency = self.agent_config['agent']['target_network_update_frequency'] \
             if 'target_network_update_frequency' in self.agent_config['agent'] else 100 # was 1000
        self.gamma = self.agent_config['agent']['discount_factor'] if 'discount_factor' in self.agent_config['agent'] else 0.99
        print('gamma ', self.gamma)
        self.torch_deterministic = True
        wandb.init(name=f' Discrete Learning rate: {self.learning_rate} episodes: {self.max_episodes} target update frequency: {self.target_network_update_frequency}')
        torch.set_num_threads(1)

    def train(self, alpha):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # TRY NOT TO MODIFY: start the game
        obs, _ = self.env.reset()
        i = 0
        with open(BEST_ACTUAL_ACTION_REWARD_LOG, 'w') as wfile:
            wfile.write(f'\nalpha={alpha} lr={self.learning_rate} target_network_update_frequency={self.target_network_update_frequency} min_epsilon={self.min_epsilon} episodes={self.total_timesteps} run_name={wandb.run.name} \n')
        for global_step in tqdm(range(self.total_timesteps)):
            done = False
            obs, _ = self.env.reset()
            # with open(BEST_ACTUAL_ACTION_REWARD_LOG, 'a') as wfile:
            #     wfile.write(f'episode={global_step} i={i}\n')
            while not done:
                # print('obs ', obs)
                actions = np.array(self.env.action_space.sample())
                scaled_actions = actions * (100. / (self.num_actions-1))
                print('taken action', scaled_actions, 'state ', obs)

                action_alpha_list = [*scaled_actions, alpha, obs]
                if obs[0]==0 and obs[1]==3: #obs==[0,3]
                    print('inspect here')
                print('obs before step ', obs)
                next_obs, rewards, terminations, truncations, infos = self.env.step(action_alpha_list)
                print('next_obs' , next_obs)
                if obs[0]==0 and obs[1]==3:
                    if infos["best_reward"] != 17:
                        print('inspect')
                print('rewards ', rewards )
                if i % 100 <= 3 or i % 100 >= (100-3):
                    with open(BEST_ACTUAL_ACTION_REWARD_LOG, 'a') as wfile:
                        wfile.write(f'i={i} \n')
                        wfile.write(f'chosen action={actions} obs={obs}\n')
                        wfile.write(f'actial_reward={rewards}\n')

                if i % 100 <= 3 or i % 100 >= (100-3):
                    with open(BEST_ACTUAL_ACTION_REWARD_LOG, 'a') as wfile:
                        wfile.write(f'best_action={infos["best_action"]} best_reward={infos["best_reward"]}\n')
                obs = next_obs
                done = terminations or truncations
                i+=1


