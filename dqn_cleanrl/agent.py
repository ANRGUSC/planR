

# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqn_ataripy
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

from q_learning.visualizer import visualize_explained_variance


# ALGO LOGIC: initialize agent here:

class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        # print('input shape ', np.array(env.observation_space.shape).prod())
        # print('output shape ', env.action_space.nvec)
        self.lin1 = nn.Linear(np.array(env.observation_space.shape).prod(), 4)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(4, 8)
        self.out = nn.Linear(8, env.action_space.nvec[0])
        self.network = nn.Sequential(
            nn.Linear(np.array(env.observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84,env.action_space.nvec[0]),
        )

    def forward(self, x):
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.relu(x)
        x = self.out(x)
        return x


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):

    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

class DQNCleanrlAgent:
    def __init__(self, env, run_name, shared_config_path, agent_config_path=None, override_config=None):
        # Load Shared Config
        self.shared_config = load_config(shared_config_path)
        self.writer = SummaryWriter(f"runs/{run_name}")

        # Load Agent Specific Config if path provided
        if agent_config_path:
            self.agent_config = load_config(agent_config_path)
        else:
            self.agent_config = {}

        # If override_config is provided, merge it with the loaded agent_config
        if override_config:
            self.agent_config.update(override_config)

        # Access the results directory from the shared_config
        self.results_directory = self.shared_config['directories']['results_directory']

        # Create a unique subdirectory for each run to avoid overwriting results
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.results_subdirectory = os.path.join(self.results_directory, run_name, timestamp)
        os.makedirs(self.results_subdirectory, exist_ok=True)

        # Set up logging to the correct directory
        log_file_path = os.path.join(self.results_subdirectory, 'agent_log.txt')
        logging.basicConfig(filename=log_file_path, level=logging.INFO)
        # Initialize agent-specific configurations and variables
        self.env = env
        self.run_name = run_name
        self.max_episodes = self.agent_config['agent']['max_episodes']
        self.learning_rate = self.agent_config['agent']['learning_rate']
        self.discount_factor = self.agent_config['agent']['discount_factor']
        self.exploration_rate = self.agent_config['agent']['exploration_rate']
        self.min_exploration_rate = self.agent_config['agent']['min_exploration_rate']
        self.exploration_decay_rate = self.agent_config['agent']['exploration_decay_rate']

        # Parameters for adjusting learning rate over time
        self.learning_rate_decay = self.agent_config['agent']['learning_rate_decay']
        self.min_learning_rate = self.agent_config['agent']['min_learning_rate']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        self.training_data = []
        self.possible_actions = [list(range(0, (k))) for k in self.env.action_space.nvec]
        self.possible_states = [list(range(0, (k))) for k in self.env.observation_space.nvec]
        self.all_actions = [str(i) for i in list(itertools.product(*self.possible_actions))]
        self.all_states = [str(i) for i in list(itertools.product(*self.possible_states))]

        self.states = list(itertools.product(*self.possible_states))

        # moving average for early stopping criteria
        self.moving_average_window = 100  # Number of episodes to consider for moving average
        self.stopping_criterion = 0.01  # Threshold for stopping
        self.prev_moving_avg = -float('inf')  # Initialize to negative infinity to ensure any reward is considered an improvement in the first episode.
        self.num_actions = self.env.action_space.nvec[0]
        self.q_network = QNetwork(self.env).to(self.device)
        self.seed = 1
        self.total_timesteps = self.max_episodes
        self.tau = self.agent_config['agent']['tau'] if 'tau' in self.agent_config['agent'] else 1.0
        self.learning_starts = 0
        self.buffer_size = 1000000
        self.start_e = 1
        # self.end_e = 0.1
        # self.exploration_fraction = self.exploration_rate
        self.train_frequency = 10
        self.batch_size = 32
        self.target_network_update_frequency = self.agent_config['agent']['target_network_update_frequency'] \
             if 'target_network_update_frequency' in self.agent_config['agent'] else 100 # was 1000
        self.gamma = 0.99
        self.torch_deterministic = True
        wandb.init(name=f' Discrete Learning rate: {self.learning_rate} episodes: {self.max_episodes} target update frequency: {self.target_network_update_frequency}')



    def train(self, alpha):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = self.torch_deterministic
        q_network = QNetwork(self.env).to(self.device)
        optimizer = optim.Adam(q_network.parameters(), lr=self.learning_rate)
        target_network = QNetwork(self.env).to(self.device)
        target_network.load_state_dict(q_network.state_dict())
        rb = ReplayBuffer(
            self.buffer_size,
            self.env.observation_space,
            self.env.action_space,
            self.device,
            optimize_memory_usage=True,
            handle_timeout_termination=False,
        )
        start_time = time.time()
        actual_rewards = []
        predicted_rewards = []

        # TRY NOT TO MODIFY: start the game
        obs, _ = self.env.reset()
        for global_step in range(self.total_timesteps):
            # ALGO LOGIC: put action logic here

            done = False
            episode_tot_reward = 0
            episode_len = 0
            self.env.reset()
            # state = self.env.reset()
            # c_state = state[0]
            # converted_state = str(tuple(c_state))
            # variables for explained variance logging
            e_predicted_rewards = []
            e_returns = []

            while not done:
                # print('curr obs ', obs)
                epsilon = linear_schedule(self.start_e, self.min_learning_rate, self.exploration_rate * self.total_timesteps, global_step)
                predicted_reward = 0
                # print('epsilon ', epsilon)
                if random.random() < epsilon:
                    actions = np.array(self.env.action_space.sample())

                else:
                    q_values = q_network(torch.Tensor(obs).to(self.device))
                    predicted_reward = q_values[torch.argmax(q_values).cpu()]

                    actions = np.array([torch.argmax(q_values).cpu().numpy()])
                    # print(actions)

                # TRY NOT TO MODIFY: execute the game and log data.
                scaled_actions = actions * 50
                converted_state = str(tuple(obs))
                # state_idx = self.all_states.index(converted_state)
                # predicted_reward = self.q_table[state_idx, actions]
                if isinstance(predicted_reward, int):
                    e_predicted_rewards.append(predicted_reward)
                else:
                    e_predicted_rewards.append(predicted_reward.item())

                action_alpha_list = [*scaled_actions, alpha]
                # print('action alpha list', action_alpha_list)
                next_obs, rewards, terminations, truncations, infos = self.env.step(action_alpha_list)
                obs = next_obs
                e_returns.append(float(rewards))

                next_obs_rb = np.array(next_obs, dtype=float)
                obs_rb = np.array(obs, dtype=float)
                actions_rb = np.array(actions, dtype=float)
                rb.add(obs_rb, next_obs_rb, actions_rb, rewards, terminations, infos)

                episode_tot_reward += rewards
                episode_len += 1
                done = terminations or truncations
            episode_mean_reward = episode_tot_reward / episode_len
            # print('episode reward: ', episode_mean_reward, global_step)
            wandb.log({
                'discrete/average_return': episode_mean_reward,
            })
            predicted_rewards.append(e_predicted_rewards)
            actual_rewards.append(e_returns)

            # # TRY NOT TO MODIFY: record rewards for plotting purposes
            # if "final_info" in infos:
            #     for info in infos["final_info"]:
            #         if info and "episode" in info:
            #             print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
            #             self.writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
            #             self.writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

            # next_obs = np.array(next_obs, dtype=float)
            # obs = np.array(obs, dtype=float)
            # actions = np.array(actions, dtype=float)
            # rb.add(obs, next_obs, actions, rewards, terminations, infos)

            # obs = next_obs

            # ALGO LOGIC: training.
            if global_step > self.learning_starts:
                if global_step % self.train_frequency == 0:
                    data = rb.sample(self.batch_size)
                    with torch.no_grad():
                        target_max, _ = target_network(data.next_observations.to(dtype=torch.float32)).max(dim=1)
                        td_target = data.rewards.flatten() + self.gamma * target_max * (1 - data.dones.flatten())
                    observations = data.observations.float()
                    old_val = q_network(observations).gather(1, data.actions).squeeze()
                    loss = F.mse_loss(td_target, old_val)
                    wandb.log({
                        'discrete/loss': loss,
                    })
                    # optimize the model
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # update target network
                if global_step % self.target_network_update_frequency == 0:
                    for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                        target_network_param.data.copy_(
                            self.tau * q_network_param.data + (1.0 - self.tau) * target_network_param.data
                        )
        explained_variance_path = visualize_explained_variance(actual_rewards, predicted_rewards, self.results_subdirectory, self.max_episodes)
        wandb.log({"discrete/Explained Variance": [wandb.Image(explained_variance_path)]})#IMP
