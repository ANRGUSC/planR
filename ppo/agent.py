import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import torch
from tianshou.utils.net.common import ActorCritic
from tianshou.utils.net.discrete import Actor, Critic, IntrinsicCuriosityModule
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.utils import TensorboardLogger, WandbLogger
from .net import Net
matplotlib.use('Agg')
from tianshou.data import Batch, ReplayBuffer
import numpy as np
import itertools
from tianshou.policy import ICMPolicy, PPOPolicy
from tianshou.env import ShmemVectorEnv, DummyVectorEnv
from .utilities import load_config
from .visualizer import visualize_all_states, visualize_q_table, visualize_variance_in_rewards_heatmap, \
    visualize_explained_variance, visualize_variance_in_rewards, visualize_infected_vs_community_risk_table, states_visited_viz
import os
import io
import json
import logging
from datetime import datetime
from tqdm import tqdm
import wandb
import random
import pandas as pd
import csv
from tianshou.env import DummyVectorEnv
from torch.utils.tensorboard import SummaryWriter




class PPOagent:
    def __init__(self, env, run_name, shared_config_path, agent_config_path=None, override_config=None):
        self.env = env
        self.run_name = run_name

        self.shared_config = load_config(shared_config_path)
        
        if agent_config_path:
            self.agent_config = load_config(agent_config_path)
        else:
            self.agent_config = {}

        # If override_config is provided, merge it with the loaded agent_config
        if override_config:
            self.agent_config.update(override_config)
        print("Check", env.observation_space, env.action_space)
        self.action_shape = np.prod(env.action_space.nvec)
        self.state_shape = env.observation_space.shape
        print(f"state_shape: {self.state_shape}, action_shape: {self.action_shape}")
        self.moving_average_window = 100
        self.max_episodes = self.agent_config['agent']['max_episodes']
        self.learning_rate = self.agent_config['agent']['learning_rate']
        self.discount_factor = self.agent_config['agent']['discount_factor']
        self.eps_clip = self.agent_config['agent']['eps_clip']

        self.training_num = 1 # can change
        self.batch_size = 32
        # self.test_num = 100
        self.test_num = 1
        self.hidden_shape = self.agent_config['agent']['hidden_shape']
        self.buffer_size = 100
        self.stopping_criterion = 0.001
        self.steps_before_stop = 10

        self.net = Net(
            self.state_shape,
            self.hidden_shape
        )

        def dist(p):
            return torch.distributions.Categorical(logits=p)
        actor = Actor(self.net, self.action_shape, softmax_output=False)
        critic = Critic(self.net)
        optim = torch.optim.Adam(ActorCritic(actor, critic).parameters(), eps=1e-5, lr= self.learning_rate)
        # decayRate = 0.96
        # my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optim, gamma=decayRate)
        self.policy = PPOPolicy(
            actor=actor,
            critic=critic,
            optim=optim,
            dist_fn=dist,
            eps_clip=self.eps_clip,
        )

        # buffer_num should be equal to (suggested) or larger than #envs
        # print(f"env.spec: {env.spec}")
        self.train_envs = DummyVectorEnv(
            [lambda: gym.make(env.spec) for _ in range(self.training_num)]
        )
        self.test_envs = DummyVectorEnv(
            [lambda: gym.make(env.spec) for _ in range(self.test_num)]
        )

        self.train_collector = Collector(
            self.policy,
            self.train_envs,
            VectorReplayBuffer(self.buffer_size, len(self.train_envs)),
            exploration_noise=True
        )

        self.test_collector = Collector(self.policy, self.test_envs, exploration_noise=True)
        self.logdir = 'log'
        now = datetime.now().strftime("%y%m%d-%H%M%S")
        algo_name = "ppo"
        seed = 0
        task = "CampusymEnv"
        log_name = os.path.join(task, algo_name, str(seed), now)
        # log_path = os.path.join(self.logdir, 'wandb', 'CampusGymEnv')
        log_path = os.path.join(self.logdir, log_name)
        writer = SummaryWriter(log_path)
        self.logger = TensorboardLogger(writer)

    def save_agent(self):
        policy_dir = self.shared_config['directories']['policy_directory']
        if not os.path.exists(policy_dir):
            os.makedirs(policy_dir)

        file_path = os.path.join(policy_dir, f'ppo_{self.run_name}.pth')
        torch.save(self.policy.state_dict(), file_path)
        print(f"PPO saved to {file_path}")

    def train(self, alpha):
        """Train the agent."""
        # self.train_collector.collect(n_step=self.batch_size*self.training_num)
        prev_moving_avg = float("-inf")
        peak_steps = 0
        rewards_per_episode = []
        self.policy.train()
        for episode in tqdm(range(int(self.max_episodes))):  # total step
            collect_result = self.train_collector.collect(n_episode=1)
            print(f'res: {collect_result}')
            self.policy.update(0, self.train_collector.buffer, batch_size=self.batch_size, repeat=1)
            self.train_collector.reset_buffer(keep_statistics=True)
            avg_episode_return = collect_result['rew']/collect_result['n/st']
            rewards_per_episode.append(avg_episode_return)
            if episode >= self.moving_average_window - 1:
                window_rewards = rewards_per_episode[max(0, episode - self.moving_average_window + 1):episode + 1]
                moving_avg = np.mean(window_rewards)
                std_dev = np.std(window_rewards)
                wandb.log({
                    'Moving Average': moving_avg,
                    'Standard Deviation': std_dev,
                    'average_return': avg_episode_return,
                    'step': episode  # Ensure the x-axis is labeled correctly as 'Episodes'
                })
                # if moving_avg - prev_moving_avg < self.stopping_criterion:
                #     peak_steps += 1
                #     if peak_steps >= self.steps_before_stop:
                #         print(f"Stopping at episode {episode} with moving average reward {moving_avg}")
                #         break
                # else:
                #     peak_steps = 0
                # prev_moving_avg = moving_avg

            
        self.policy.eval()
        # self.test_envs.seed(100)
        self.test_collector.reset()
        result = self.test_collector.collect(n_episode=self.test_num)
        rews, lens = result["rews"], result["lens"]
        print(f"Final reward: {rews.mean()}, length: {lens.mean()}")
        torch.save(self.policy.state_dict(), 'dqn.pth')
        return {}

        

    def test(self, episodes, alpha, baseline_policy=None):
        """Test the trained agent with extended evaluation metrics."""
        pass

    def test_baseline_random(self, episodes, alpha, baseline_policy=None):
        """Test the trained agent with extended evaluation metrics."""
        pass