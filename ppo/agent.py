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
        self.shared_config_path = shared_config_path
        self.agent_config_path = agent_config_path
        self.override_config = override_config
        print("Check", env.observation_space, env.action_space)
        # self.state_shape = np.prod(env.observation_space.nvec)
        # self.action_shape = (3,np.prod(self.env.action_space.nvec))
        self.state_shape = env.observation_space.nvec
        self.action_shape = env.action_space.nvec
        # self.action_shape = (3,3)
        # self.state_shape = env.observation_space.shape
        # self.action_shape = np.expand_dims( env.action_space.shape, axis=0)
        # print(env.observation_space.nvec, env.action_space.nvec)
        print(f"state_shape: {self.state_shape}, action_shape: {self.action_shape}")
        self.training_num = 8 # can change
        self.batch_size = 64
        # self.test_num = 100
        self.max_episodes = 10000
        self.test_num = 1
        self.hidden_shape = 128
        self.buffer_size = 10000
        self.net = Net(
            self.state_shape,
            self.hidden_shape
        )
        self.frames_stack = 4

        def dist(p):
            return torch.distributions.Categorical(logits=p)
        actor = Actor(self.net, self.action_shape, softmax_output=False)
        critic = Critic(self.net)
        optim = torch.optim.Adam(ActorCritic(actor, critic).parameters(), eps=1e-5)
        self.policy = PPOPolicy(
            actor=actor,
            critic=critic,
            optim=optim,
            dist_fn=dist,
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


    def train(self, alpha):
        """Train the agent."""
        # self.train_collector.collect(n_step=self.batch_size*self.training_num)
        for _ in tqdm(range(int(self.max_episodes))):  # total step
            collect_result = self.train_collector.collect(n_episode=1)
            print(f'res: {collect_result}')
            wandb.log({'reward': collect_result['rew'], })
            print(f"collect_result: {collect_result}")
            
        self.policy.eval()
        # self.test_envs.seed(100)
        self.test_collector.reset()
        result = self.test_collector.collect(n_episode=self.test_num)
        rews, lens = result["rews"], result["lens"]
        print(f"Final reward: {rews.mean()}, length: {lens.mean()}")
        return {}

        

    def test(self, episodes, alpha, baseline_policy=None):
        """Test the trained agent with extended evaluation metrics."""
        pass

    def test_baseline_random(self, episodes, alpha, baseline_policy=None):
        """Test the trained agent with extended evaluation metrics."""
        pass