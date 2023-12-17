import matplotlib
import matplotlib.pyplot as plt
import torch
from tianshou.utils.net.common import ActorCritic
from tianshou.utils.net.discrete import Actor, Critic, IntrinsicCuriosityModule
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.utils import TensorboardLogger, WandbLogger
from ppo.net import Net
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



class PPOagent:
    def __init__(self, env, run_name, shared_config_path, agent_config_path=None, override_config=None):
        self.env = env
        self.run_name = run_name
        self.shared_config_path = shared_config_path
        self.agent_config_path = agent_config_path
        self.override_config = override_config
        state_shape = env.observation_space.shape or env.observation_space.n
        self.action_shape = env.action_space.shape or env.action_space.n
        self.training_episodes = 1000
        self.hidden_shape = 128
        self.buffer_size = 10000
        self.net = Net(
            state_shape,
            self.hidden_shape
        )
        self.frames_stack = 4
        def dist(p):
            return torch.distributions.Categorical(logits=p)
        actor = Actor(self.net, self.action_shape, softmax_output=False)
        critic = Critic(self.net)
        optim = torch.optim.Adam(ActorCritic(actor, critic).parameters(), eps=1e-5)
        policy = PPOPolicy(
            actor=actor,
            critic=critic,
            optim=optim,
            dist_fn=dist,
        )
        replay_buffer = ReplayBuffer(size=10000)

# here we set up a collector with a single environment
        collector = Collector(policy, env, buffer=replay_buffer)

        # the collector supports vectorized environments as well
        vec_buffer = VectorReplayBuffer(total_size=10000, buffer_num=3)
        # buffer_num should be equal to (suggested) or larger than #envs
        envs = DummyVectorEnv([lambda: env for _ in range(3)])

        collector = Collector(policy, envs, buffer=vec_buffer)

        # collect 3 episodes
        collector.collect(n_episode=3)
        # collect at least 2 steps
        collector.collect(n_step=2)
        # collect episodes with visual rendering ("render" is the sleep time between
        # rendering consecutive frames)
        collector.collect(n_episode=1, render=0.03)
        # logger = WandbLogger(
        #     save_interval=1,
        #     name=log_name.replace(os.path.sep, "__"),
        #     run_id=args.resume_id,
        #     config=args,
        #     project=args.wandb_project,
        # )


    def train(self, alpha):
        """Train the agent."""
        pass
        

    def test(self, episodes, alpha, baseline_policy=None):
        """Test the trained agent with extended evaluation metrics."""
        pass

    def test_baseline_random(self, episodes, alpha, baseline_policy=None):
        """Test the trained agent with extended evaluation metrics."""
        pass