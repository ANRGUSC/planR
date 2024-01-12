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
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.results_directory = self.shared_config['directories']['results_directory']
        self.results_subdirectory = os.path.join(self.results_directory, run_name, timestamp)
        os.makedirs(self.results_subdirectory, exist_ok=True)
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
        # self.policy.eval()
        actual_rewards = []
        predicted_rewards = []
        visited_state_counts = {}
        buf = ReplayBuffer(size=20)
        for episode in tqdm(range(int(self.max_episodes))):  # total step
        # for episode in tqdm(range(2)):  # total step
        
            e_return = []
            e_allowed = []
            e_infected_students = []
            e_return = []
            e_community_risk = []
            e_predicted_rewards = []
            total_reward = 0
            done = False
            state = self.env.reset()
            
            obs = state[0].reshape(1, -1)
            while not done:
                batch = Batch(obs=obs, act=None, rew=None, done=None, obs_next=None, info=None, policy=None)
                val = self.policy(batch)
                action = val.act[0]
                e_predicted_rewards.append(val.logits[0][action-1].item())
                state, reward, done, _, info = self.env.step(action)
                next_obs = np.array(state).reshape(1, -1)
                buf.add(Batch(obs=obs, act=action, rew=reward, done=done, obs_next=next_obs, terminated = done, truncated=0, info=info, policy=None))
                obs = next_obs
                discrete_state = str(tuple(i//10 for i in state))
                if discrete_state not in visited_state_counts:
                    visited_state_counts[discrete_state] = 1
                else:
                    visited_state_counts[discrete_state] += 1
                    
                week_reward = float(reward)
                total_reward += week_reward
                e_return.append(week_reward)
                e_allowed.append(info['allowed'])
                e_infected_students.append(info['infected'])
                e_community_risk.append(info['community_risk'])

                # Example usage:
            # If enough episodes have been run, check for convergence
            self.policy.update(0, buf, batch_size=self.batch_size, repeat=1)
            avg_episode_return = sum(e_return) / len(e_return)
            rewards_per_episode.append(avg_episode_return)
            if episode >= self.moving_average_window - 1:
                window_rewards = rewards_per_episode[max(0, episode - self.moving_average_window + 1):episode + 1]
                moving_avg = np.mean(window_rewards)
                std_dev = np.std(window_rewards)

                # Store the current moving average for comparison in the next episode
                

                # Log the moving average and standard deviation along with the episode number
                wandb.log({
                    'Moving Average': moving_avg,
                    'Standard Deviation': std_dev,
                    'average_return': total_reward/len(e_return),
                    'step': episode  # Ensure the x-axis is labeled correctly as 'Episodes'
                })
            predicted_rewards.append(e_predicted_rewards)
            actual_rewards.append(e_return)
        
        visit_counts = list(visited_state_counts.values())
        states = list(visited_state_counts.keys())
        states_visited_path = states_visited_viz(states, visit_counts,alpha, self.results_subdirectory)
        wandb.log({"States Visited": [wandb.Image(states_visited_path)]})#IMP

        avg_rewards = [sum(lst) / len(lst) for lst in actual_rewards]
        # Pass actual and predicted rewards to visualizer
        print(actual_rewards)
        print(predicted_rewards)
        explained_variance_path = visualize_explained_variance(actual_rewards, predicted_rewards, self.results_subdirectory, self.max_episodes)
        wandb.log({"Explained Variance": [wandb.Image(explained_variance_path)]})#IMP
            


            
        self.policy.eval()
        # self.test_envs.seed(100)
        self.test_collector.reset()
        # result = self.test_collector.collect(n_episode=self.test_num)
        # # rews, lens = result["rews"], result["lens"]
        # print(f"Final reward: {rews.mean()}, length: {lens.mean()}")
        torch.save(self.policy.state_dict(), 'dqn.pth')
        return {}

        

    def test(self, episodes, alpha, baseline_policy=None):
        """Test the trained agent with extended evaluation metrics."""
        pass

    def test_baseline_random(self, episodes, alpha, baseline_policy=None):
        """Test the trained agent with extended evaluation metrics."""
        pass