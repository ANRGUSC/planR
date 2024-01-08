import gymnasium as gym
import matplotlib
import torch
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.policy import DDPGPolicy
from tianshou.env import DummyVectorEnv
from tianshou.utils import TensorboardLogger
from .utilities import load_config
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
from tqdm import tqdm
import wandb
from torch.nn import functional as F 
from ddpg.net import Actor, Critic
import numpy as np

class DDPGAgent:
    def __init__(self, env, run_name, shared_config_path, agent_config_path=None, override_config=None):
        self.env = env
        self.run_name = run_name
        self.shared_config_path = load_config(shared_config_path)
        
        if agent_config_path:
            self.agent_config = load_config(agent_config_path)
        else:
            self.agent_config = {}

        # self.shared_config_path = shared_config_path
        # self.agent_config_path = agent_config_path
        # self.override_config = override_config
            
        if override_config:
            self.agent_config.update(override_config)

        self.state_shape = env.observation_space.shape
        self.action_shape = np.prod(env.action_space.nvec)
        self.max_episodes = self.agent_config['agent']['max_episodes']
        self.learning_rate = self.agent_config['agent']['learning_rate']
        self.batch_size = 32
        self.hidden_shape = 128
        self.buffer_size = 100

        self.actor = Actor(self.state_shape, self.action_shape)
        self.critic = Critic(self.state_shape, self.action_shape)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.learning_rate)
        self.policy = DDPGPolicy(actor=self.actor, critic=self.critic, actor_optim=self.actor_optim, critic_optim=self.critic_optim)

        # self.train_envs = DummyVectorEnv([lambda: gym.make(env.id) for _ in range(self.batch_size)])
        print(f"env.spec: {env.spec}")
        self.train_envs = DummyVectorEnv([lambda: gym.make(env.spec) for _ in range(self.batch_size)])
        # self.train_collector = Collector(self.policy, self.train_envs, ReplayBuffer(self.buffer_size))
        self.train_collector = Collector(self.policy, self.train_envs, VectorReplayBuffer(self.buffer_size, len(self.train_envs)), exploration_noise=True)
        self.logdir = 'log'
        now = datetime.now().strftime("%y%m%d-%H%M%S")
        algo_name = "ddpg"
        seed = 0
        task = "CampusymEnv"
        log_name = os.path.join(task, algo_name, str(seed), now)
        log_path = os.path.join(self.logdir, log_name)
        writer = SummaryWriter(log_path)
        self.logger = TensorboardLogger(writer)

        self.gamma = 0.99  # Discount factor for Q-values
        self.tau = 0.005  # Soft update parameter
        self.action_noise = 0.1  # Exploration noise
        self.max_action = 1.0  # Maximum action magnitude

        self.target_actor = Actor(self.state_shape, self.action_shape)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic = Critic(self.state_shape, self.action_shape)
        self.target_critic.load_state_dict(self.critic.state_dict())

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def train(self, alpha):
        for _ in tqdm(range(self.max_episodes)):
            collect_result = self.train_collector.collect(n_step=self.batch_size)
            wandb.log({'reward': collect_result['rew'].mean()})
            print(f"collect_result: {collect_result}")

            batch = self.train_collector.buffer.sample_batch(self.batch_size)
            obs = batch['obs']
            action = batch['act']
            reward = batch['rew']
            next_obs = batch['obs_next']
            done = batch['done']

            with torch.no_grad():
                next_action = self.target_actor(next_obs)
                target_q = self.target_critic(next_obs, next_action)
                target_q = reward + (1 - done) * self.gamma * target_q

            current_q = self.critic(obs, action)
            critic_loss = F.mse_loss(current_q, target_q)
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

            actor_loss = -self.critic(obs, self.actor(obs)).mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            self.soft_update(self.target_actor, self.actor, self.tau)
            self.soft_update(self.target_critic, self.critic, self.tau)

    def test(self, episodes):
        pass
