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
import tianshou as ts
from datetime import datetime
from tqdm import tqdm
import wandb
from torch.nn import functional as F 
from tianshou.utils.net.discrete import Actor, Critic
from tianshou.utils.net.common import ActorCritic
from .net import Net
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

        # print("Check", env.observation_space, env.action_space)
        self.action_shape = np.prod(env.action_space.nvec)
        self.state_shape = env.observation_space.shape
        # print("state_shape: ", self.state_shape, "action_shape: ", self.action_shape)
        self.max_episodes = self.agent_config['agent']['max_episodes']
        self.learning_rate = self.agent_config['agent']['learning_rate']
        self.batch_size = 32
        self.hidden_shape = 128
        self.buffer_size = 10000
        self.moving_average_window = 100
        self.net = Net(
            self.state_shape,
            self.hidden_shape,
        )

        self.actor = Actor(self.net,self.action_shape, softmax_output=False)
        self.critic = Critic(self.net)
        
        # self.soft_update_op = ts.get_soft_update_op(1e-2, [self.actor, self.critic])

        # self.critc_loss = ts.losses.value_mse(self.critic)
        
        optim = torch.optim.Adam(ActorCritic(self.actor, self.critic).parameters(), eps=1e-5)
        self.policy = DDPGPolicy(
            actor=self.actor,
            critic=self.critic,
            actor_optim=optim,
            critic_optim=optim,
        )
        
        
        self.train_envs = DummyVectorEnv([lambda: gym.make(env.spec) for _ in range(self.batch_size)])
        self.test_envs = DummyVectorEnv([lambda: gym.make(env.spec) for _ in range(self.batch_size)])

        self.train_collector = Collector(
            self.policy,
            self.train_envs,
            VectorReplayBuffer(self.buffer_size, len(self.train_envs)),
            exploration_noise=True
            )
        
        self.test_collector = Collector(self.policy, self.test_envs, exploration_noise=True)
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
        self.tau = 0.001  # Soft update parameter
        self.action_noise = 0.1  # Exploration noise
        self.max_action = 1.0  # Maximum action magnitude

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def train(self, alpha):
        rewards_per_episode = []
        for episode in tqdm(range(int(self.max_episodes))):
        # for _ in tqdm(range(self.max_episodes)):
            collect_result = self.train_collector.collect(n_episode=1)
            # print(collect_result)
            # reward_mean = collect_result["rews"].mean()
            avg_episode_return = collect_result['rew']/collect_result['n/st']
            rewards_per_episode.append(avg_episode_return)
            if episode >= self.moving_average_window -1:
                window_rewards = rewards_per_episode[max(0, episode - self.moving_average_window + 1):episode + 1]
                moving_avg = np.mean(window_rewards)
                std_dev = np.std(window_rewards)
                wandb.log({
                    'Moving Average': moving_avg,
                    'Standard Deviation': std_dev,
                    'average_return': avg_episode_return,
                    'step': episode
                })
            # wandb.log({'reward': reward_mean})
            # print(f"collect_result: {collect_result}")

            # batch = self.train_collector.buffer.sample(self.batch_size)
            # obs = batch[0].obs
            # action = batch[0].act
            # reward = batch[0].rew
            # next_obs = batch[0].obs_next
            # done = batch[0].done

            # with torch.no_grad():
            #     # print(f"reward shape: {reward.shape}")
            #     # print(f"done shape: {done.shape}")
            #     next_action = self.target_actor(next_obs)[0]
            #     target_q = self.target_critic(next_obs, next_action)  
            #     reward = torch.from_numpy(reward).float()
            #     done = torch.from_numpy(done).float()  
            #     target_q = reward + (1 - done) * self.gamma * target_q[0]

            # current_q = self.critic(obs, action)
            # critic_loss = F.mse_loss(current_q[0], target_q[0])
            # self.critic_optim.zero_grad()
            # critic_loss.backward()
            # self.critic_optim.step()

            # actor_loss = -self.critic(obs, self.actor(obs))[0].mean()
            # self.actor_optim.zero_grad()
            # actor_loss.backward()
            # self.actor_optim.step()

            # self.soft_update(self.target_actor, self.actor, self.tau)
            # self.soft_update(self.target_critic, self.critic, self.tau)


    def test(self, episodes):
        pass
