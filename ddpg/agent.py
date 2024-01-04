import gym
import matplotlib
import torch
from tianshou.data import Collector, ReplayBuffer
from tianshou.policy import DDPGPolicy
from tianshou.env import DummyVectorEnv
from tianshou.utils import TensorboardLogger
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
from tqdm import tqdm
import wandb

from ddpg.net import Actor, Critic

class DDPGAgent:
    def __init__(self, env, run_name, shared_config_path, agent_config_path=None, override_config=None):
        self.env = env
        self.run_name = run_name
        self.shared_config_path = shared_config_path
        self.agent_config_path = agent_config_path
        self.override_config = override_config

        self.state_shape = env.observation_space.shape[0]
        self.action_shape = env.action_space.shape[0]
        self.max_episodes = 10000
        self.batch_size = 64
        self.hidden_shape = 128
        self.buffer_size = 10000

        self.actor = Actor(self.state_shape, self.action_shape)
        self.critic = Critic(self.state_shape, self.action_shape)
        self.policy = DDPGPolicy(actor=self.actor, critic=self.critic, actor_optim=torch.optim.Adam, critic_optim=torch.optim.Adam)

        self.train_envs = DummyVectorEnv([lambda: gym.make(env.spec) for _ in range(self.batch_size)])
        self.train_collector = Collector(self.policy, self.train_envs, ReplayBuffer(self.buffer_size))
        self.logdir = 'log'
        now = datetime.now().strftime("%y%m%d-%H%M%S")
        algo_name = "ddpg"
        seed = 0
        task = "CampusymEnv"
        log_name = os.path.join(task, algo_name, str(seed), now)
        log_path = os.path.join(self.logdir, log_name)
        writer = SummaryWriter(log_path)
        self.logger = TensorboardLogger(writer)

    def train(self):
        for _ in tqdm(range(self.max_episodes)):
            collect_result = self.train_collector.collect(n_step=self.batch_size)
            wandb.log({'reward': collect_result['rew'].mean()})
            print(f"collect_result: {collect_result}")

        self.policy.eval()
        result = self.train_collector.collect(n_step=self.batch_size)
        rews, lens = result["rews"], result["lens"]
        print(f"Final reward: {rews.mean()}, length: {lens.mean()}")
        return {}

    def test(self, episodes):
        pass
