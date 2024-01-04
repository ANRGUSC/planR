import torch
import torch.nn as nn
import numpy as np

class Actor(nn.Module):
    def __init__(self, state_shape, action_shape):
        super(Actor, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(np.prod(state_shape), 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, np.prod(action_shape)),
        )
        self.output_dim = action_shape

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        batch = obs.shape[0]
        action = self.actor(obs.view(batch, -1))
        return action, state

class Critic(nn.Module):
    def __init__(self, state_shape, action_shape):
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(np.prod(state_shape) + np.prod(action_shape), 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, obs, action, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float)
        batch = obs.shape[0]
        q_value = self.critic(torch.cat([obs.view(batch, -1), action.view(batch, -1)], dim=-1))
        return q_value, state

class Net(nn.Module):
    def __init__(self, state_shape, action_shape):
        super(Net, self).__init__()
        self.actor = Actor(state_shape, action_shape)
        self.critic = Critic(state_shape, action_shape)

    def forward(self, obs, state=None, info={}):
        action, _ = self.actor(obs)
        q_value, _ = self.critic(obs, action)
        return action, q_value
