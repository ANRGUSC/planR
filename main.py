import os
import yaml
import gymnasium as gym
import numpy as np
import json
import io
import wandb
from pathlib import Path
from campus_gym.envs.campus_gym_env import CampusGymEnv


def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


# Load shared configurations
shared_config_path = os.path.join('config', 'config_shared.yaml')
shared_config = load_config(shared_config_path)

# # Initialize wandb
# wandb.init(project=shared_config['wandb']['project'], entity=shared_config['wandb']['entity'])

# Initialize environment
env = gym.make(shared_config['environment']['environment_id'])


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def run_training(alpha, agent_type, is_sweep=False):
    tr_name = wandb.run.name
    agent_name = f"sweep_{tr_name}" if is_sweep else str(tr_name)

    # Load agent-specific configuration and initialize the appropriate agent
    agent_config_path = os.path.join('config', f'config_{agent_type}.yaml')
    agent_config = load_config(agent_config_path)
    wandb.config.update(agent_config)  # Update wandb config with agent-specific configurations

    # Dynamically import the agent class from the specified agent_type package
    AgentClass = getattr(__import__('q_learning.agent', fromlist=['QLearningAgent']), 'QLearningAgent')

    agent = AgentClass(env, agent_name,
                       shared_config_path=shared_config_path,
                       agent_config_path=agent_config_path)

    training_data = agent.train(alpha)

    return training_data, agent_name


def train_sweep():
    # Initialize wandb
    run = wandb.init(project=shared_config['wandb']['project'], entity=shared_config['wandb']['entity'])
    config = run.config  # Get the configuration parameters for this run of the sweep
    alpha = config.alpha  # Extract alpha from the configuration parameters
    agent_type = 'qlearning'

    # Run the training with the overridden parameters
    # Run the training with the overridden parameters
    run_data, training_name = run_training(alpha, agent_type, is_sweep=True)  # Pass True for is_sweep parameter


if __name__ == '__main__':
    # Load sweep configurations
    sweep_config_path = os.path.join('config', 'sweep.yaml')
    sweep_config = load_config(sweep_config_path)

    # Initialize the sweep
    sweep_id = wandb.sweep(sweep_config, project=shared_config['wandb']['project'],
                           entity=shared_config['wandb']['entity'])

    # Run the sweep
    wandb.agent(sweep_id, function=train_sweep)