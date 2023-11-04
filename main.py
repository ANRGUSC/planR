import os
import yaml
import gymnasium as gym
import numpy as np
import json
import io
import wandb
import argparse
from pathlib import Path
from campus_gym.envs.campus_gym_env import CampusGymEnv

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def initialize_environment(shared_config_path):
    shared_config = load_config(shared_config_path)
    env = gym.make(shared_config['environment']['environment_id'])
    return env, shared_config

def run_training(env, shared_config_path, alpha, agent_type, is_sweep=False):

    if not is_sweep:  # if not a sweep, initialize wandb here
        shared_config = load_config(shared_config_path)
        wandb.init(project=shared_config['wandb']['project'], entity=shared_config['wandb']['entity'])

    if wandb.run is None:
        raise RuntimeError(
            "wandb run has not been initialized. Please make sure wandb.init() is called before run_training.")

    tr_name = wandb.run.name
    agent_name = f"sweep_{tr_name}" if is_sweep else str(tr_name)

    agent_config_path = os.path.join('config', f'config_{agent_type}.yaml')
    agent_config = load_config(agent_config_path)
    wandb.config.update(agent_config)
    wandb.config.update({'alpha': alpha})

    # Here, get alpha value from wandb.config if is_sweep is True, else get it from args.alpha
    alpha = wandb.config.alpha if is_sweep else args.alpha

    AgentClass = getattr(__import__('q_learning.agent', fromlist=['QLearningAgent']), 'QLearningAgent')
    agent = AgentClass(env, agent_name,
                       shared_config_path=shared_config_path,
                       agent_config_path=agent_config_path)

    training_data = agent.train(alpha)

    print("Running Training...")
    return training_data, agent_name

def run_sweep(env, shared_config_path):
    shared_config = load_config(shared_config_path)
    run = wandb.init(project=shared_config['wandb']['project'], entity=shared_config['wandb']['entity'])
    config = run.config
    alpha = config.alpha
    agent_type = 'qlearning'

    run_data, training_name = run_training(env, shared_config_path, alpha, agent_type, is_sweep=True)
    print("Running Sweep...")

def run_evaluation(env, shared_config):
    # Placeholder for the logic of your evaluation functionality.
    # You can add your actual logic for evaluation later.
    print("Running Evaluation...")


def main():
    parser = argparse.ArgumentParser(description='Run training, evaluation, or a sweep.')
    parser.add_argument('mode', choices=['train', 'eval', 'sweep'], help='Mode to run the script in.')
    parser.add_argument('--alpha', type=float, default=0.6, help='Reward parameter alpha.')
    parser.add_argument('--agent_type', default='qlearning', help='Type of agent to use.')

    global args
    args = parser.parse_args()

    shared_config_path = os.path.join('config', 'config_shared.yaml')
    env, shared_config = initialize_environment(shared_config_path)

    if args.mode == 'train':
        run_training(env, shared_config_path, args.alpha, args.agent_type)

    elif args.mode == 'eval':
        run_evaluation(env, shared_config)
    elif args.mode == 'sweep':
        sweep_config_path = os.path.join('config', 'sweep.yaml')
        sweep_config = load_config(sweep_config_path)
        sweep_id = wandb.sweep(sweep_config, project=shared_config['wandb']['project'],
                               entity=shared_config['wandb']['entity'])
        wandb.agent(sweep_id, function=lambda: run_sweep(env, shared_config_path))
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == '__main__':
    main()
