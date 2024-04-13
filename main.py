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

def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def initialize_environment(shared_config_path):
    shared_config = load_config(shared_config_path)
    env = gym.make(shared_config['environment']['environment_id'])
    return env, shared_config

def format_agent_class_name(agent_type):
    # Define special cases where acronyms should remain in uppercase
    special_acronyms = {
        'ppo': 'PPO',
        'dqn': 'DQN',
        'a2c': 'A2C',
        'ddpg': 'DDPG',
        'sac': 'SAC',
        'td3': 'TD3',
        # Add other special acronyms here
    }

    parts = agent_type.split('_')
    formatted_parts = [special_acronyms.get(part, part.capitalize()) for part in parts]
    return ''.join(formatted_parts) + 'Agent'

def run_training(env, shared_config_path, alpha, agent_type, is_sweep=False):

    if not is_sweep:  # if not a sweep, initialize wandb here
        shared_config = load_config(shared_config_path)
        wandb.init(project=shared_config['wandb']['project'], entity=shared_config['wandb']['entity'])

    if wandb.run is None:
        raise RuntimeError(
            "wandb run has not been initialized. Please make sure wandb.init() is called before run_training.")

    tr_name = wandb.run.name + '_' + str(alpha)
    agent_name = f"sweep_{tr_name}" if is_sweep else str(tr_name)

    agent_config_path = os.path.join('config', f'config_{agent_type}.yaml')
    agent_config = load_config(agent_config_path)
    wandb.config.update(agent_config)
    wandb.config.update({'alpha': alpha})
    effective_alpha = wandb.config.alpha if is_sweep else alpha
    env.alpha = effective_alpha

    # Dynamically import the agent class based on agent_type
    AgentModule = __import__(f'{agent_type}.agent', fromlist=[f'{format_agent_class_name(agent_type)}'])
    AgentClass = getattr(AgentModule, f'{format_agent_class_name(agent_type)}')
    if is_sweep:
        agent = AgentClass(env, agent_name,
                           shared_config_path=shared_config_path,
                           override_config=dict(wandb.config))
    else:
        agent = AgentClass(env, agent_name,
                           shared_config_path=shared_config_path,
                           agent_config_path=agent_config_path)

    agent.train(effective_alpha)

    # Save the run_name for later use
    with open('train_run_names.txt', 'a') as file:
        file.write(agent_name + '\n')

    print("Done Training with alpha: ", alpha, "agent_type: ", agent_type, "agent_name: ", agent_name)
    return agent_name

def run_sweep(env, shared_config_path, agent_type):
    shared_config = load_config(shared_config_path)
    run = wandb.init(project=shared_config['wandb']['project'], entity=shared_config['wandb']['entity'])
    config = run.config
    alpha = config.alpha
    print(alpha)
    agent_type = agent_type

    run_training(env, shared_config_path, alpha, agent_type, is_sweep=True)
    print("Running Sweep...")

def run_evaluation(env, shared_config_path, agent_type, alpha, run_name):
    print("Running Evaluation...")

    # Load agent configuration
    agent_config_path = os.path.join('config', f'config_{agent_type}.yaml')
    load_config(agent_config_path)

    AgentModule = __import__(f'{agent_type}.agent', fromlist=[f'{format_agent_class_name(agent_type)}'])
    AgentClass = getattr(AgentModule, f'{format_agent_class_name(agent_type)}')
    agent = AgentClass(env, run_name,
                       shared_config_path=shared_config_path,
                       agent_config_path=os.path.join('config', f'config_{agent_type}.yaml'))

    # Load the trained Q-table (assuming it's saved after training)
    q_table_path = os.path.join('policy', f'q_table_{run_name}.npy')
    agent.q_table = np.load(q_table_path)

    # Run the test
    test_episodes = 5 # Define the number of test episodes
    evaluation_metrics = agent.test(test_episodes, alpha)

    # Print or process the evaluation metrics as needed
    print("Evaluation Metrics:", evaluation_metrics)


def run_evaluation_random(env, shared_config_path, agent_type, alpha, run_name):
    print("Running Evaluation...")

    # Load agent configuration
    agent_config_path = os.path.join('config', f'config_{agent_type}.yaml')
    load_config(agent_config_path)

    # Initialize agent
    AgentModule = __import__(f'{agent_type}.agent', fromlist=[f'{format_agent_class_name(agent_type)}'])
    AgentClass = getattr(AgentModule, f'{format_agent_class_name(agent_type)}')
    agent = AgentClass(env, run_name,
                       shared_config_path=shared_config_path,
                       agent_config_path=os.path.join('config', f'config_{agent_type}.yaml'))
    # Run the test
    test_episodes = 4  # Define the number of test episodes
    evaluation_metrics = agent.test_baseline_random(test_episodes, alpha)

    # Print or process the evaluation metrics as needed
    print("Evaluation Metrics for random agent:", evaluation_metrics)

def main():
    parser = argparse.ArgumentParser(description='Run training, evaluation, or a sweep.')
    parser.add_argument('mode', choices=['train', 'eval', 'random', 'sweep'], help='Mode to run the script in.')
    parser.add_argument('--alpha', type=float, default=0.9, help='Reward parameter alpha.')
    parser.add_argument('--agent_type', default='q_learning', help='Type of agent to use.')
    parser.add_argument('--run_name', default=None, help='Unique name for the training run or evaluation.')

    global args
    args = parser.parse_args()

    shared_config_path = os.path.join('config', 'config_shared.yaml')
    env, shared_config = initialize_environment(shared_config_path)

    if args.mode == 'train':
        run_training(env, shared_config_path, args.alpha, args.agent_type)

    elif args.mode == 'eval':
        run_evaluation(env, shared_config_path, args.agent_type, args.alpha, args.run_name)

    elif args.mode == 'random':
        run_evaluation_random(env, shared_config_path, args.agent_type, args.alpha, args.run_name)

    elif args.mode == 'sweep':
        sweep_config_path = os.path.join('config', 'sweep.yaml')
        sweep_config = load_config(sweep_config_path)
        sweep_id = wandb.sweep(sweep_config, project=shared_config['wandb']['project'],
                               entity=shared_config['wandb']['entity'])
        wandb.agent(sweep_id, function=lambda: run_sweep(env, shared_config_path, args.agent_type))
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == '__main__':
    main()
