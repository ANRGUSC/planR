import os
import yaml
import gymnasium as gym
import numpy as np
import json
import calendar
import multiprocessing as mp
from functools import partial
# from agents.qlearning import Agent
# from agents.deepqlearning import DeepQAgent
# from agents.simpleagent import SimpleAgent
from dqn_pearl.agent import DQNPearlAgent
from dqn_cleanrl.agent import DQNCleanrlAgent
from dqn_cleanrl.random_agent import RandomAgent

# from agents.dqn import KerasAgent
from pathlib import Path
import wandb
import random
import codecs, json
import io
import wandb
import argparse
from pathlib import Path
from campus_gym.envs.campus_gym_env import CampusGymEnv
from Pearl.pearl.utils.instantiations.environments.gym_environment import GymEnvironment


print('asdasdas',wandb.__path__)

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
    # env = GymEnvironment(shared_config['environment']['environment_id'])
    return env, shared_config

def run_training(env, shared_config_path, alpha, agent_type, is_sweep=False):

    print('training')
    # print('run training sweep config .....', dict(wandb.config))
    if not is_sweep:  # if not a sweep, initialize wandb here
        shared_config = load_config(shared_config_path)
        wandb.init(project=shared_config['wandb']['project'])

    if wandb.run is None:
        raise RuntimeError(
            "wandb run has not been initialized. Please make sure wandb.init() is called before run_training.")

    tr_name = wandb.run.name + '_' + str(alpha)
    agent_name = f"sweep_{tr_name}" if is_sweep else str(tr_name)

    agent_config_path = os.path.join('config', f'config_{agent_type}.yaml')
    agent_config = load_config(agent_config_path)
    sweep_config = {
        'agent': dict(wandb.config)
    }
    # agent_config.update(dict(wandb.config))
    # print('agent config ', agent_config)
    # wandb.config.update(agent_config)
    wandb.config.update({'alpha': alpha})
    effective_alpha = wandb.config.alpha if is_sweep else alpha
    print('effectivealphaaaa ', effective_alpha)
    print('after update sweep config .....',sweep_config)


    # Here, get alpha value from wandb.config if is_sweep is True, else get it from args.alpha
    # alpha = wandb.config.alpha if is_sweep else args.alpha

    AgentClass = getattr(__import__('dqn_cleanrl.agent', fromlist=['DQNCleanrlAgent']), 'DQNCleanrlAgent')
    if is_sweep:
        print('SHARED CONFIG PATH', shared_config_path)
        # print('override ', wandb.config)
        agent = AgentClass(env, agent_name,
                           shared_config_path=shared_config_path,
                           override_config=sweep_config)
    else:
        agent = AgentClass(env, agent_name,
                           shared_config_path=shared_config_path,
                           agent_config_path=agent_config_path)

    # agent = RandomAgent(env, '123',
    #                    shared_config_path=shared_config_path,
    #                    agent_config_path=agent_config_path)
    print('agent', agent)
    agent.train(effective_alpha)

    # Save the run_name for later use
    with open('quals_final_run_names.txt', 'a') as file:
        file.write(agent_name + '\n')

    print("Done Training...")
    return agent_name

def run_sweep(env, shared_config_path):
    shared_config = load_config(shared_config_path)
    run = wandb.init(project=shared_config['wandb']['project'], entity=shared_config['wandb']['entity'])
    config = run.config
    alpha = config.alpha
    # print('alphaaa', alpha)
    agent_type = 'qlearning'
    # print('sweep config .....', dict(wandb.config))

    run_training(env, shared_config_path, alpha, agent_type, is_sweep=True)
    print("Running Sweep...")

# def run_sweep_wrapper(config=None):
#     shared_config = load_config(shared_config_path)
#     run = wandb.init(project=shared_config['wandb']['project'], entity=shared_config['wandb']['entity'])
#     print('sweep wrapper config', dict(wandb.config))
#     run_sweep(env, shared_config_path)

def run_evaluation(env, shared_config_path, agent_type, alpha, run_name):
    print("Running Evaluation...")

    # Load agent configuration
    agent_config_path = os.path.join('config', f'config_{agent_type}.yaml')
    load_config(agent_config_path)

    # # Load the last run_name
    # with open('last_run_name.txt', 'r') as file:
    #     run_name = file.read().strip()

    # Initialize agent
    AgentClass = getattr(__import__('q_learning.agent', fromlist=['QLearningAgent']), 'QLearningAgent')
    agent = AgentClass(env, run_name,
                       shared_config_path=shared_config_path,
                       agent_config_path=os.path.join('config', f'config_{agent_type}.yaml'))

    # Load the trained Q-table (assuming it's saved after training)
    q_table_path = os.path.join('policy', f'q_table_{run_name}.npy')
    agent.q_table = np.load(q_table_path)

    # Run the test
    test_episodes = 4  # Define the number of test episodes
    evaluation_metrics = agent.test(test_episodes, alpha)

    # Print or process the evaluation metrics as needed
    print("Evaluation Metrics:", evaluation_metrics)


def run_evaluation_random(env, shared_config_path, agent_type, alpha, run_name):
    print("Running Evaluation...")

    # Load agent configuration
    agent_config_path = os.path.join('config', f'config_{agent_type}.yaml')
    load_config(agent_config_path)

    # # Load the last run_name
    # with open('last_run_name.txt', 'r') as file:
    #     run_name = file.read().strip()

    # Initialize agent
    AgentClass = getattr(__import__('q_learning.agent', fromlist=['QLearningAgent']), 'QLearningAgent')
    agent = AgentClass(env, run_name,
                       shared_config_path=shared_config_path,
                       agent_config_path=os.path.join('config', f'config_{agent_type}.yaml'))

    # # Load the trained Q-table (assuming it's saved after training)
    # q_table_path = os.path.join('policy', f'q_table_{run_name}.npy')
    # agent.q_table = np.load(q_table_path)

    # Run the test
    test_episodes = 4  # Define the number of test episodes
    evaluation_metrics = agent.test_baseline_random(test_episodes, alpha)

    # Print or process the evaluation metrics as needed
    print("Evaluation Metrics for random agent:", evaluation_metrics)

def main():
    parser = argparse.ArgumentParser(description='Run training, evaluation, or a sweep.')
    parser.add_argument('mode', choices=['train', 'eval', 'random', 'sweep'], help='Mode to run the script in.')
    parser.add_argument('--alpha', type=float, default=0.2, help='Reward parameter alpha.')
    parser.add_argument('--agent_type', default='qlearning', help='Type of agent to use.')
    parser.add_argument('--run_name', default='abcd', help='Unique name for the training run or evaluation.')

    global args
    args = parser.parse_args()

    global shared_config_path
    shared_config_path = os.path.join('config', 'config_shared.yaml')
    global env, shared_config
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
        print(shared_config['wandb']['project'])
        # sweep_id = wandb.sweep(sweep_config, project=shared_config['wandb']['project'],
        #                        entity=shared_config['wandb']['entity'])
        sweep_id = wandb.sweep(sweep_config, project=shared_config['wandb']['project'])
        wandb.agent(sweep_id, function=lambda: run_sweep(env, shared_config_path))
        # wandb.agent(sweep_id, function=run_sweep_wrapper)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == '__main__':
    main()
