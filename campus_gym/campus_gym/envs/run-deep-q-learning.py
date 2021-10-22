import gym
from gym.envs.registration import register
import sys
import numpy as np
import json
from joblib import Parallel, delayed
import calendar
import time

import wandb

sys.path.append('../../..')
sys.path.append('../../../campus_digital_twin')
sys.path.append('../../../agents')
from agents.deepqlearning import DeepQAgent

a_list = np.arange(0.1, 0.9, 0.1)

# agent hyper-parameters
EPISODES = 5000
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
EXPLORATION_RATE = 0.2

register(
    id='campus-v0',
    entry_point='campus_gym_env:CampusGymEnv',
)
env = gym.make('campus-v0')


# When using wandb.
# - Set up your default hyperparameters
# hyperparameter_defaults = dict(
#     episodes=EPISODES,
#     discount_factor=DISCOUNT_FACTOR,
#     learning_rate=LEARNING_RATE,
#     exploration_rate=EXPLORATION_RATE
# )
# wandb.init(project='planr', entity='elizabethondula', config=hyperparameter_defaults, name="epsilongreedy")
# config = wandb.config


def run_training(alpha):
    """
    alpha: is a weight parameter that ranges between 0-1.
    It varies based on input data.

    Each training run has a training_name provided by a user.

    During training a unique input dataset is randomly generated that matches the initial parameters

    """
    gmt = time.gmtime()
    tr_name = calendar.timegm(gmt)

    # Create agent for a given environment using the agent hyper-parameters:
    deep_q_agent = DeepQAgent(env, tr_name, EPISODES, LEARNING_RATE,
                       DISCOUNT_FACTOR, EXPLORATION_RATE)
    # Train the agent using your chosen weight parameter (ALPHA)
    deep_q_agent.train(0.9)

    # Retrieve training and store for later evaluation.
    training_data = deep_q_agent.training_data

    with open(f'results/E-greedy/rewards/{tr_name}-{EPISODES}-{format(alpha, ".1f")}episode_rewards.json',
              'w+') as rfile:
        json.dump(training_data[0], rfile)
    with open(f'results/E-greedy/{tr_name}-{EPISODES}-{format(alpha, ".1f")}episode_allowed.json', 'w+') as afile:
        json.dump(training_data[1], afile)
    with open(f'results/E-greedy/{tr_name}-{EPISODES}-{format(alpha, ".1f")}episode_infected.json', 'w+') as ifile:
        json.dump(training_data[2], ifile)
    with open(f'results/E-greedy/{tr_name}-{EPISODES}-{format(alpha, ".1f")}episode_actions.json', 'w+') as actfile:
        json.dump(training_data[3], actfile)
    #
    # print("Done Training. Check results/E-greedy folder for training data")


if __name__ == '__main__':
    run_training(0.9)
    # alpha_list = np.arange(0, 1, 0.1)
    #
    # # Do training for each alpha in parallel
    # Parallel(n_jobs=-1)(delayed(run_training)(alpha) for alpha in alpha_list)
