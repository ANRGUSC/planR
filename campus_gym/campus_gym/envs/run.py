from tqdm import tqdm
import random
import gym
from gym.envs.registration import register
import sys
import numpy as np
import itertools
import json
from joblib import Parallel, delayed
import copy
import wandb

sys.path.append('../../..')
sys.path.append('../../../campus_digital_twin')
sys.path.append('../../../agents')
from agents.epsilon_greedy import QLAgent

a_list = np.arange(0.1, 0.9, 0.1)

# Agent parameters
EPISODES = 3000
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.2
EXPLORATION_RATE = 0.2
ALPHA = 0.85

if __name__ == '__main__':
    register(
        id='campus-v0',
        entry_point='campus_gym_env:CampusGymEnv',
    )
    env = gym.make('campus-v0')
    run_name = "Test1"

    agent = QLAgent(env, run_name, EPISODES, LEARNING_RATE,
                    DISCOUNT_FACTOR, EXPLORATION_RATE)
    agent.train(ALPHA)

    # Retrieve training and testing data and store as file for later analysis
    training_data = agent.training_data

    with open('results/E-greedy/episode_rewards.json', 'w+') as rewardfile:
        json.dump(training_data[0], rewardfile)
    with open('results/E-greedy/episode_allowed.json', 'w+') as allowedfile:
        json.dump(training_data[1], allowedfile)
    with open('results/E-greedy/episode_infected.json', 'w+') as infectedfile:
        json.dump(training_data[2], infectedfile)
    with open('results/E-greedy/episode_actions.json', 'w+') as actionsfile:
        json.dump(training_data[3], actionsfile)

    print("Done Training. Check results/E-greedy folder for training data")

    print("Testing Model...")

    agent.test(ALPHA)
    test_data = agent.test_data

    with open('results/E-greedy/test_rewards.json', 'w+') as test_rfile:
        json.dump(test_data[0], test_rfile)
    with open('results/E-greedy/test_allowed.json', 'w+') as test_afile:
        json.dump(test_data[1], test_afile)
    with open('results/E-greedy/test_infected.json', 'w+') as test_ifile:
        json.dump(test_data[2], test_ifile)
    with open('results/E-greedy/test_actions.json', 'w+') as test_acfile:
        json.dump(test_data[3], test_acfile)

    print("Done Testing. Check results/E-greedy folder for testing data")
