import gym
from gym.envs.registration import register
import sys
import numpy as np
import json
from joblib import Parallel, delayed

# import wandb

sys.path.append('../../..')
sys.path.append('../../../campus_digital_twin')
sys.path.append('../../../agents')
from agents.epsilon_greedy import Agent

a_list = np.arange(0.1, 0.9, 0.1)

# agent hyper-parameters
EPISODES = 5000
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
EXPLORATION_RATE = 0.2
ALPHA = 0.9

register(
    id='campus-v0',
    entry_point='campus_gym_env:CampusGymEnv',
)
env = gym.make('campus-v0')


# TODO:  To use when setting up wandb.
# - Set up your default hyperparameters
# hyperparameter_defaults = dict(
#     episodes=EPISODES,
#     discount_factor=DISCOUNT_FACTOR,
#     learning_rate=LEARNING_RATE,
#     exploration_rate=EXPLORATION_RATE,
#     alpha=ALPHA,  # Reward parameter that can be
# )
# wandb.init(project='planr', entity='elizabethondula', config=hyperparameter_defaults, name="epsilongreedy")
# config = wandb.config


def run_training(ALPHA):
    """
    ALPHA is a weight parameter that ranges between 0-1.
    It varies based on input data.

    Each training run has a training_name provided by a user.

    An assumption is hereby made that that during each training,
    a unique input dataset is randomly generated.
    """
    training_name = sys.argv[0]

    # Create agent for a given environment using the agent hyper-parameters:
    e_greedy_agent = Agent(env, training_name, EPISODES, LEARNING_RATE,
                           DISCOUNT_FACTOR, EXPLORATION_RATE)
    # Train the agent using your chosen weight parameter (ALPHA)
    e_greedy_agent.train(ALPHA)

    # Retrieve training and testing data and store as file for evaluation.
    training_data = e_greedy_agent.training_data

    with open(f'results/E-greedy/{training_name}-{EPISODES}-{format(ALPHA, ".1f")}episode_rewards.json', 'w+') as rewardfile:
        json.dump(training_data[0], rewardfile)
    with open(f'results/E-greedy/{training_name}-{EPISODES}-{format(ALPHA, ".1f")}episode_allowed.json', 'w+') as allowedfile:
        json.dump(training_data[1], allowedfile)
    with open(f'results/E-greedy/{training_name}-{EPISODES}-{format(ALPHA, ".1f")}episode_infected.json', 'w+') as infectedfile:
        json.dump(training_data[2], infectedfile)
    with open(f'results/E-greedy/{training_name}-{EPISODES}-{format(ALPHA, ".1f")}episode_actions.json', 'w+') as actionsfile:
        json.dump(training_data[3], actionsfile)

    print("Done Training. Check results/E-greedy folder for training data")


if __name__ == '__main__':
    alpha_list = np.arange(0, 1, 0.1)
    Parallel(n_jobs=-1)(delayed(run_training)(alpha) for alpha in alpha_list)

    # TODO: # A method for testing is yet to be determined.

    # Dummy Test
    # e_greedy_agent.test(ALPHA)
    # test_data = e_greedy_agent.test_data
    #
    # with open('results/E-greedy/test_rewards.json', 'w+') as test_rfile:
    #     json.dump(test_data[0], test_rfile)
    # with open('results/E-greedy/test_allowed.json', 'w+') as test_afile:
    #     json.dump(test_data[1], test_afile)
    # with open('results/E-greedy/test_infected.json', 'w+') as test_ifile:
    #     json.dump(test_data[2], test_ifile)
    # with open('results/E-greedy/test_actions.json', 'w+') as test_acfile:
    #     json.dump(test_data[3], test_acfile)
    #
    # print("Done Testing. Check results/E-greedy folder for testing data")
