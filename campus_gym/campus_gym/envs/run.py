from tqdm import tqdm
import random
import gym
from gym.envs.registration import register
import sys
import numpy as np
import itertools
import json
import copy

sys.path.append('../../..')
sys.path.append('../../../campus_digital_twin')
sys.path.append('../../../agents')
from agents.epsilon_greedy import QLAgent
a = 0.7
a_list = np.arange(0.1, 0.9, 0.1)


if __name__ == '__main__':
    register(
        id='campus-v0',
        entry_point='campus_gym_env:CampusGymEnv',
    )
    env = gym.make('campus-v0')
    run_name = "Test1"
    agent = QLAgent(env, run_name)
    alpha_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    agent.train(alpha_list[7])
    print("Done Training")

    print("Testing Model")

    # test_rewards = {}
    # test_allowed = {}
    # test_infected = {}
    # for j in range (len(test_alpha_list)):
    #     for i in test_alpha_list:
    #         test_rewards[i] = agent.test(i)[0]
    #         test_allowed[i] = copy.deepcopy(agent.test(i)[1])
    #         test_infected[i] = copy.deepcopy(agent.test(i)[2])
    #
    #     agent.test_data['Rewards'] = copy.deepcopy(test_rewards)
    #     agent.test_data['Allowed'] = copy.deepcopy(test_allowed)
    #     agent.test_data['Infected'] = test_infected
    #
    #     with open((str(i) + 'testing_rewards.json'), 'w') as reward_file:
    #         json.dump(agent.test_data['Rewards'], reward_file)
    #     with open((str(i) + 'testing_allowed.json'), 'w') as allowed_file:
    #         json.dump(agent.test_data['Allowed'], allowed_file)
    #     with open((str(i) + 'testing_infected.json'), 'w') as infected_file:
    #         json.dump(agent.test_data['Infected'], infected_file)
    #
    # print("Done Testing")
