import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import gym
from gym.envs.registration import register
import sys
import pandas as pd
import numpy as np
import itertools
import json
import copy
from datetime import datetime
from collections import deque
sys.path.append('../../..')
sys.path.append('../../../campus_digital_twin')

def list_to_int(s, n):
    print(s, n)
    s.reverse()
    a = 0
    for i in range(len(s)):
        a = a + s[i] * pow(n, i)
    return a


# get the multidiscrete list from an index
def int_to_list(num, n, size):
    outlist = [0] * size
    i = 0
    while num != 0:
        bit = num % n
        outlist[i] = bit
        num = (int)(num / n)
        i = i + 1
    outlist.reverse()
    return outlist

def get_discrete_value(number):

    value = 0
    if number in range(0, 33):
        value = 0
    elif number in range(34, 66):
        value = 1
    elif number in range(67, 100):
        value =  2
    return value


# convert actions to discrete values 0,1,2
def action_conv_disc(action_or_state):
    discaction = []
    for i in (action_or_state):
        action_val = get_discrete_value(i)
        discaction.append(action_val)
        #discaction.append((int)(action[i] / 50 ))
    return discaction

# convert list of discrete values to 0 to 100 range
def disc_conv_action(discaction):
    action = []
    for i in range(len(discaction)):
        action.append((int)(discaction[i] * 50))
    return action

class QLAgent():
    def __init__(self, env):
        # hyperparameters
        self.max_episodes = 3000
        self.learning_rate = 0.1  # alpha
        self.discount_factor = 0.2  # gamma
        self.env = env
        self.exploration_rate = 0.2 # epsilon

        # initialize q table
        rows = np.prod(env.observation_space.nvec)
        columns = np.prod(env.action_space.nvec)
        self.q_table = np.zeros((rows, columns))
        self.possible_actions = [list(range(0, (k))) for k in self.env.action_space.nvec]
        self.possible_states = [list(range(0, (k))) for k in self.env.observation_space.nvec]
        self.all_actions = [str(i) for i in list(itertools.product(*self.possible_actions))]
        self.all_states = [str(i) for i in list(itertools.product(*self.possible_states))]
        self.training_data = []
        self.test_data = []

    def _policy(self, mode, state, exploration_rate=0):
        if mode=='train':
            if random.uniform(0, 1) > exploration_rate:
                dstate = str(tuple(action_conv_disc(state)))
                action = np.argmax(self.q_table[self.all_states.index(dstate)])
            else:
                sampled_actions = str(tuple(self.env.action_space.sample().tolist()))
                action = self.all_actions.index(sampled_actions)

        elif mode == 'test':
            dstate = str(tuple(action_conv_disc(state)))
            action = np.argmax(self.q_table[self.all_states.index(dstate)])

        return action

    def train(self):
        episodes = self.max_episodes
        learning_rate = self.learning_rate
        discount_factor = self.discount_factor
        exploration_rate = self.exploration_rate

        #reset Q table
        rows = np.prod(env.observation_space.nvec)
        columns = np.prod(env.action_space.nvec)
        self.q_table = np.zeros((rows, columns))
        state = self.env.reset()

        # Analysis Metrics
        episode_actions = {}
        episode_states = {}
        episode_rewards = {}
        episode_allowed = {}
        episode_infected_students = {}
        episode_qtable = {}
        exploration_episodes = 0
        exloitation_episodes = 0
        infected_students_df_dict = {}
        allowed_students_df_dict = {}

        for i in tqdm(range(0, episodes)):
            state = self.env.render()
            done = False

            e_infected_students = []
            e_return = []
            e_allowed = []
            actions_taken_until_done = []
            count = 0

            while not done:

                action = self._policy('train', state, exploration_rate)
                converted_state = str(tuple(action_conv_disc(state)))
                list_action = list(eval(self.all_actions[action]))
                observation, reward, done, info = env.step([i * 50 for i in list_action])
                old_value = self.q_table[self.all_states.index(converted_state), action]
                d_observation = str(tuple(action_conv_disc(observation)))
                next_max = np.max(self.q_table[self.all_states.index(d_observation)])
                new_value = (1 - self.learning_rate) * old_value + self.learning_rate * (reward[0] + self.discount_factor * next_max)
                self.q_table[self.all_states.index(converted_state), action] = new_value
                state = observation
                episode_reward = int(reward[0])
                e_infected_students.append(reward[2])
                x = copy.deepcopy(reward[1])
                e_allowed.append(x)
                e_return.append(episode_reward)
                actions_taken_until_done.append(list_action)

            episode_rewards[i] = e_return
            episode_allowed[i] = e_allowed
            episode_infected_students[i] = e_infected_students
            episode_actions[i] = actions_taken_until_done
            np.save(f"qtables/{i}{i}-qtable.npy", self.q_table)
        self.training_data = [episode_rewards, episode_allowed, episode_infected_students, episode_actions]

    def test(self):
        exploration_rate = self.exploration_rate
        state = self.env.reset()
        done = False
        weekly_rewards = []
        while not done:
            action = self._policy('test', state, exploration_rate)
            list_action = list(eval(self.all_actions[action]))  # 3 levels, 3 courses
            next_state, reward, done, info = env.step([i * 50 for i in list_action])
            state = next_state
            print(state, list_action, reward[0], reward[1])
            weekly_rewards.append(reward[0])
            print("***************************")

        return weekly_rewards

if __name__ == '__main__':
    register(
        id='campus-v0',
        entry_point='campus_gym_env:CampusGymEnv',
    )
    env = gym.make('campus-v0')
    agent = QLAgent(env)
    agent.train()
    print(agent.training_data[1])

    with open('rewards_0.45.json', 'w') as opfile:
        json.dump(agent.training_data[0], opfile)

    with open('allowed.json', 'w') as opfile:
        json.dump(agent.training_data[1], opfile)

    with open('infected.json', 'w') as opfile:
        json.dump(agent.training_data[2], opfile)

    with open('actions.json', 'w') as opfile:
        json.dump(agent.training_data[3], opfile)
    print("Testing Model")
    test_rewards = {}
    for i in range(0, 1):
        test_rewards[i] = agent.test()
    agent.test_data = test_rewards
    with open('testing_rewards.json', 'w') as opfile:
        json.dump(agent.test_data, opfile)




