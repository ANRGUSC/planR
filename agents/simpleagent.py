import random
import numpy as np
from tqdm import tqdm
import os
import copy
import itertools
import logging

def list_to_int(s, n):
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
        value = 2
    return value


# convert actions to discrete values 0,1,2
def action_conv_disc(action_or_state):
    discaction = []
    for i in (action_or_state):
        action_val = get_discrete_value(i)
        discaction.append(action_val)
    return discaction


# convert list of discrete values to 0 to 100 range
def disc_conv_action(discaction):
    action = []
    for i in range(len(discaction)):
        action.append((int)(discaction[i] * 50))
    return action


# get average of a list
def get_average_of_nested_list(list_to_avg):
    avg_ep_allowed = []
    for episode in list_to_avg:
        avg = int(sum(episode) / len(episode))
        avg_ep_allowed.append(avg)
    return avg_ep_allowed


class SimpleAgent():
    """An agent is initialized with the following hyperparameters for training:
    - learning_rate:
    - episodes:
    - discount_factor:
    - exploration_rate:

    The action is a proposal to a campus operator.
    It is currently comprised of 3 discrete levels:
     - 0: a class/course is going to be scheduled online.
     - 1: 50% of students are allowed to attend in-person while the rest attend online.
     - 2: 100% of students are allowed to attend in-person.
    """

    def __init__(self, env, run_name, episodes, learning_rate, discount_factor, exploration_rate):

        # hyperparameters
        self.max_episodes = episodes
        # Environment and run name
        self.env = env
        self.run_name = run_name
        self.training_data = []
        self.test_data = []
    def train(self, alpha):
        # Evaluation Metrics
        episode_rewards = {}
        episode_allowed = {}
        episode_infected_students = {}

        for i in tqdm(range(0, self.max_episodes)):
            logging.info(f'------ Episode: {i} -------')
            self.env.reset()
            done = False

            e_infected_students = []
            e_return = []
            e_allowed = []

            while not done:
                c_list_action = [50, 0, 0]
                action_alpha_list = [*c_list_action, alpha]
                observation, reward, done, info = self.env.step(action_alpha_list)
                week_reward = int(reward)
                e_return.append(week_reward)
                e_allowed = info['allowed']
                e_infected_students = info['infected']
                print(info, reward)
            episode_rewards[i] = e_return
            episode_allowed[i] = e_allowed
            episode_infected_students = e_infected_students

        self.training_data = [episode_rewards, episode_allowed, episode_infected_students]
