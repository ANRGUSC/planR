import random
import itertools
import numpy as np
from tqdm import tqdm
import sys
import os
import copy
import csv

RESULTS = os.path.join(os.getcwd(), 'results', 'E-greedy')

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


class QLAgent():

    def __init__(self, env, run_name):
        # hyperparameters
        self.max_episodes = 30
        self.learning_rate = 0.1  # alpha
        self.discount_factor = 0.2  # gamma
        self.exploration_rate = 0.2  # epsilon

        self.env = env
        self.run_name = run_name

        # initialize q table
        rows = np.prod(env.observation_space.nvec)
        columns = np.prod(env.action_space.nvec)
        self.q_table = np.zeros((rows, columns))
        self.possible_actions = [list(range(0, (k))) for k in self.env.action_space.nvec]
        self.possible_states = [list(range(0, (k))) for k in self.env.observation_space.nvec]
        self.all_actions = [str(i) for i in list(itertools.product(*self.possible_actions))]
        self.all_states = [str(i) for i in list(itertools.product(*self.possible_states))]
        self.training_data = []
        self.test_data = {}

    def _policy(self, mode, state):
        global action
        if mode == 'train':
            if random.uniform(0, 1) > self.exploration_rate:
                dstate = str(tuple(action_conv_disc(state)))
                action = np.argmax(self.q_table[self.all_states.index(dstate)])
            else:
                sampled_actions = str(tuple(self.env.action_space.sample().tolist()))
                action = self.all_actions.index(sampled_actions)

        elif mode == 'test':
            dstate = str(tuple(action_conv_disc(state)))
            action = np.argmax(self.q_table[self.all_states.index(dstate)])

        return action

    def train(self, alpha):
        # reset Q table
        rows = np.prod(self.env.observation_space.nvec)
        columns = np.prod(self.env.action_space.nvec)
        self.q_table = np.zeros((rows, columns))

        # Analysis Metrics
        episode_actions = {}
        episode_rewards = {}
        episode_allowed = {}
        episode_infected_students = {}

        for i in tqdm(range(0, self.max_episodes)):
            state = self.env.render()
            done = False

            e_infected_students = []
            e_return = []
            e_allowed = []
            actions_taken_until_done = []

            while not done:
                action = self._policy('train', state)
                converted_state = str(tuple(action_conv_disc(state)))
                list_action = list(eval(self.all_actions[action]))
                c_list_action = [i * 50 for i in list_action]
                action_alpha_list = [*c_list_action, alpha]
                observation, reward, done, info = self.env.step(action_alpha_list)
                old_value = self.q_table[self.all_states.index(converted_state), action]
                d_observation = str(tuple(action_conv_disc(observation)))
                next_max = np.max(self.q_table[self.all_states.index(d_observation)])
                new_value = (1 - self.learning_rate) * old_value + self.learning_rate * (
                        reward[0] + self.discount_factor * next_max)
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
            np.save(f"qtables/{self.run_name}-{i}-qtable.npy", self.q_table)

        with open(RESULTS+'actions.csv','w') as f:
            for key in episode_actions.keys():
                f.write("%s,%s\n"%(key,episode_actions[key]))
        with open(RESULTS+'infected.csv','w') as f:
            for key in episode_infected_students.keys():
                f.write("%s,%s\n"%(key,episode_infected_students[key]))
        with open(RESULTS+'allowed.csv','w') as f:
            for key in episode_allowed.keys():
                f.write("%s,%s\n"%(key,episode_allowed[key]))
        with open(RESULTS+'rewards.csv','w') as f:
            for key in episode_rewards.keys():
                f.write("%s,%s\n"%(key,episode_rewards[key]))

        self.training_data = [episode_rewards, episode_allowed, episode_infected_students, episode_actions]

    def test(self, alpha):
        exploration_rate = self.exploration_rate
        state = self.env.reset()
        done = False
        weekly_rewards = []
        allowed_students = []
        infected_students = []
        while not done:
            action = self._policy('test', state, exploration_rate)
            list_action = list(eval(self.all_actions[action]))
            next_state, reward, done, info = self.env.step([i * 50 for i in list_action], alpha)
            state = next_state
            weekly_rewards.append(reward[0])
            allowed_students.append(copy.deepcopy(reward[1]))
            infected_students.append(reward[2])

        return weekly_rewards, allowed_students, infected_students