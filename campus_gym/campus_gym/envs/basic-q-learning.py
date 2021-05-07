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
a = 0.60
a_list = np.arange(0.1, 0.9, 0.1)


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
    def __init__(self, env):
        # hyperparameters
        self.max_episodes = 3
        self.learning_rate = 0.1  # alpha
        self.discount_factor = 0.2  # gamma
        self.env = env
        self.exploration_rate = 0.2  # epsilon

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

    def _policy(self, mode, state, exploration_rate=0):
        global action
        if mode == 'train':
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
        exploration_rate = self.exploration_rate

        # reset Q table
        rows = np.prod(env.observation_space.nvec)
        columns = np.prod(env.action_space.nvec)
        self.q_table = np.zeros((rows, columns))

        # Analysis Metrics
        episode_actions = {}
        episode_rewards = {}
        episode_allowed = {}
        episode_infected_students = {}

        for i in tqdm(range(0, episodes)):
            state = self.env.render()
            done = False

            e_infected_students = []
            e_return = []
            e_allowed = []
            actions_taken_until_done = []

            while not done:
                action = self._policy('train', state, exploration_rate)
                converted_state = str(tuple(action_conv_disc(state)))
                list_action = list(eval(self.all_actions[action]))
                observation, reward, done, info = env.step([i * 50 for i in list_action])
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
            next_state, reward, done, info = env.step([i * 50 for i in list_action])
            state = next_state
            weekly_rewards.append(reward[0])
            allowed_students.append(copy.deepcopy(reward[1]))
            infected_students.append(reward[2])

        return weekly_rewards, allowed_students, infected_students


if __name__ == '__main__':
    register(
        id='campus-v0',
        entry_point='campus_gym_env:CampusGymEnv',
    )
    env = gym.make('campus-v0')
    agent = QLAgent(env)
    agent.train()
    print("Done Training")

    with open((str(a) + 'rewards.json'), 'w') as opfile:
        json.dump(agent.training_data[0], opfile)

    with open((str(a) + 'allowed.json'), 'w') as opfile:
        json.dump(agent.training_data[1], opfile)

    with open((str(a) + 'infected.json'), 'w') as opfile:
        json.dump(agent.training_data[2], opfile)

    with open((str(a) + 'actions.json'), 'w') as opfile:
        json.dump(agent.training_data[3], opfile)
    print("Testing Model")

    test_rewards = {}
    test_allowed = {}
    test_infected = {}
    test_alpha_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for i in test_alpha_list:
        test_rewards[i] = agent.test(i)[0]
        test_allowed[i] = copy.deepcopy(agent.test(i)[1])
        test_infected[i] = copy.deepcopy(agent.test(i)[2])
    agent.test_data['Rewards'] = copy.deepcopy(test_rewards)
    agent.test_data['Allowed'] = copy.deepcopy(test_allowed)
    agent.test_data['Infected'] = test_infected

    with open((str(a) + 'testing_rewards.json'), 'w') as reward_file:
        json.dump(agent.test_data['Rewards'], reward_file)
    with open((str(a) + 'testing_allowed.json'), 'w') as allowed_file:
        json.dump(agent.test_data['Allowed'], allowed_file)
    with open((str(a) + 'testing_infected.json'), 'w') as infected_file:
        json.dump(agent.test_data['Infected'], infected_file)

    print("Done Testing")
