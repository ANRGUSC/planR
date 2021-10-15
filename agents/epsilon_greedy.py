import random
import numpy as np
from tqdm import tqdm
import os
import copy
import itertools

# import wandb

RESULTS = os.path.join(os.getcwd(), 'results')


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


class Agent():
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
        self.learning_rate = learning_rate  # alpha
        self.discount_factor = discount_factor  # gamma
        self.exploration_rate = exploration_rate  # epsilon

        # Environment and run name
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
        self.test_data = []

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
        """The tabular approach is used for training the agent where.

        Given a state i.e observation(no.of infected students):
        1. Action is taken using the epsilon-greedy approach.
        2. The Q table is then updated based on the Bellman equation.
        3. The actions taken, rewards and observations(allowed and infected)
        are then logged for later analysis."""
        # reset Q table
        rows = np.prod(self.env.observation_space.nvec)
        columns = np.prod(self.env.action_space.nvec)
        self.q_table = np.zeros((rows, columns))

        # Evaluation Metrics
        episode_actions = {}
        episode_rewards = {}
        episode_allowed = {}
        episode_infected_students = {}

        for i in tqdm(range(0, self.max_episodes)):
            state = self.env.reset()
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

                week_reward = int(reward[0])
                e_infected_students.append(reward[2])
                x = copy.deepcopy(reward[1])
                e_allowed.append(x)
                e_return.append(week_reward)
                actions_taken_until_done.append(list_action)

            episode_rewards[i] = e_return
            episode_allowed[i] = e_allowed
            episode_infected_students[i] = e_infected_students
            episode_actions[i] = actions_taken_until_done
            reward = int(sum(e_return) / len(e_return))
            allowed = [sum(x) / len(x) for x in zip(*e_allowed)]
            allowed_l = int(sum(allowed) / len(allowed))
            infected = [sum(x) / len(x) for x in zip(*e_infected_students)]
            infected_l = int(sum(infected) / len(infected))
            # Get average and log
            #wandb.log({'reward': reward, 'allowed': allowed_l, 'infected': infected_l})
            #np.save(f"{RESULTS}/qtables/{self.run_name}-{i}-qtable.npy", self.q_table)

        self.training_data = [episode_rewards, episode_allowed, episode_infected_students, episode_actions]
