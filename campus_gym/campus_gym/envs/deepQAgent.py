import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import gym
from gym.envs.registration import register
import sys
import pandas as pd
import numpy as np
from argparse import ArgumentParser
import itertools
import json
import copy
from numpyencoder import NumpyEncoder

sys.path.append('../../..')
sys.path.append('../../../campus_digital_twin')
plt.close("all")


# take a list of discrete values to an index
# needed because qtable needs integer index
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
        # discaction.append((int)(action[i] / 50 ))
    return discaction


# convert list of discrete values to 0 to 100 range
def disc_conv_action(discaction):
    action = []
    for i in range(len(discaction)):
        action.append((int)(discaction[i] * 50))
    return action


class DeepQLAgent():
    def __init__(self, args, env):

        # agent internal parameters
        self.env = env
        self.possible_actions_i = [list(range(0, (k))) for k in self.env.action_space.nvec]
        self.possible_states_i = [list(range(0, (k))) for k in self.env.observation_space.nvec]
        self.possible_actions = tf.convert_to_tensor(
            np.array([list(range(0, (k))) for k in self.env.action_space.nvec]))
        self.possible_states = tf.convert_to_tensor(
            np.array([list(range(0, (k))) for k in self.env.observation_space.nvec]))
        self.all_actions = [str(i) for i in list(itertools.product(*self.possible_actions_i))]
        self.all_states = [str(i) for i in list(itertools.product(*self.possible_states_i))]
        self.training_data = []
        self.test_data = []

        # hyperparameters
        self.max_episodes = 1
        self.max_semesters = 3
        self.learning_rate = 0.1  # alpha
        self.discount_factor = 0.2  # gamma
        self.exploration_rate = 0.2  # epsilon

        self.env = env

        # nn_model parameters
        self.in_units = int(np.prod(env.observation_space.nvec))
        self.out_units = int(np.prod(env.action_space.nvec))
        self.hidden_units = int(args.hidden_units)

        # construct nn model
        self._nn_model()

        # save nn model
        self.saver = tf.train.Saver()

    def _nn_model(self):

        self.a0 = tf.placeholder(tf.float32, shape=[1, self.in_units])  # input layer
        self.y = tf.placeholder(tf.float32, shape=[1, self.out_units])  # ouput layer

        # from input layer to hidden layer
        w1 = tf.Variable(tf.zeros([self.in_units, self.hidden_units], dtype=tf.float32), name='w1')  # weight
        b1 = tf.Variable(tf.random_uniform([self.hidden_units], 0, 0.01, dtype=tf.float32), name='b1')  # bias
        a1 = tf.nn.relu(tf.matmul(self.a0, w1) + b1)  # the ouput of hidden layer

        # from hidden layer to output layer
        w2 = tf.Variable(tf.zeros([self.hidden_units, self.out_units], dtype=tf.float32), name='w2')  # weight
        b2 = tf.Variable(tf.random_uniform([self.out_units], 0, 0.01, dtype=tf.float32), name='b2')  # bias

        # Q-value and Action
        self.a2 = tf.matmul(a1, w2) + b2  # the predicted_y (Q-value) of four actions
        self.action = tf.argmax(self.a2, 1)  # the agent would take the action which has maximum Q-value

        # loss function
        loss = tf.reduce_sum(tf.square(self.a2 - self.y))

        # upate model, minimizing loss function
        self.update_model = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(loss)

    def train(self):

        discount_factor = self.discount_factor
        exploration_rate = self.exploration_rate
        # Analysis Metrics
        episode_actions = {}
        episode_rewards = {}
        episode_allowed = {}
        episode_infected_students = {}

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(self.max_episodes):
                state = self.env.reset()
                done = False

                e_infected_students = []
                e_return = []
                e_allowed = []
                actions_taken_until_done = []
                count = 0
                while not done:
                    converted_state = str(tuple(action_conv_disc(state)))
                    dstate = self.all_states.index(converted_state)
                    action, pred_Q = sess.run([self.action, self.a2],
                                              feed_dict={self.a0: np.eye(81)[dstate:dstate + 1]})
                    if np.random.rand() < exploration_rate:  # exploration
                        action[0] = list_to_int(env.action_space.sample().tolist(), 3)  # take a random action
                    list_action = int_to_list(action[0], 3, 3)
                    next_state, rewards, done, info = env.step([i * 50 for i in list_action])
                    next_converted_state = str(tuple(action_conv_disc(next_state)))
                    next_dstate = self.all_states.index(converted_state)
                    next_Q = sess.run(self.a2, feed_dict={self.a0: np.eye(81)[next_dstate:next_dstate + 1]})

                    update_Q = pred_Q
                    update_Q[0, action[0]] = rewards[0] + discount_factor * np.max(next_Q)

                    sess.run([self.update_model],
                             feed_dict={self.a0: np.identity(81)[dstate:dstate + 1], self.y: update_Q})
                    state = next_state
                    print(rewards)

                    # update episode data (rewards, allowed students, infected students, states observed, actions taken)
                    episode_reward = int(rewards[0])
                    x = copy.deepcopy(rewards[1])
                    infected_students = rewards[2]

                    e_allowed.append(x)
                    e_infected_students.append(infected_students)
                    e_return.append(episode_reward)
                    actions_taken_until_done.append(list_action)

                episode_rewards[i] = e_return
                episode_allowed[i] = e_allowed
                episode_infected_students[i] = e_infected_students
                episode_actions[i] = actions_taken_until_done

            self.saver.save(sess, "./nn_model.ckpt")

        self.training_data = [episode_rewards, episode_allowed, episode_infected_students, episode_actions]

    def test(self):
        exploration_rate = self.exploration_rate
        state = self.env.reset()
        done = False
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.import_meta_graph("./nn_model.ckpt.meta")  # restore model
            saver.restore(sess, tf.train.latest_checkpoint('./'))  # 載入參數
            state = env.reset()
            weekly_rewards = []
            while not done:
                converted_state = str(tuple(action_conv_disc(state)))
                dstate = self.all_states.index(converted_state)
                action, pred_Q = sess.run([self.action, self.a2],
                                          feed_dict={self.a0: np.eye(81)[dstate:dstate + 1]})
                list_action = int_to_list(action[0], 3, 3)
                next_state, reward, done, info = env.step([i * 50 for i in list_action])
                state = next_state
                #print(state, list_action, reward[0], reward[1])
                weekly_rewards.append(reward[0])
                print("***************************")


def arg_parse():
    parser = ArgumentParser()
    parser.add_argument("--hidden_units", help="hidden units", default=10)
    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parse()
    register(
        id='campus-v0',
        entry_point='campus_gym_env:CampusGymEnv',
    )
    env = gym.make('campus-v0')  # try for different environements
    agent = DeepQLAgent(args, env)
    print("Start Training")
    agent.train()
    print("Testing Model")
    print(agent.test())

    with open('deep_rewards.json', 'w') as opfile:
        json.dump(agent.training_data[0], opfile, cls=NumpyEncoder)

    with open('deep_allowed.json', 'w') as opfile:
        json.dump(agent.training_data[1], opfile, cls=NumpyEncoder)

    with open('deep_infected.json', 'w') as opfile:
        json.dump(agent.training_data[2], opfile, cls=NumpyEncoder)

    with open('deep_actions.json', 'w') as opfile:
        json.dump(agent.training_data[3], opfile, cls=NumpyEncoder)
    print("Testing Model")
    test_rewards = {}
    for i in range(0, 10):
        test_rewards[i] = agent.test()
    agent.test_data = test_rewards
    with open('deep_testing_rewards.json', 'w') as opfile:
        json.dump(agent.test_data, opfile, cls=NumpyEncoder)
