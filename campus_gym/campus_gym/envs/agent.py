import tensorflow.compat.v1 as tf
import gym
from gym.envs.registration import register
import sys
sys.path.append('../../..')
sys.path.append('../../../campus_digital_twin')
import numpy as np
import random
from argparse import ArgumentParser
import tqdm

tf.disable_v2_behavior()
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


# convert actions to discrete values 0,1,2
def action_conv_disc(action):
    discaction = []
    for i in range(len(action)):
        discaction.append((int)(action[i] / 50))
    return discaction


# convert list of discrete values to 0 to 100 range
def disc_conv_action(discaction):
    action = []
    for i in range(len(discaction)):
        action.append((int)(discaction[i] * 50))
    return action

class Environment():
    def __init__(self):
        pass

    def CampusGymStudentInfection(self):
        register(
            id='campus-v0',
            entry_point='campus_gym_env:CampusGymEnv',
        )
        env = gym.make('campus-v0')  # try for different environements
        return env

class QAgent():

    def __init__(self, args, env):
        # set hyperparameters
        self.max_episodes = int(args.max_episodes)
        self.max_actions = int(args.max_actions)
        self.learning_rate = float(args.learning_rate)
        self.discount = float(args.discount)
        self.exploration_rate = float(args.exploration_rate)
        self.exploration_decay = 1.0 / float(args.max_episodes)

        # get environmnet
        self.env = env

        # initialize Q(s, a)

        self.Q = np.zeros([np.prod(env.observation_space.nvec), np.prod(env.action_space.nvec)])
        self.data = {}

    def _policy(self, mode, state, e_rate=0):
        if mode == 'train':
            if random.random() > e_rate:
                action = list_to_int(env.action_space.sample().tolist(), 3)  # Explore action space
                return action  # exploitation
            else:
                dstate = action_conv_disc(state)
                action = np.argmax(self.Q[list_to_int(dstate, 3)])
                return action  # exploration
        elif mode == 'test':
            dstate = action_conv_disc(state)
            action = np.argmax(self.Q[list_to_int(dstate, 3)])
            return action

    def train(self):
        # get hyper-parameters
        max_episodes = self.max_episodes
        max_actions = self.max_actions
        learning_rate = self.learning_rate
        discount = self.discount
        exploration_rate = self.exploration_rate
        exploration_decay = 1.0 / self.max_episodes
        epochs = 0

        # reset Q for initialize
        self.Q = np.zeros([np.prod(env.observation_space.nvec), np.prod(env.action_space.nvec)])

        # start training

        for i in (range(max_episodes)):
            state = self.env.reset()  # reset the environment per eisodes
            done = False
            while not done:
                action = self._policy('train', state, exploration_rate)
                list_action = int_to_list(action, 3, 3)
                new_state, reward, done, info = self.env.step([i * 50 for i in list_action])
                dstate = action_conv_disc(state)
                # The formulation of updating Q(s, a)
                self.Q[dstate, action] = self.Q[list_to_int(action_conv_disc(state), 3), action] + learning_rate * (
                            reward[0] + discount * np.max(self.Q[list_to_int(action_conv_disc(new_state), 3)]) -
                            self.Q[list_to_int(action_conv_disc(state), 3), action])
                state = new_state  # update the current state
                # if done == True:  # if fall in the hole or arrive to the goal, then this episode is terminated.
                #     break
                #print(self.displayQ())
                epochs += 1
                self.data[i] = [state, list_action, reward]

            if exploration_rate > 0.001:
                exploration_rate -= exploration_decay

    def test(self):
        # Setting hyper-parameters
        max_actions = self.max_actions
        state = self.env.reset()  # reset the environment
        for a in range(max_actions):
            self.env.render()  # show the environment states
            dstate = action_conv_disc(state)
            action = np.argmax(self.Q[list_to_int(dstate, 3)])
            #action = np.max(self.Q[list_to_int(action_conv_disc(state), 3)])  # take action with the Optimal Policy
            print(action)
            list_action = int_to_list(action, 3, 3)
            new_state, reward, done, info = self.env.step([i * 50 for i in list_action])  # arrive to next_state after taking the action
            state = new_state  # update current state
            if done:
                print("======")
                self.env.render()
                break
            print("======")
        self.env.close()

    def displayQ(self):
        print("Q\n", self.Q)

def arg_parse():
    parser = ArgumentParser()
    parser.add_argument("--max_episodes", help="max training episode", default=50)
    parser.add_argument("--max_actions", help="max actions per episode", default=99)
    parser.add_argument("--learning_rate", help="learning rate alpha for Q-learning", default=0.83)
    parser.add_argument("--discount", help="discount factpr for Q-learning", default=0.93)
    parser.add_argument("--exploration_rate", help="exploration_rate", default=1.0)
    return parser.parse_args()
# class DeepQAgent():
#     def __init__(self, args, env):
#         """
#         setting hyper-parameters and initialize NN model
#         """
#         self.max_episodes = 20000
#         self.max_actions = 99
#         self.discount = 0.93
#         self.exploration_rate = 1.0
#         self.exploration_decay = 1.0 / 20000
#         # get envirionment
#         self.env = env
#
#         # nn_model parameters
#         self.in_units = env.observation_space.nvec
#         self.out_units = env.action_space.nvec
#         self.hidden_units = int(args.hidden_units)
#
#         # construct nn model
#         self._nn_model(self.env)
#
#         # save nn model
#         self.saver = tf.train.Saver()
#
#     def _nn_model(self, env):
#         self.a0 = tf.placeholder(tf.float32, shape=[1, self.in_units])  # input layer
#         self.y = tf.placeholder(tf.float32, shape=[1, self.out_units])  # ouput layer
#
#         # from input layer to hidden layer
#         self.w1 = tf.Variable(tf.zeros([self.in_units, self.hidden_units], dtype=tf.float32))  # weight
#         self.b1 = tf.Variable(tf.random_uniform([self.hidden_units], 0, 0.01, dtype=tf.float32))  # bias
#         self.a1 = tf.nn.relu(tf.matmul(self.a0, self.w1) + self.b1)  # the ouput of hidden layer
#
#         # from hidden layer to output layer
#         self.w2 = tf.Variable(tf.zeros([self.hidden_units, self.out_units], dtype=tf.float32))  # weight
#         self.b2 = tf.Variable(tf.random_uniform([self.out_units], 0, 0.01, dtype=tf.float32))  # bias
#
#         # Q-value and Action
#         self.a2 = tf.matmul(self.a1, self.w2) + self.b2  # the predicted_y (Q-value) of four actions
#         self.action = tf.argmax(self.a2, 1)  # the agent would take the action which has maximum Q-value
#
#         # loss function
#         self.loss = tf.reduce_sum(tf.square(self.a2 - self.y))
#
#         # upate model
#         self.update_model = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(self.loss)
#
#     def train(self):
#
#         # get hyper parameters
#         max_episodes = self.max_episodes
#         max_actions = self.max_actions
#         discount = self.discount
#         exploration_rate = self.exploration_rate
#         exploration_decay = self.exploration_decay
#
#         # start training
#         with tf.Session() as sess:
#             sess.run(tf.global_variables_initializer())  # initialize tf variables
#             for i in range(max_episodes):
#                 state = env.reset()  # reset the environment per eisodes
#                 for j in range(max_actions):
#                     # get action and Q-values of all actions
#                     action, pred_Q = sess.run([self.action, self.a2], feed_dict={self.a0: np.eye(16)[state:state + 1]})
#
#                     # if explorating, then taking a random action instead
#                     if np.random.rand() < exploration_rate:
#                         action[0] = env.action_space.sample()
#
#                         # get nextQ in given next_state
#                     next_state, rewards, done, info = env.step(action[0])
#                     next_Q = sess.run(self.a2, feed_dict={self.a0: np.eye(16)[next_state:next_state + 1]})
#
#                     # update
#                     update_Q = pred_Q
#                     update_Q[0, action[0]] = rewards + discount * np.max(next_Q)
#
#                     sess.run([self.update_model],
#                              feed_dict={self.a0: np.identity(16)[state:state + 1], self.y: update_Q})
#                     state = next_state
#
#                     # if fall in the hole or arrive to the goal, then this episode is terminated.
#                     if done:
#                         if exploration_rate > 0.001:
#                             exploration_rate -= exploration_decay
#                         break
#             # save model
#             save_path = self.saver.save(sess, "./nn_model.ckpt")
#
#     def test(self, Q):
#
#         # get hyper-parameters
#         max_actions = self.max_actions
#         # start testing
#         with tf.Session() as sess:
#             # restore the model
#             sess.run(tf.global_variables_initializer())
#             saver = tf.train.import_meta_graph("./nn_model.ckpt.meta")  # restore model
#             saver.restore(sess, tf.train.latest_checkpoint('./'))  # restore variables
#
#             # testing result
#             state = env.reset()
#             for j in range(max_actions):
#                 env.render()  # show the environments
#                 # always take optimal action
#                 action, pred_Q = sess.run([self.action, self.a2], feed_dict={self.a0: np.eye(16)[state:state + 1]})
#                 # update
#                 next_state, rewards, done, info = env.step(action[0])
#                 state = next_state
#                 if done:
#                     env.render()
#                     break
#
#     def displayQ():
#         pass
#
#
# def dqarg_parse():
#     parser = ArgumentParser()
#     parser.add_argument("--max_episodes", help="max training episode", default=20000)
#     parser.add_argument("--max_actions", help="max actions per episode", default=99)
#     parser.add_argument("--discount", help="discount factpr for Q-learning", default=0.95)
#     parser.add_argument("--exploration_rate", help="exploration_rate", default=1.0)
#     parser.add_argument("--hidden_units", help="hidden units", default=10)
#     return parser.parse_args()

env = Environment().CampusGymStudentInfection()  # construct the environment

"""
Q learning Agent
"""
args = arg_parse()  # get hyper-parameters
agent = QAgent(args, env)  # get agent
agent.train()
q_learning_data = agent.data
print(q_learning_data)
# unpack the q_learning_data
e_current_states = {}
e_reward = {}
e_infected_students = {}
e_allowed_students = {}

for episode, data in q_learning_data.items():
    e_current_states[episode] = data[0]
    e_reward[episode] = data[2][0]
    e_infected_students[episode] = data[2][1]
    e_allowed_students[episode] = data[2][2]


# print("Testing Model")
# agent.test()

"""
Deep Q Learning Agent
"""
# dqargs = dqarg_parse()
# dqagent = DeepQAgent(dqargs, env) # get agent
# print("START TRAINING...")
# dqagent.train()
# print("\n\nTEST\n\n")
# dqagent.test()

