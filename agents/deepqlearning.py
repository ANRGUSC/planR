import tensorflow as tf
import numpy as np
import random
from tqdm import tqdm
import itertools
import copy
import logging

tf.compat.v1.disable_eager_execution()
logging.basicConfig(filename='deepq.log', filemode='w+', format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)




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


class DeepQAgent:

    def __init__(self, env, run_name, episodes, learning_rate, discount_factor, exploration_rate):
        # set hyperparameters
        self.max_episodes = episodes
        # self.max_actions = int(args.max_actions)
        self.discount = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = 1.0 / float(episodes)
        self.learning_rate = learning_rate
        # get envirionment
        self.env = env
        self.possible_actions = [list(range(0, (k))) for k in self.env.action_space.nvec]
        self.possible_states = [list(range(0, (k))) for k in self.env.observation_space.nvec]
        self.all_actions = [str(i) for i in list(itertools.product(*self.possible_actions))]
        self.all_states = [str(i) for i in list(itertools.product(*self.possible_states))]

        # nn_model parameters
        self.in_units = len(self.possible_states)
        self.out_units = len(self.all_actions)
        self.hidden_units = 10

        # construct nn model
        self._nn_model()

        # save nn model
        self.saver = tf.compat.v1.train.Saver()
        self.training_data = []

    def _nn_model(self):
        """This is a dense neural network model with one hidden layer."""

        self.a0 = tf.compat.v1.placeholder(tf.float32, shape=[1, self.in_units])  # input layer
        self.y = tf.compat.v1.placeholder(tf.float32, shape=[1, self.out_units])  # ouput layer

        # from input layer to hidden layer
        w1 = tf.Variable(tf.zeros([self.in_units, self.hidden_units], dtype=tf.float32), name='w1')  # weight
        b1 = tf.Variable(tf.compat.v1.random_uniform([self.hidden_units], 0, 0.01, dtype=tf.float32), name='b1')  # bias
        a1 = tf.nn.relu(tf.matmul(self.a0, w1) + b1)  # the output of hidden layer

        # from hidden layer to output layer
        w2 = tf.Variable(tf.zeros([self.hidden_units, self.out_units], dtype=tf.float32), name='w2')  # weight
        b2 = tf.Variable(tf.compat.v1.random_uniform([self.out_units], 0, 0.01, dtype=tf.float32), name='b2')  # bias

        # Q-value and Action
        self.a2 = tf.matmul(a1, w2) + b2  # the predicted_y (Q-value)
        self.action = tf.argmax(self.a2, 1)  # the agent would take the action which has maximum Q-value

        # loss function
        loss = tf.reduce_sum(tf.square(self.a2 - self.y))

        # update model, minimizing loss function
        self.update_model = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(loss)

    def train(self, alpha):
        # hyper parameter
        discount = self.discount
        exploration_rate = self.exploration_rate
        exploration_decay = self.exploration_decay
        episode_actions = {}
        episode_rewards = {}
        episode_allowed = {}
        episode_infected_students = {}

        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())

            for i in tqdm(range(self.max_episodes)):
                logging.info(f'------ Episode: {i} -------')
                state = self.env.reset()
                done = False
                e_infected_students = []
                e_return = []
                e_allowed = []
                actions_taken_until_done = []

                while not done:
                    # get action and Q-values of all actions
                    action, pred_Q = sess.run([self.action, self.a2],
                                              feed_dict={self.a0: [state]})

                    if np.random.rand() < exploration_rate:  # exploration
                        sampled_actions = str(tuple(self.env.action_space.sample().tolist()))
                        action = [self.all_actions.index(sampled_actions)]
                    list_action = list(eval(self.all_actions[action[0]]))
                    c_list_action = [i * 50 for i in list_action]
                    action_alpha_list = [*c_list_action, alpha]
                    next_state, reward, done, info = self.env.step(action_alpha_list)
                    next_Q = sess.run(self.a2, feed_dict={self.a0: [next_state]})
                    update_Q = pred_Q
                    update_Q[0, action[0]] = reward[0] + discount * np.max(next_Q)
                    sess.run([self.update_model],
                             feed_dict={self.a0: [next_state], self.y: update_Q})
                    state = next_state
                    if exploration_rate > 0.001:
                        exploration_rate -= exploration_decay
                    week_reward = reward[0]
                    e_infected_students.append(reward[2])
                    allowed = copy.deepcopy(reward[1])
                    e_allowed.append(allowed)
                    e_return.append(week_reward)
                    actions_taken_until_done.append(list_action)
                    logging.info(f'Action taken: {list_action}')
                    logging.info(f'Reward: {reward[0]}')
                    logging.info(f'Allowed: {reward[1]}')
                    logging.info(f'Infected: {reward[2]}')
                    logging.info("*********************************")

                episode_rewards[i] = e_return
                episode_allowed[i] = e_allowed
                episode_infected_students[i] = e_infected_students
                episode_actions[i] = actions_taken_until_done

            self.training_data = [episode_rewards, episode_allowed, episode_infected_students, episode_actions]
            self.saver.save(sess, "nn_model.ckpt")
