import random
import numpy as np
from tqdm import tqdm
import os
import copy
import itertools
import logging
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


import wandb
wandb.init(project="campus-plan", entity="leezo")

RESULTS = os.path.join(os.getcwd(), 'results')
# logging.basicConfig(filename='indoor_risk_model.log', filemode='w+', format='%(name)s - %(levelname)s - %(message)s',
#                     level=logging.INFO)

random.seed(10)
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
    if number in range(0, 11):
        value = 0
    elif number in range(12, 21):
        value = 1
    elif number in range(22, 31):
        value = 2
    elif number in range(32, 41):
        value = 3
    elif number in range(42, 51):
        value = 4
    elif number in range(52, 61):
        value = 5
    elif number in range(61, 71):
        value = 6

    elif number in range(71, 81):
        value = 7
    elif number in range(81, 91):
        value = 8
    elif number in range(91, 101):
        value = 9
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

        wandb.config.learning_rate = self.learning_rate
        wandb.config.discount_factor = self.discount_factor
        wandb.config.max_expisodes = self.max_episodes

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
        self.states = list(itertools.product(*self.possible_states))

        self.training_data = []
        self.test_data = []
        self.exploration_decay = 1.0 / float(self.max_episodes)

    def _policy(self, mode, state):
        global action
        if mode == 'train':
            if random.uniform(0, 1) > self.exploration_rate:
                dstate = str(tuple(state))
                action = np.argmax(self.q_table[self.all_states.index(dstate)])

            else:
                sampled_actions = str(tuple(self.env.action_space.sample().tolist()))
                action = self.all_actions.index(sampled_actions)

        elif mode == 'test':
            dstate = str(tuple(state))
            action = np.argmax(self.q_table[self.all_states.index(dstate)])

        return action

    def train(self):
        """The tabular approach is used for training the agent.

        Given a state i.e observation(no.of infected students):
        1. Action is taken using the epsilon-greedy approach.
        2. The Q table is then updated based on the Bellman equation.
        3. The actions taken, rewards and observations(infected students)
        are then logged for later analysis."""
        # reset Q table
        rows = np.prod(self.env.observation_space.nvec)
        columns = np.prod(self.env.action_space.nvec)
        self.q_table = np.zeros((rows, columns))
        print("Q table", rows, columns)

        # Evaluation Metrics
        episode_actions = {}
        episode_rewards = {}
        episode_allowed = {}
        episode_infected_students = {}

        for i in tqdm(range(0, self.max_episodes)):
            logging.info(f'------ Episode: {i} -------')
            state = self.env.reset()
            done = False

            e_infected_students = []
            e_return = []
            e_allowed = []
            actions_taken_until_done = []

            while not done:
                action = self._policy('train', state)
                converted_state = str(tuple(state))
                list_action = list(eval(self.all_actions[action]))
                # logging.info(f'Action taken: {list_action}')
                # c_list_action = [i * 50 for i in list_action]
                #action_alpha_list = [*c_list_action, alpha]
                observation, reward, done, info = self.env.step(list_action)

                # updating the Q-table
                old_value = self.q_table[self.all_states.index(converted_state), action]
                d_observation = str(tuple(observation))
                next_max = np.max(self.q_table[self.all_states.index(d_observation)])
                new_value = (1 - self.learning_rate) * old_value + self.learning_rate * (
                        reward + self.discount_factor * next_max)
                self.q_table[self.all_states.index(converted_state), action] = new_value

                state = observation


                week_reward = int(reward)
                e_return.append(week_reward)
                e_allowed = info['allowed']
                e_infected_students = info['infected']
                logging.info(f'Reward: {reward}')
                logging.info("*********************************")
                #print(info, reward)
                if self.exploration_rate > 0.001:
                    self.exploration_rate -= self.exploration_decay
            #print(sum(e_return)/len(e_return))
            episode_rewards[i] = e_return
            episode_allowed[i] = e_allowed
            episode_infected_students = e_infected_students
            wandb.log({'reward': sum(e_return) / len(e_return)})

            # Get average and log
            #wandb.log({'reward': reward, 'allowed': allowed_l, 'infected': infected_l})
            #np.save(f"{RESULTS}/qtable/{self.run_name}-{i}-qtable.npy", self.q_table)



        self.training_data = [episode_rewards, episode_allowed, episode_infected_students]

    def test(self):
        max_actions = 2

        for a in range(max_actions):
            state = self.env.reset() # reset the environment
            print(state)
            done = False

            while not done:
                dstate = str(tuple(state))
                action = np.argmax(self.q_table[self.all_states.index(dstate)])
                list_action = list(eval(self.all_actions[action]))
                new_state, reward, done, info = self.env.step(list_action) # arrive to next_state after taking the action
                state = new_state # update current state


    def test_all_states(self):
        # Random samples
        # student_status = random.sample(range(0, 100), 15)
        # community_risk = np.random.uniform(low= 0.1, high = 0.9, size=15)
        # actions = []
        #print("All States", self.states)
        actions = {}
        for i in self.states:
            action = np.argmax(self.q_table[self.all_states.index(str(i))])
            actions[(i[0], i[1])] = action

        #print(actions)
        # student_status = []
        # s = 0
        # for i in range(15):
        #     student_status.append(s)
        #     s += 5
        #
        # community_risk = np.linspace(0,1,15)
        # actions = {}
        # print(student_status)
        # print("******************************************************")
        # print(community_risk)

        # for i in student_status:
        #     for j in community_risk:
        #         state = [i, int(j*100)]
        #         formatted_state = np.array(action_conv_disc(state))
        #         dstate = str(tuple(formatted_state))
        #         action = np.argmax(self.q_table[self.all_states.index(dstate)])
        #         actions[(i, j)] = action
        #
        x_values = []
        y_values = []
        colors = []
        for k, v in actions.items():
            x_values.append(k[0])
            y_values.append(k[1])
            colors.append(v)

        c = ListedColormap(['red', 'green', 'blue'])
        s = plt.scatter(y_values, x_values, c=colors, cmap=c)
        plt.xlabel("Community risk")
        plt.ylabel("Infected students")
        plt.legend(*s.legend_elements(), loc='upper left')
        plt.show()


    # put in two for-loops, one to go through each value of infected students (student status) and the other to go through each value of community_risk
    # inside these for_lopops, construct the state as the tuple [infected_students, community_risk*100]
    # then construct the state in the appropriate format by doing:
    # formatted_state  = np.array ( action_conv_disc (state) )
    # call dstate = str(tuple(formatted_state))
    # now find out what action would be taken for this formmatted states as:
    # nmp.argmmax(self.q_table[self.all_states.index(dstate)])
    # record this action and go to the next input state in the forloops



