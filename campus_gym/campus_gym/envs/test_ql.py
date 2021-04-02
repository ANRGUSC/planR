# %%time
import time
from tqdm import tqdm
"""Training the agent"""
import matplotlib.pyplot as plt
import random
# from IPython.display import clear_output
import gym
from gym.envs.registration import register
import sys
import pandas as pd

sys.path.append('../../..')
sys.path.append('../../../campus_digital_twin')
plt.close("all")
import numpy as np

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
register(
            id='campus-v0',
            entry_point='campus_gym_env:CampusGymEnv',
        )
env = gym.make('campus-v0')  # try for different environements

q_table = np.zeros([np.prod(env.observation_space.nvec), np.prod(env.action_space.nvec)])
print(q_table.shape)
# print(type(q_table))
# print(q_table.size)
# print(np.size(q_table[2]))

# Hyperparameters
alpha = 0.1
gamma = 0.2
epsilon = 0.6

# For plotting metrics
all_epochs = []
all_penalties = []
episode_actions = {}
episode_states = {}
episode_rewards = {}
episode_allowed_students = {}
episode_infected_students = {}
episode_qtable = {}
exploration_episodes = 0
exloitation_episodes = 0
infected_students_df_dict ={}
allowed_students_df_dict = {}
episodes = []
for i in tqdm(range(1, 100)):
    state = env.reset()
    epochs, penalties, reward, = 0, 0, 0
    done = False
    # Because we don't know how many timesteps will be there we will store values until done.
    # The len of the list should be equal to the number of weeks.
    e_allowed_students = []
    e_infected_students = []
    e_return = []
    actions_taken_until_done = []
    while not done:
        episode_reward = 0
        allowed_students = 0
        infected_students = 0
        if random.uniform(0, 1) < epsilon:
            action = list_to_int(env.action_space.sample().tolist(), 3)  # Explore action space
            exploration_episodes +=1
        else:
            exloitation_episodes += 1
            #print(state)
            dstate = action_conv_disc(state)

            action = np.argmax(q_table[list_to_int(dstate, 3)])  # Exploit learned values

        list_action = int_to_list(action, 3, 3)

        #        print(int_to_list(action,3, 5))

        # This is where the agent interacts with the environment.
        # Could be implemented
        next_state, reward, done, info = env.step([i * 50 for i in list_action])
        # I think the above line could be changed to use disc_conv_action
        #print(list_action)
        #print(reward)
        episode_reward = int(reward[0])
        allowed_students = reward[1]
        infected_students = reward[2]
        old_value = q_table[list_to_int(action_conv_disc(state), 3), action]
        next_max = np.max(q_table[list_to_int(action_conv_disc(next_state), 3)])

        new_value = (1 - alpha) * old_value + alpha * (reward[0] + gamma * next_max)

        q_table[list_to_int(action_conv_disc(state), 3), action] = new_value
        e_infected_students.append(infected_students)
        e_allowed_students.append(allowed_students)
        e_return.append(episode_reward)
        actions_taken_until_done.append(list_action)

        if reward == -10:
            penalties += 1

        state = next_state
        epochs += 1
    episode_rewards[i] = e_return
    episode_allowed_students[i] = e_allowed_students
    episode_infected_students[i] = e_infected_students
    episode_actions[i] = actions_taken_until_done
    episode_qtable[i] = q_table
    columns = ['c1', 'c2', 'c3']
    infected_students_df_dict = {k: pd.DataFrame(v, columns=columns) for k, v in episode_infected_students.items()}
    allowed_students_df_dict = {k: pd.DataFrame(v, columns=columns) for k, v in episode_allowed_students.items()}

# infected students per episode
course_ai  = []
course_bi = []
course_ci = []
for episode, df in infected_students_df_dict.items():
   course_ai.append(int(df['c1'].sum()/15))
   course_bi.append(int(df['c2'].sum()/15))
   course_ci.append(int(df['c3'].sum())/15)

# Plot infected students
# scatter

plt.scatter(episodes, course_ai)
plt.scatter(episodes, course_bi)
plt.scatter(episodes, course_ci)
plt.xlabel('Episodes')
plt.ylabel('Infected students')
plt.legend()
plt.show()
# bar
width =0.3
plt.bar(np.arange(len(course_ai)), course_ai, width=width)
plt.bar(np.arange(len(course_bi))+ width, course_bi, width=width)
plt.bar(np.arange(len(course_ci))+ width + width, course_ci, width=width)
plt.xlabel('Episodes')
plt.ylabel('Infected Students')
plt.show()

### Allowed Students

course_aa  = []
course_ba = []
course_ca = []
for episode, df in allowed_students_df_dict.items():
   course_aa.append(int(df['c1'].sum()/15))
   course_ba.append(int(df['c2'].sum()/15))
   course_ca.append(int(df['c3'].sum()/15))

# Plot allowed students

# Scatter
plt.scatter(episodes, course_aa)
plt.scatter(episodes, course_ba)
plt.scatter(episodes, course_ca)
plt.xlabel('Episodes')
plt.ylabel('Allowed students')
plt.legend()
plt.show()

# Bar
plt.bar(np.arange(len(course_aa)), course_aa, width=width)
plt.bar(np.arange(len(course_ba))+ width, course_ba, width=width)
plt.bar(np.arange(len(course_ca))+ width + width, course_ca, width=width)
plt.xlabel('Episodes')
plt.ylabel('Allowed Students')
plt.show()

print(course_ca)
# Plot rewards
rewards = {k:int(sum(v)/len(v)) for k,v in episode_rewards.items()}
plt.bar(*zip(*rewards.items()))
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.legend()
plt.show()



