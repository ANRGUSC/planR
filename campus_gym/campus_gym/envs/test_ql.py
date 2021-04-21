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
print("Here is the current shape of the q table", q_table.shape)
# print(type(q_table))
# print(q_table.size)
# print(np.size(q_table[2]))

# Hyperparameters
alpha = 0.1
gamma = 0.2
epsilon = 0.18

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
episodes = 1
for i in tqdm(range(0,episodes)):
    state = env.reset()
    epochs, penalties, reward, = 0, 0, 0
    done = False
    # Because we don't know how many timesteps will be there we will store values until done.
    # The len of the list should be equal to the number of weeks.
    e_allowed_students = []
    e_infected_students = []
    e_return = []
    actions_taken_until_done = []
    e_states = []
    while not done:
        episode_reward = 0
        allowed_students = 0
        infected_students = 0
        if random.uniform(0, 1) < epsilon:
            action = list_to_int(env.action_space.sample().tolist(), 3)  # Explore action space
        else:
            dstate = action_conv_disc(state)
            action = np.argmax(q_table[list_to_int(dstate, 3)])  # Exploit learned values

        list_action = int_to_list(action, 3, 3)

        #        print(int_to_list(action,3, 5))

        # This is where the agent interacts with the environment.
        # Could be implemented
        next_state, reward, done, info = env.step([i * 50 for i in list_action])
        # I think the above line could be changed to use disc_conv_action
        print("List Action",list_action)
        print("Reward", reward)
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
        e_states.append(state)
        actions_taken_until_done.append(list_action)

        if reward == -10:
            penalties += 1

        state = next_state
        epochs += 1
    episode_rewards[i] = e_return
    episode_allowed_students[i] = e_allowed_students
    episode_infected_students[i] = e_infected_students
    episode_actions[i] = actions_taken_until_done
    episode_states[i] = e_states
    #episode_qtable[i] = q_table

    if i % 10 == 0:
        np.save(f"qtables/{i}-qtable.npy", q_table )



    # For analysis.
    # columns = ['c1', 'c2', 'c3']
    # infected_students_df_dict = {k: pd.DataFrame(v, columns=columns) for k, v in episode_infected_students.items()}
    # allowed_students_df_dict = {k: pd.DataFrame(v, columns=columns) for k, v in episode_allowed_students.items()}


print("Infected Students",infected_students_df_dict)
# infected students per episode
course_ai  = []
course_bi = []
course_ci = []
for episode, df in infected_students_df_dict.items():
   course_ai.append(int(df['c1'].sum()/15))
   course_bi.append(int(df['c2'].sum()/15))
   course_ci.append(int(df['c3'].sum())/15)

print("Course A", course_ai)
# Plot infected students
# scatter

# plt.scatter(episodes, course_ai)
# plt.scatter(episodes, course_bi)
# plt.scatter(episodes, course_ci)
# plt.xlabel('Episodes')
# plt.ylabel('Infected students')
# plt.legend()
# plt.show()
# # bar
# width =0.3
# plt.bar(np.arange(len(course_ai)), course_ai, width=width)
# plt.bar(np.arange(len(course_bi))+ width, course_bi, width=width)
# plt.bar(np.arange(len(course_ci))+ width + width, course_ci, width=width)
# plt.xlabel('Episodes')
# plt.ylabel('Infected Students')
# plt.show()

### Allowed Students

course_aa  = {}
course_ba = {}
course_ca = {}
for episode, df in allowed_students_df_dict.items():
    course_aa[episode] = df['c1'].tolist()
    course_ba[episode] = df['c2'].tolist()
    course_ca[episode] = df['c3'].tolist()

avg_course_aa = []
avg_course_ba = []
avg_course_ca = []

for a_episode, a_weekly_allowed_students in course_aa.items():
    avg_course_aa.append(int(sum(a_weekly_allowed_students) / len(a_weekly_allowed_students)))
for b_episode, b_weekly_allowed_students in course_ba.items():
    avg_course_ba.append(int(sum(b_weekly_allowed_students) / len(b_weekly_allowed_students)))
for c_episode, c_weekly_allowed_students in course_ca.items():
    avg_course_ca.append(int(sum(c_weekly_allowed_students) / len(c_weekly_allowed_students)))

# plt.scatter(list(episode_allowed_students.keys()), avg_course_aa, c='b', label='c1')
# plt.scatter(list(episode_allowed_students.keys()), avg_course_ba, c='r', label='c2')
# plt.scatter(list(episode_allowed_students.keys()), avg_course_ca, c='g', label='c3')
# plt.xlabel("Episodes")
# plt.ylabel("Allowed students")
# plt.legend()
# allowed_file_name = 'allowed_students' + str(epsilon)
# plt.savefig(str(episodes)+ allowed_file_name+'.png')
# plt.show()

course_ai  = {}
course_bi = {}
course_ci = {}
for episode, df in infected_students_df_dict.items():
    course_ai[episode] = df['c1'].tolist()
    course_bi[episode] = df['c2'].tolist()
    course_ci[episode] = df['c3'].tolist()

avg_course_ai = []
avg_course_bi = []
avg_course_ci = []

for a_episode, a_weekly_infected_students in course_ai.items():
    avg_course_ai.append(int(sum(a_weekly_infected_students) / len(a_weekly_infected_students)))
for b_episode, b_weekly_infected_students in course_bi.items():
    avg_course_bi.append(int(sum(b_weekly_infected_students) / len(b_weekly_infected_students)))
for c_episode, c_weekly_infected_students in course_ci.items():
    avg_course_ci.append(int(sum(c_weekly_infected_students) / len(c_weekly_infected_students)))

# plt.scatter(list(episode_infected_students.keys()), avg_course_ai, c='c', label='c1')
# plt.scatter(list(episode_infected_students.keys()), avg_course_bi, c='y', label='c2')
# plt.scatter(list(episode_infected_students.keys()), avg_course_ci, c='m', label='c3')
# plt.xlabel("Episodes")
# plt.ylabel("infected students")
# plt.legend()
# infected_file_name = 'infected_students' + str(epsilon)
# plt.savefig(str(episodes)+ infected_file_name+'.png')
# plt.show()
#
avg_rewards = []
for key, weekly_rewards in episode_rewards.items():
    avg_rewards.append(int(sum(weekly_rewards)/len(weekly_rewards)))
plt.scatter(list(episode_rewards.keys()), avg_rewards)
plt.xlabel("Episode")
plt.ylabel("Rewards")
reward_file_name = 'avg-rewards ' + str(epsilon)
plt.savefig(str(episodes)+ reward_file_name+'.png')
plt.show()


for key, weekly_allowed_students in course_aa.items():
    avg_allowed_students = int(sum(weekly_allowed_students)/len(weekly_allowed_students))
    plt.plot(weekly_allowed_students, label=key)

plt.title("CourseA Allowed students for episode" + str(episodes))
plt.xlabel("weeks")
plt.ylabel("Allowed students")
aa_allowed_file_name = 'courceA_allowed_students' + str(epsilon)
plt.savefig(str(episodes)+ aa_allowed_file_name+'.png')
plt.show()

for key, weekly_allowed_students in course_ba.items():
    plt.plot(sum(weekly_allowed_students), label=key)

plt.title("CourseB Allowed students per episode"+ str(episodes))
plt.xlabel("weeks")
plt.ylabel("Allowed students")
ba_allowed_file_name = 'courceB_allowed_students' + str(epsilon)
plt.savefig(str(episodes)+ ba_allowed_file_name+'.png')
plt.show()

for key, weekly_allowed_students in course_ca.items():
    plt.plot(weekly_allowed_students, label=key)

plt.title("CourseC Allowed students per episode"+ str(episodes))
plt.xlabel("weeks")
plt.ylabel("Allowed students")
ca_allowed_file_name = 'courceC_allowed_students' + str(epsilon)
plt.savefig(str(episodes)+ ca_allowed_file_name+'.png')
plt.show()
#
# # ### Infected Students
#
course_ai  = {}
course_bi = {}
course_ci = {}
for episode, df in infected_students_df_dict.items():
    course_ai[episode] = df['c1'].tolist()
    course_bi[episode] = df['c2'].tolist()
    course_ci[episode] = df['c3'].tolist()

for key, weekly_allowed_students in course_ai.items():
    plt.plot(weekly_allowed_students, label=key)

plt.title("CourseA infected students per episode" + str(episodes))
plt.xlabel("weeks")
plt.ylabel("Infected students")
ia_infected_file_name = 'courceA_infected_students' + str(epsilon)
plt.savefig(str(episodes)+ ia_infected_file_name+'.png')
plt.show()

for key, weekly_allowed_students in course_bi.items():
    plt.plot(weekly_allowed_students, label=key)

plt.title("CourseB Infected students per episode"+ str(episodes))
plt.xlabel("weeks")
plt.ylabel("Infected students")
ib_infected_file_name = 'courceB_infected_students' + str(epsilon)
plt.savefig(str(episodes)+ ib_infected_file_name+'.png')
plt.show()

for key, weekly_infected_students in course_ci.items():
    plt.plot(weekly_infected_students, label=key)

plt.title("CourseC Infected students per episode"+ str(episodes))
plt.xlabel("weeks")
plt.ylabel("Infected students")
ic_infected_file_name = 'courceC_infected_students' + str(epsilon)
plt.savefig(str(episodes)+ic_infected_file_name+'.png')
plt.show()

for key, weekly_rewards in episode_rewards.items():
    plt.plot(weekly_rewards, label=key)

plt.title("Rewards")
plt.xlabel("Weeks")
plt.ylabel("Rewards")
ic_infected_file_name = 'rewards ' + str(epsilon)
plt.savefig(str(episodes)+ ic_infected_file_name+'.png')
plt.show()
#
print("Episode Rewards", episode_rewards)
print("Infected Students dataframe", infected_students_df_dict)
print("Allowed students dataframe",allowed_students_df_dict)

#
average_rewards = dict(zip(episode_rewards.keys(), [int(sum(item)/len(item)) for item in episode_rewards.values()]))
lists = sorted(average_rewards.items()) # sorted by key, return a list of tuples

x, y = zip(*lists) # unpack a list of pairs into two tuples

plt.scatter(x, y)
plt.show()
