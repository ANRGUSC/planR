# %%time
"""Training the agent"""
import matplotlib.pyplot as plt
import random
# from IPython.display import clear_output
import gym
from gym.envs.registration import register
import sys

sys.path.append('../../..')
sys.path.append('../../../campus_digital_twin')
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
# observation = env.reset()

q_table = np.zeros([np.prod(env.observation_space.nvec), np.prod(env.action_space.nvec)])
print(type(q_table))
print(q_table.size)
print(np.size(q_table[2]))

# Hyperparameters
alpha = 0.1
gamma = 0.2
epsilon = 0.2

# For plotting metrics
all_epochs = []
all_penalties = []
episode_rewards = {}
episode_allowed_students = {}
episode_infected_students = {}
exploration_episodes = 0
exloitation_episodes = 0
for i in range(1, 100):
    if i%10 == 0:
      print("episode: " + str(i))
    state = env.reset()

    epochs, penalties, reward, = 0, 0, 0
    done = False

    # Because we don't know how many timesteps will be there we will store values until done.
    # The len of the list should be equal to the number of weeks.
    e_allowed_students = []
    e_infected_students = []
    e_return = []

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
        #            print(action)

        #print(action)
        list_action = int_to_list(action, 3, 5)
        #        print(int_to_list(action,3, 5))
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

        if reward == -10:
            penalties += 1

        state = next_state
        epochs += 1
    episode_rewards[i] = e_return
    episode_allowed_students[i] = e_allowed_students
    episode_infected_students[i] = e_infected_students

    # if i % 10 == 0:
    #     #  clear_output(wait=True)
    #     print(f"Episode: {i}")

#print(episode_rewards)
rewards = {k:sum(v) for k,v in episode_rewards.items()}
allowed_students = {k:sum(v) for k,v in episode_allowed_students.items()}
infected_students = {k:sum(v) for k,v in episode_infected_students.items()}

episodes = list(rewards.keys())
# Rewards
rewards_points = list(rewards.values())
# plotting the reward points
#plt.scatter(episodes, rewards_points, label = "Rewards")

# Allowed students
allowed_students_points = list(allowed_students.values())
# plotting the allowed student points
#plt.scatter(episodes, allowed_students_points, label = "Allowed Students")

# Infected students
infected_students_points = list(infected_students.values())
# plotting the infected student points
plt.scatter(allowed_students_points, infected_students_points)

plt.xlabel('Allowed Students')
# Set the y axis label of the current axis.
plt.ylabel('Infected Students')
# Set a title of the current axes.
title = 'Allowed Students vs Infected Students (epsilon = '
plt.title(title + str(epsilon) + ')')
# show a legend on the plot
plt.legend()
# Display a figure.
plt.show()

print(exploration_episodes, exloitation_episodes)
print(allowed_students_points)
print(rewards_points)
print(infected_students_points)

print(episode_rewards, episode_allowed_students, episode_infected_students)



