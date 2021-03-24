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
gamma = 0.6
epsilon = 0.5

# For plotting metrics
all_epochs = []
all_penalties = []
episode_rewards = {}
exploration_episodes = 0
exloitation_episodes = 0
for i in range(1, 100):
    #   print("episode: " + str(i))
    state = env.reset()

    epochs, penalties, reward, = 0, 0, 0
    done = False
    episode_return = 0
    while not done:
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
        episode_return += reward
        old_value = q_table[list_to_int(action_conv_disc(state), 3), action]
        next_max = np.max(q_table[list_to_int(action_conv_disc(next_state), 3)])

        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[list_to_int(action_conv_disc(state), 3), action] = new_value

        if reward == -10:
            penalties += 1

        state = next_state
        epochs += 1
    episode_rewards[i] = episode_return

    # if i % 10 == 0:
    #     #  clear_output(wait=True)
    #     print(f"Episode: {i}")

#print(episode_rewards)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
rewards = list(episode_rewards.values())
episodes = list(episode_rewards.keys())
ax.bar(episodes, rewards)
plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.show()
print(exploration_episodes, exloitation_episodes)


