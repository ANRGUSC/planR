import os
import string
import subprocess
import time
import gymnasium as gym
# import gym
from tqdm import tqdm

import campus_gym
import sys
import numpy as np
import json
import calendar
import multiprocessing as mp
from functools import partial
from agents.qlearning import Agent
from agents.deepqlearning import DeepQAgent
# from agents.simpleagent import SimpleAgent
# from agents.dqn import KerasAgent
from pathlib import Path
import wandb
import random
import codecs, json
import io
# from keras.models import load_model
import itertools
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from campus_gym.envs.campus_gym_env import CampusGymEnv

#wandb.init(project="planr-5", entity="leezo")
# agent hyper-parameters
EPISODES = 200
LEARNING_RATE = 0.003 # increment by half of it
DISCOUNT_FACTOR = 0.9
EXPLORATION_RATE = 0.1
# print(f'available IDs: {gym.envs.registry.keys()}')

np.random.seed(100)
env = CampusGymEnv()
# env = gym.make('CampusGymEnv-v0')

random.seed(100)
# env.seed(100)
#wandb.config.update({"Episodes": EPISODES, "Learning_rate": LEARNING_RATE,
                    #"Discount_factor": DISCOUNT_FACTOR, "Exploration_rate": EXPLORATION_RATE})

# more episodes, learning rate, change reward func?
# draw reward graph for first 1000 episodes, then first 2000 episodes
# start with 10000 episodes

batch_size = 5
if not os.path.exists(os.getcwd()):
    os.makedirs(os.getcwd())

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def subprocess_cmd(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    proc_stdout = process.communicate()[0].strip()
    print(proc_stdout)


def generate_data():
    try:

        os.chdir("campus_data_generator")

        subprocess_cmd('python3 generate_simulation_params.py')
        subprocess_cmd('python3 generate_model_csv_files.py')
        subprocess_cmd('python3 test_generate_model_csv_files.py')
        print("Dataset generated")

    except:
        print("Error generating dataset files")


def run_training(alpha):
    #tr_name = wandb.run.name
    agent_name = str('abcd')
    agent = DeepQAgent(env, EPISODES, LEARNING_RATE,
                  DISCOUNT_FACTOR, EXPLORATION_RATE)
    # agent = Agent(env, agent_name, EPISODES, LEARNING_RATE,
    #               DISCOUNT_FACTOR, EXPLORATION_RATE)
    training_data = agent.train(alpha)
    # agent.test_all_states(alpha)
    return training_data, agent_name


if __name__ == '__main__':
    generate_data()
    alpha = 0.3 #:float(sys.argv[1])
    run_data, training_name = run_training(alpha)
    file_name = str(EPISODES) + "-" + str(alpha) + "-" + training_name + "training_data" + ".json"
    with io.open(file_name, 'w', encoding='utf8') as outfile:
        training_data_ = json.dumps(run_data, indent=4, sort_keys=True, ensure_ascii=False, cls=NpEncoder)
        outfile.write(training_data_)



    # state_size = np.prod(env.observation_space.nvec)
    # episode_rewards = {}
    #
    # for e in tqdm (range(EPISODES)):
    #     state = env.reset()
    #     print("State before np", state)
    #
    #     state = np.reshape(state, [1, 2])
    #     print("State after np", state)
    #
    #     done = False
    #     time = 0
    #     e_return = []
    #     while not done:
    #         # env.render()
    #         print("State Training", state)
    #         action = agent.act(state)
    #         next_state, reward, done, _ = env.step(action)
    #         reward = reward if not done else -10
    #         next_state = np.reshape(next_state, [1,2])
    #         agent.remember(state, action, reward, next_state, done)
    #         state = next_state
    #         if done:
    #             print("episode: {}/{}, score: {}, e: {:.2}"
    #                   .format(e, EPISODES - 1, time, agent.epsilon))
    #         time += 1
    #         e_return.append(reward)
    #     if len(agent.memory) > batch_size:
    #         agent.train(batch_size)
    #     episode_rewards[e] = e_return
    #     wandb.log({'reward': sum(e_return) / len(e_return)})
    #     # if e % 50 == 0:
    #     #     name = os.getcwd() + "/" + "weights_" + "{:04d}".format(e) + ".h5"
    #     #     print("File path", name)
    #     #     agent.save(name)
    # name = os.getcwd() + "/" + "weights_" + "{:04d}".format(EPISODES) + ".h5"
    # print("File path", name)
    # agent.save(name)
    # agent.training_data = [episode_rewards]
    #
    # # Show training performance
    #
    # rewards = agent.training_data[0]
    # avg_rewards = {k: sum(v) / len(v) for k, v in rewards.items()}
    # lists = sorted(avg_rewards.items())
    # x, y = zip(*lists)
    # plt.plot(x, y)
    # plt.title(" Deep Q learning with experience replay")
    # plt.xlabel('Episodes')
    # plt.ylabel('Expected return')
    # plt.show()
    #
    # # Test
    # saved_path = name
    # print("Path", saved_path)
    # model = load_model(saved_path)
    # possible_states = [list(range(0, (k))) for k in env.observation_space.nvec]
    # all_states = list(itertools.product(*possible_states))
    #
    # actions = {}
    # for i in all_states:
    #     f_state = np.reshape(i, [1, 2])
    #     print("f_state", f_state)
    #     action = np.argmax(model.predict(f_state)[0])
    #     print("Action", action)
    #     actions[(i[0], i[1])] = action
    #
    # x_values = []
    # y_values = []
    # colors = []
    # for k, v in actions.items():
    #     x_values.append(k[0])
    #     y_values.append(k[1])
    #     colors.append(v)
    #
    # c = ListedColormap(['red', 'green', 'blue'])
    # s = plt.scatter(y_values, x_values, c=colors, cmap=c)
    # plt.xlabel("Community risk")
    # plt.ylabel("Infected students")
    # plt.legend(*s.legend_elements(), loc='upper left')
    # plt.show()

    # act_values = self.model.predict(state)
    # np.argmax(act_values[0])
