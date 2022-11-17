import os
import subprocess
import time
import gym

import campus_gym
import sys
import numpy as np
import json
import calendar
import multiprocessing
from agents.qlearning import Agent
from agents.deepqlearning import DeepQAgent
from agents.simpleagent import SimpleAgent
from agents.dqn import KerasAgent
from pathlib import Path
import wandb
import random

# agent hyper-parameters
EPISODES = 3
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
EXPLORATION_RATE = 1.0
env = gym.make('CampusGymEnv-v0')
random.seed(10)
env.seed(10)
wandb.config.update({"Episodes": EPISODES, "Learning_rate": LEARNING_RATE,
                     "Discount_factor": DISCOUNT_FACTOR, "Exploration_rate": EXPLORATION_RATE})

batch_size = 32
output_dir = "my_model"

if not os.path.exists(os.getcwd()+output_dir):
    os.makedirs(os.getcwd() + output_dir)
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



def run_training(agent_name):
    gmt = str(calendar.timegm(time.gmtime()))
    tr_name = gmt + agent_name
    # Create agent for the given environment using the agent hyper-parameters:
    # agent = DeepQAgent(env, tr_name, EPISODES, LEARNING_RATE,
    #                    DISCOUNT_FACTOR, EXPLORATION_RATE)
    # Train the agent
    # agent.train()

    # # Retrieve t0.
    # training_data = agent.training_data
    # os.chdir("../")
    # rewardspath = f'{os.getcwd()}/results/{agent_type}/rewards/{tr_name}-{EPISODES}-{format(alpha, ".1f")}rewards.json'
    # mode = 'a+' if os.path.exists(rewardspath) else 'w'
    # with open(rewardspath, mode) as rfile:
    #     json.dump(training_data[0], rfile)

    # allowedpath = f'{os.getcwd()}/results/{agent_type}/{tr_name}-{EPISODES}-{format(alpha, ".1f")}allowed.json'
    # mode_a = 'a+' if os.path.exists(rewardspath) else 'w+'
    # with open(allowedpath, mode_a) as afile:
    #      json.dump(training_data[1], afile)
    #
    # infectedpath = f'{os.getcwd()}/results/{agent_type}/{tr_name}-{EPISODES}-{format(alpha, ".1f")}infected.json'
    # mode_b = 'a+' if os.path.exists(rewardspath) else 'w+'
    # with open(infectedpath, mode_b) as ifile:
    #     json.dump(training_data[1], ifile)
    #
    # with open(f'results/E-greedy/{tr_name}-{EPISODES}-{format(alpha, ".1f")}episode_infected.json', 'w+') as ifile:
    #     json.dump(training_data[2], ifile)
    # with open(f'results/E-greedy/{tr_name}-{EPISODES}-{format(alpha, ".1f")}episode_actions.json', 'w+') as actfile:
    #     json.dump(training_data[3], actfile)




if __name__ == '__main__':
    generate_data()
    agent_name = str(sys.argv[1])
    agent = KerasAgent(env, agent_name, EPISODES, LEARNING_RATE,
                       DISCOUNT_FACTOR, EXPLORATION_RATE)

    # agent.train()
    # agent.test_all_states()


    state_size = np.prod(env.observation_space.nvec)

    for e in range(EPISODES):
        state = env.reset()
        print("State before np", state)

        state = np.reshape(state, [1, 2])
        print("State after np", state)

        done = False
        time = 0
        while not done:
            # env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1,2])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES - 1, time, agent.epsilon))
            time += 1
        if len(agent.memory) > batch_size:
            agent.train(batch_size)
        if e % 2 == 0:
            print("Saving file")
            name = os.getcwd() + "/" + output_dir + "weights_" + "{:04d}".format(e) + ".hdf5"
            print("File path", name)
            agent.save(name)


    # # multiprocessing pool object
    # #pool = multiprocessing.Pool()
    #
    # # pool object with number of element
    # pool = multiprocessing.Pool(processes=4)
    #
    # # input list
    # alpha_list = [round(float(i), 1) for i in np.arange(0, 1, 0.1)]
    #
    # # map the function to the list and pass
    # # function and input list as arguments
    # pool.map(run_training, alpha_list)
