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
from pathlib import Path
import wandb

# agent hyper-parameters
EPISODES = 200
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
EXPLORATION_RATE = 0.3
env = gym.make('CampusGymEnv-v0')
wandb.config.update({"Episodes": EPISODES, "Learning_rate": LEARNING_RATE,
                     "Discount_factor": DISCOUNT_FACTOR, "Exploration_rate": EXPLORATION_RATE})


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
    agent = DeepQAgent(env, agent_name, EPISODES, LEARNING_RATE,
                       DISCOUNT_FACTOR, EXPLORATION_RATE)
    agent.train()

    print("Testing all the states")
    agent.test_all_states()
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
