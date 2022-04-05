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
import logging

logger = logging.getLogger()
fhandler = logging.FileHandler(filename='egreedy.log', mode='w+')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fhandler.setFormatter(formatter)
logger.addHandler(fhandler)
logger.setLevel(logging.INFO)
# logging.basicConfig(filename='deepq.log', filemode='w+', format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)

# agent hyper-parameters
EPISODES = 10
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
EXPLORATION_RATE = 0.2


env = gym.make('CampusGymEnv-v0')


def subprocess_cmd(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    proc_stdout = process.communicate()[0].strip()
    print(proc_stdout)


def generate_data():
    try:

        os.chdir("campus_data_generator")

        subprocess_cmd('python3 generate_simulation_params.py')
        subprocess_cmd('python3 generate_model_csv_files.py')
        print("Dataset generated")

    except:
        print("Error generating dataset files")


def run_training(alpha):
    gmt = str(calendar.timegm(time.gmtime()))
    method = "deepq"
    tr_name = gmt + method
    agent_type = "deepq"

    # Create agent for the given environment using the agent hyper-parameters:
    agent = Agent(env, tr_name, EPISODES, LEARNING_RATE,
                   DISCOUNT_FACTOR, EXPLORATION_RATE)
    # Train the agent using a chosen reward weight parameter (ALPHA)
    agent.train(alpha)

    # Retrieve t0.
    training_data = agent.training_data
    #os.chdir("../")
    rewardspath = f'{os.getcwd()}/results/{agent_type}/rewards/{tr_name}-{EPISODES}-{format(alpha, ".1f")}rewards.json'
    mode = 'a+' if os.path.exists(rewardspath) else 'w+'
    with open(rewardspath, mode) as rfile:
        json.dump(training_data[0], rfile)

    allowedpath = f'{os.getcwd()}/results/{agent_type}/{tr_name}-{EPISODES}-{format(alpha, ".1f")}allowed.json'
    mode_a = 'a+' if os.path.exists(rewardspath) else 'w+'
    with open(allowedpath, mode_a) as afile:
         json.dump(training_data[1], afile)

    infectedpath = f'{os.getcwd()}/results/{agent_type}/{tr_name}-{EPISODES}-{format(alpha, ".1f")}infected.json'
    mode_b = 'a+' if os.path.exists(rewardspath) else 'w+'
    with open(infectedpath, mode_b) as ifile:
        json.dump(training_data[1], ifile)

    # with open(f'results/E-greedy/{tr_name}-{EPISODES}-{format(alpha, ".1f")}episode_infected.json', 'w+') as ifile:
    #     json.dump(training_data[2], ifile)
    # with open(f'results/E-greedy/{tr_name}-{EPISODES}-{format(alpha, ".1f")}episode_actions.json', 'w+') as actfile:
    #     json.dump(training_data[3], actfile)

    print("Done Training. Check results/E-greedy folder for training data")


if __name__ == '__main__':
    #generate_data()
    run_training(alpha=0.9)
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

