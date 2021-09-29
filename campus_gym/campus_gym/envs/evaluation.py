import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats
import json
import os
from joblib import Parallel, delayed
import seaborn as sns
alpha_list = [round(float(i), 1) for i in np.arange(0, 1, 0.1)]
training_name = ""
episodes = 3000


def average_students_run(df):
    avg_students = []
    for episode in df:
        data = np.array(list(df[episode]))
        means = [int(i) for i in list(np.mean(data, axis=1))]
        episode_means = int(sum(means) / len(means))
        avg_students.append(episode_means)
    mean_students = int(sum(avg_students) / len(avg_students))
    return mean_students


def get_avg_alpha(alpha_p):
    alpha = round(alpha_p, 1)
    training_allowed_df = pd.read_json(f'results/E-greedy/{training_name}-{episodes}-{alpha}episode_allowed.json')
    training_infected_df = pd.read_json(f'results/E-greedy/{training_name}-{episodes}-{alpha}episode_infected.json')
    allowed_avg = average_students_run(training_allowed_df)
    infected_avg = average_students_run(training_infected_df)
    return allowed_avg, infected_avg


def plot_allowed_vs_infected():
    allowed_infected = Parallel(n_jobs=4)(delayed(get_avg_alpha)(i) for i in alpha_list)
    allowed_infected_df = pd.DataFrame(allowed_infected, columns=['allowed', 'infected'])
    print(alpha_list)
    allowed_infected_df.insert(2, "alpha", alpha_list)
    print(allowed_infected_df)

    groups = allowed_infected_df.groupby('alpha')
    for name, group in groups:
        plt.plot(group.allowed, group.infected, marker='o', linestyle='', markersize=12, label=name)
    plt.legend()
    plt.xlabel('Allowed students')
    plt.ylabel('Infected students')
    plt.savefig(f'results/{training_name}-allowed_vs_infected.png')
    plt.close()


def plot_raw_expected_rewards():
    training_rewards_df = pd.read_json(f'results/E-greedy/{training_name}-{episodes}-{0.9}episode_rewards.json')
    average_rewards = list(map(int, list(training_rewards_df.mean(axis=0))))
    x_axis = list(range(0, len(average_rewards)))
    confidence_intervals = []
    for episode in training_rewards_df:
        ci = 1.96 * np.std(training_rewards_df[episode]) / np.mean(training_rewards_df[episode])
        confidence_intervals.append(ci)

    x = np.array(x_axis[0::200])
    y = np.array(average_rewards[0::200])
    y_err = np.array(confidence_intervals[0::200])
    plt.errorbar(x, y, yerr=y_err, label='both limits (default)', capsize=5, ecolor='red', color='grey')
    plt.title('Expected Average Rewards')
    plt.xlabel('Episodes')
    plt.ylabel('Expected rewards')
    plt.savefig(f'results/{training_name}-expected-average-rewards.png')
    plt.close()


def json_to_csv(run_name, no_of_episodes, alpha):
    training_rewards_df = pd.read_json(f'results/E-greedy/{run_name}-{no_of_episodes}-{alpha}episode_rewards.json')
    training_rewards_df.to_csv(f'results/E-greedy/{run_name}-{no_of_episodes}-{alpha}episode_rewards.csv')
    print(training_rewards_df)


def evaluate_training():
    path_to_files = 'results/E-greedy/rewards'
    json_files = [pos_json for pos_json in os.listdir(path_to_files) if pos_json.endswith('.json')]
    all_runs_sum_rewards= []
    for index, js in enumerate(json_files):
        with open(os.path.join(path_to_files, js)) as json_file:
            json_text = json.load(json_file)
            rewards_list = list(json_text.values())
            cumulative_rewards = [int(sum(i)/len(i)) for i in rewards_list]
            all_runs_sum_rewards.append(cumulative_rewards)

    cumulative_rewards_df = pd.DataFrame(all_runs_sum_rewards)
    mean_of_episodes = list(cumulative_rewards_df.mean())
    x_axis = list(range(0, len(mean_of_episodes)))

    confidence_intervals = []
    for episode in cumulative_rewards_df:
        ci = 1.96 * np.std(cumulative_rewards_df[episode]) / np.sqrt(10)
        confidence_intervals.append(ci)
    x = np.array(x_axis[0::200])
    y = np.array(mean_of_episodes[0::200])
    y_err = np.array(confidence_intervals[0::200])
    # fig, ax = plt.subplots()
    # ax.plot(x, y)
    # ax.fill_between(x, (y - y_err), (y + y_err), color='b', alpha=.1)
    plt.errorbar(x, y, yerr=y_err, label='both limits (default)', capsize=3, ecolor='blue', color='grey')
    plt.title('RL Agent performance')
    plt.xlabel('Episodes')
    plt.ylabel('Expected rewards')
    plt.show()

    # plt.savefig(f'results/3000-expected-average-rewards.png')
    #plt.close()


evaluate_training()
