import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats
from joblib import Parallel, delayed

alpha_list = [round(float(i), 1) for i in np.arange(0, 1, 0.1)]
training_name = "run.py"
episodes = 5000


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


def evaluate_training(run_name, no_of_episodes, alpha):
    training_rewards_df = pd.read_json(f'results/E-greedy/{run_name}-{no_of_episodes}-{alpha}episode_rewards.json')
    training_rewards_df.to_csv(f'results/E-greedy/{run_name}-{no_of_episodes}-{alpha}episode_rewards.csv')
    print(training_rewards_df)


# plot_expected_rewards()
# plot_allowed_vs_infected()
evaluate_training('run.py', 5000, 0.9)
