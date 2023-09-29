import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import numpy as np
import wandb
import scipy.stats as stats


def visualize_all_states(q_table, all_states, states, run_name, max_episodes, alpha, results_subdirectory):
    method_name = "viz all states"
    actions = {}
    for i in states:
        action = np.argmax(q_table[all_states.index(str(i))])
        actions[(i[0], i[1])] = action

    x_values = []
    y_values = []
    colors = []
    for k, v in actions.items():
        x_values.append(k[0])
        y_values.append(k[1])
        colors.append(v)

    c = ListedColormap(['red', 'green', 'blue'])
    s = plt.scatter(y_values, x_values, c=colors, cmap=c)
    plt.title(f"{method_name} - {run_name}")
    plt.xlabel("Community risk")
    plt.ylabel("Infected students")
    plt.legend(*s.legend_elements(), loc='upper left', bbox_to_anchor=(1.04, 1))
    file_name = f"{max_episodes}-{method_name}-{run_name}-{alpha}.png"
    file_path = f"{results_subdirectory}/{file_name}"
    plt.savefig(file_path)
    plt.close()

    # Log the image to wandb
    wandb.log({"All_States_Visualization": [wandb.Image(file_path)]})

def visualize_q_table(q_table, results_subdirectory, episode):
    method_name = "viz q table"
    plt.figure(figsize=(10, 10))
    sns.heatmap(q_table, cmap="YlGnBu", annot=True, fmt=".2f")
    plt.title(f'Q-Table at Episode {episode} - {method_name}')
    plt.xlabel('Actions')
    plt.ylabel('States')
    file_path = f"{results_subdirectory}/qtable-{method_name}-{episode}.png"
    plt.savefig(file_path)
    plt.close()


def visualize_insights(rewards, results_subdirectory, episode):
    method_name = "viz insights"
    # Assuming rewards is a list of lists where each inner list contains the rewards for one episode

    # Prepare data
    episodes = list(range(len(rewards)))
    data = {"Episode": episodes, "Reward": rewards}

    # Plotting the smooth curve with a confidence interval
    sns.lineplot(x="Episode", y="Reward", data=data, errorbar='sd', color='blue', label='Mean Reward')

    plt.title('Mean Rewards with Confidence Intervals')
    plt.xlabel('Episode')
    plt.ylabel('Mean Reward')
    file_path = f"{results_subdirectory}/mean_rewards_with_ci-{method_name}-{episode}.png"
    plt.savefig(file_path)
    plt.close()

    # Plot Variance (Box Plot)
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=rewards)
    plt.title(f'Variance in Rewards - {method_name}')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig(f"{results_subdirectory}/variance_in_rewards-{method_name}-{episode}.png")
    plt.close()


def visualize_explained_variance(actual_rewards, predicted_rewards, results_subdirectory, episode):
    # Calculate explained variance
    residuals = np.array(actual_rewards) - np.array(predicted_rewards)
    explained_variance = 1 - np.var(residuals) / np.var(actual_rewards)

    # Visualize explained variance
    plt.figure(figsize=(10, 6))
    plt.plot(explained_variance)
    plt.title('Explained Variance over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Explained Variance')
    file_path = f"{results_subdirectory}/explained_variance-{episode}.png"
    plt.savefig(file_path)
    plt.close()



