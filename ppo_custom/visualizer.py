import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import numpy as np
import wandb
import scipy.stats as stats
import pandas as pd
from tabulate import tabulate
import torch
import torch.nn as nn
import torch.optim as optim

# policy matrix
def visualize_all_states(model, all_states, run_name, max_episodes, alpha, results_subdirectory):
    method_name = "viz all states"
    actions = {}
    for i in all_states:
        policy_dist, _ = model(torch.FloatTensor(i).unsqueeze(0))
        action = torch.argmax(policy_dist).item()
        actions[(i[0], i[1])] = [action]

    x_values = []
    y_values = []
    colors = []
    # print(actions)
    for k, v in actions.items():
        x_values.append(k[0])
        y_values.append(k[1])
        if v[0] == 0:
            colors.append('red')
        elif v[0] == 1:
            colors.append('green')
        else:
            colors.append('blue')

    plt.scatter(y_values, x_values, c=colors)
    plt.title(f"{method_name} - {run_name}")
    plt.xlabel("Community risk")
    plt.ylabel("Infected students")

    # Create a legend with explicit labels
    legend_labels = ['Allow no one (0)', '50% allowed (1)', 'Allow everyone (2)']
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in ['red', 'green', 'blue']]
    plt.legend(legend_handles, legend_labels, loc='upper left', bbox_to_anchor=(1.04, 1))

    file_name = f"{max_episodes}-{method_name}-{run_name}-{alpha}.png"
    file_path = f"{results_subdirectory}/{file_name}"
    plt.savefig(file_path, bbox_inches='tight')  # Use bbox_inches='tight' to include the legend in the saved image
    plt.close()
    return file_path

    # Log the image to wandb


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


def visualize_variance_in_rewards(rewards, results_subdirectory, episode):
    method_name = "viz insights"

    bin_size = 500  # number of episodes per bin, adjust as needed
    num_bins = len(rewards) // bin_size

    # Prepare data
    bins = []
    binned_rewards = []
    for i in range(num_bins):
        start = i * bin_size
        end = start + bin_size
        bin_rewards = rewards[start:end]
        bins.extend([f"{start}-{end}"] * bin_size)
        binned_rewards.extend(bin_rewards)

    data = pd.DataFrame({"Bin": bins, "Reward": binned_rewards})

    # Plot Variance (Box Plot)
    plt.figure(figsize=(12, 6))  # Adjust figure size as needed
    sns.boxplot(x='Bin', y='Reward', data=data)
    plt.title(f'Variance in Rewards - {method_name}')
    plt.xlabel('Episode Bin')
    plt.ylabel('Reward')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.tight_layout()  # Ensure everything fits without overlapping

    file_path_boxplot = f"{results_subdirectory}/variance_in_rewards-{method_name}-{episode}.png"
    plt.savefig(file_path_boxplot)
    plt.close()
    return file_path_boxplot

    # Log the boxplot image to wandb



def visualize_variance_in_rewards_heatmap(rewards_per_episode, results_subdirectory, bin_size):
    num_bins = len(rewards_per_episode) // bin_size
    binned_rewards_var = [np.var(rewards_per_episode[i * bin_size: (i + 1) * bin_size]) for i in
                          range(len(rewards_per_episode) // bin_size)]
    # print("num bins", num_bins, "rewars per episode", len(rewards_per_episode), "binned rewards var", len(binned_rewards_var))


    # Reshape to a square since we're assuming num_bins is a perfect square
    side_length = int(np.sqrt(num_bins))
    reshaped_var = np.array(binned_rewards_var).reshape(side_length, side_length)

    plt.figure(figsize=(10, 6))
    sns.heatmap(reshaped_var, cmap='YlGnBu', annot=True, fmt=".2f")
    plt.title('Variance in Rewards per Bin')
    plt.xlabel('Bin Index')
    plt.ylabel('Bin Index')
    file_path_heatmap = f"{results_subdirectory}/variance_in_rewards_heatmap.png"
    plt.savefig(file_path_heatmap)
    plt.close()
    return file_path_heatmap




def visualize_explained_variance(actual_rewards, predicted_rewards, results_subdirectory, max_episodes):
    # Calculate explained variance for each episode\
    # Ensure predicted_rewards has compatible shape
    predicted_rewards = np.squeeze(predicted_rewards)
    explained_variances = []
    for episode in range(1, max_episodes + 1):
        actual = actual_rewards[:episode]
        predicted = predicted_rewards[:episode]
        residuals = np.array(actual) - np.array(predicted)
        if np.var(actual) == 0:  # Prevent division by zero
            explained_variance = np.nan
        else:
            explained_variance = 1 - np.var(residuals) / np.var(actual)
        explained_variances.append(explained_variance)

    # Visualize explained variance
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_episodes + 1), explained_variances)
    plt.title('Explained Variance over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Explained Variance')
    file_path = f"{results_subdirectory}/explained_variance.png"
    plt.savefig(file_path)
    plt.close()
    return file_path

def visualize_infected_vs_community_risk(inf_comm_dict, alpha, results_subdirectory):
    community_risk = inf_comm_dict['community_risk']
    infected = inf_comm_dict['infected']
    allowed = inf_comm_dict['allowed']
    # Create a new figure
    plt.figure(figsize=(10, 6))

    # set the y-axis limit
    plt.ylim(0, 120)

    # Scatter plots
    plt.scatter(community_risk, infected, color='blue', label="Infected", alpha=alpha, s=60)
    plt.scatter(community_risk, allowed, color='red', label="Allowed", alpha=alpha, s=60)

    # Set the title and labels
    plt.title('Infected vs Community Risk with alpha = ' + str(alpha))
    plt.xlabel('Community Risk')
    plt.ylabel('Number of Students')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    # Adjust layout to accommodate the legend
    plt.tight_layout()
    file_path = f"{results_subdirectory}/infected_vs_community_risk.png"
    plt.savefig(file_path)
    plt.close()
    return file_path


def visualize_infected_vs_community_risk_table(inf_comm_dict, alpha, results_subdirectory):
    community_risk = inf_comm_dict['community_risk']
    infected = inf_comm_dict['infected']
    allowed = inf_comm_dict['allowed']

    # Combine the data into a list of lists
    data = list(zip(community_risk, infected, allowed))

    # Define the headers for the table
    headers = ["Community Risk", "Infected", "Allowed"]

    # Use the tabulate function to create a table
    table = tabulate(data, headers, tablefmt="pretty")

    # Define the title with alpha
    title = f'Infected vs Community Risk with alpha = {alpha}'

    # Create a Matplotlib figure and axis to render the table
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')  # Turn off axis labels

    # Render the table
    ax.table(cellText=data, colLabels=headers, loc='center')

    # Add the title to the table
    ax.set_title(title, fontsize=14)

    # Save the table as an image
    file_path = f"{results_subdirectory}/infected_vs_community_risk_table.png"
    plt.savefig(file_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    return file_path

def states_visited_viz(states, visit_counts, alpha, results_subdirectory):
    # Sort states and corresponding visit counts
    sorted_indices = sorted(range(len(states)), key=lambda i: states[i])
    sorted_states = [states[i] for i in sorted_indices]
    sorted_visit_counts = [visit_counts[i] for i in sorted_indices]

    title = f'State Visitation Frequency During Training with alpha: = {alpha}'

    # Create a bar chart
    plt.figure(figsize=(10, 6))  # Adjust figure size as needed
    plt.bar(sorted_states, sorted_visit_counts)

    # Rotate x-axis labels if there are many states for better readability
    plt.xticks(rotation=90)

    # Add labels and title
    plt.xlabel('State')
    plt.ylabel('Visitation Count')
    plt.title(title)

    plt.tight_layout()
    file_path = f"{results_subdirectory}/states_visited.png"
    plt.savefig(file_path)
    plt.close()

    return file_path



