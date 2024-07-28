import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import product
from tqdm import tqdm
import psutil
import os
from campus_digital_twin import campus_model
import time
import matplotlib.patches as mpatches
# Set random seed for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

def estimate_infected_students(current_infected, allowed_per_course, community_risk):
    const_1 = 0.005
    const_2 = 0.01

    infected = ((const_1 * current_infected) * allowed_per_course +
                (const_2 * community_risk) * allowed_per_course ** 2)

    infected = torch.min(infected, allowed_per_course)
    return infected


def get_reward(allowed, new_infected, alpha: float):
    reward = (alpha * allowed) - ((1 - alpha) * new_infected)
    return reward


allowed = torch.tensor([0, 50, 100])  # Possible allowed values


def get_policy_regions(current_infected, community_risk, m1, c1, m2, c2):
    regions = torch.zeros_like(current_infected, dtype=torch.long)

    for i in range(current_infected.shape[0]):
        infected = current_infected[i].item()
        risk = community_risk[i].item()

        if infected < (m1 * risk + c1):
            regions[i] = 1  # Region below Line A
        elif infected < (m2 * risk + c2):
            regions[i] = 2  # Region between Line A and Line B
        else:
            regions[i] = 3  # Region above Line B

    return regions


def calculate_total_reward(current_infected, community_risk, regions, actions, alpha):
    total_reward = 0
    for i in range(current_infected.shape[0]):
        region = regions[i].item()
        action = actions[region - 1]
        new_infected = estimate_infected_students(current_infected[i], action, community_risk[i])
        reward = get_reward(action, new_infected, alpha)
        total_reward += reward.item()
    return total_reward


def evaluate_policy(current_infected, community_risk, m1, c1, m2, c2, alpha):
    regions = get_policy_regions(current_infected, community_risk, m1, c1, m2, c2)
    best_reward = -float('inf')
    best_actions = None
    for actions in product(allowed, repeat=3):  # All combinations of actions for the 3 regions
        total_reward = calculate_total_reward(current_infected, community_risk, regions, actions, alpha)
        if total_reward > best_reward:
            best_reward = total_reward
            best_actions = actions
    return best_reward, best_actions


def simulated_annealing(current_infected, community_risk, alpha, num_iterations=100, initial_temp=100, cooling_rate=0.95):
    start_time = time.time()
    process = psutil.Process()
    initial_memory = process.memory_info().rss

    m1, c1, m2, c2 = np.random.uniform(-10.0, 10.0), np.random.uniform(-100.0, 100.0), np.random.uniform(-10.0, 10.0), np.random.uniform(-100.0, 100.0)
    current_reward, current_actions = evaluate_policy(current_infected, community_risk, m1, c1, m2, c2, alpha)
    best_params = (m1, c1, m2, c2)
    best_reward = current_reward
    best_actions = current_actions

    metrics = []
    steps_taken = 0

    for iteration in tqdm(range(num_iterations), desc="Simulated Annealing Progress"):
        temp = initial_temp * (cooling_rate ** iteration)
        new_params = (
            m1 + np.random.uniform(-1.0, 1.0),
            c1 + np.random.uniform(-10.0, 10.0),
            m2 + np.random.uniform(-1.0, 1.0),
            c2 + np.random.uniform(-10.0, 10.0)
        )

        new_reward, new_actions = evaluate_policy(current_infected, community_risk, *new_params, alpha)
        steps_taken += 1

        if new_reward > current_reward or np.exp((new_reward - current_reward) / temp) > np.random.rand():
            current_reward = new_reward
            m1, c1, m2, c2 = new_params
            current_actions = new_actions
            if current_reward > best_reward:
                best_reward = current_reward
                best_params = (m1, c1, m2, c2)
                best_actions = current_actions

        metrics.append({"iteration": iteration, "reward": best_reward, "params": best_params, "steps": steps_taken})

    end_time = time.time()
    final_memory = process.memory_info().rss

    execution_time = end_time - start_time
    memory_used = final_memory - initial_memory

    metrics.append({"iteration": 'Total', "reward": best_reward, "steps": steps_taken, "execution_time": execution_time, "memory_used": memory_used})

    print("Simulated Annealing Metrics:", metrics)  # Debugging statement
    return best_params, best_actions, best_reward, metrics


def greedy_search(current_infected, community_risk, alpha, num_iterations=100, step_size=0.1):
    start_time = time.time()
    process = psutil.Process()
    initial_memory = process.memory_info().rss

    m1, c1, m2, c2 = np.random.uniform(-10.0, 10.0), np.random.uniform(-100.0, 100.0), np.random.uniform(-10.0, 10.0), np.random.uniform(-100.0, 100.0)
    best_reward, best_actions = evaluate_policy(current_infected, community_risk, m1, c1, m2, c2, alpha)
    best_params = (m1, c1, m2, c2)

    metrics = []
    steps_taken = 0
    for iteration in tqdm(range(num_iterations), desc="Greedy Search Progress"):
        candidates = [
            (m1 + step_size, c1, m2, c2),
            (m1 - step_size, c1, m2, c2),
            (m1, c1 + step_size, m2, c2),
            (m1, c1 - step_size, m2, c2),
            (m1, c1, m2 + step_size, c2),
            (m1, c1, m2 - step_size, c2),
            (m1, c1, m2, c2 + step_size),
            (m1, c1, m2, c2 - step_size)
        ]

        found_better = False
        for params in candidates:
            reward, actions = evaluate_policy(current_infected, community_risk, *params, alpha)
            steps_taken += 1
            if reward > best_reward:
                best_reward = reward
                best_params = params
                best_actions = actions
                found_better = True

        metrics.append({"iteration": iteration, "reward": best_reward, "params": best_params, "steps": steps_taken})

        if found_better:
            m1, c1, m2, c2 = best_params
        else:
            # Continue exploring by making larger perturbations if no better parameters found
            step_size *= 2  # Increase step size for wider exploration
            m1 += np.random.uniform(-1.0, 1.0)
            c1 += np.random.uniform(-10.0, 10.0)
            m2 += np.random.uniform(-1.0, 1.0)
            c2 += np.random.uniform(-10.0, 10.0)

    end_time = time.time()
    final_memory = process.memory_info().rss

    execution_time = end_time - start_time
    memory_used = final_memory - initial_memory

    metrics.append({"iteration": 'Total', "reward": best_reward, "steps": steps_taken, "execution_time": execution_time, "memory_used": memory_used})

    print("Greedy Search Metrics:", metrics)  # Debugging statement
    return best_params, best_actions, best_reward, metrics


def gradient_search(current_infected, community_risk, alpha, num_iterations=100, learning_rate=0.01):
    start_time = time.time()
    process = psutil.Process()
    initial_memory = process.memory_info().rss

    m1, c1, m2, c2 = np.random.uniform(-10.0, 10.0), np.random.uniform(-100.0, 100.0), np.random.uniform(-10.0, 10.0), np.random.uniform(-100.0, 100.0)
    current_reward, current_actions = evaluate_policy(current_infected, community_risk, m1, c1, m2, c2, alpha)
    best_params = (m1, c1, m2, c2)
    best_reward = current_reward
    best_actions = current_actions

    metrics = []
    steps_taken = 0

    for iteration in tqdm(range(num_iterations), desc="Gradient Search Progress"):
        gradients = np.random.uniform(-1.0, 1.0, 4)  # Random gradient directions
        new_params = (
            m1 + learning_rate * gradients[0],
            c1 + learning_rate * gradients[1],
            m2 + learning_rate * gradients[2],
            c2 + learning_rate * gradients[3]
        )

        new_reward, new_actions = evaluate_policy(current_infected, community_risk, *new_params, alpha)
        steps_taken += 1

        if new_reward > current_reward:
            current_reward = new_reward
            m1, c1, m2, c2 = new_params
            current_actions = new_actions
            if current_reward > best_reward:
                best_reward = current_reward
                best_params = (m1, c1, m2, c2)
                best_actions = current_actions

        metrics.append({"iteration": iteration, "reward": best_reward, "params": best_params, "steps": steps_taken})

    end_time = time.time()
    final_memory = process.memory_info().rss

    execution_time = end_time - start_time
    memory_used = final_memory - initial_memory

    metrics.append({"iteration": 'Total', "reward": best_reward, "steps": steps_taken, "execution_time": execution_time, "memory_used": memory_used})

    print("Gradient Search Metrics:", metrics)  # Debugging statement
    return best_params, best_actions, best_reward, metrics


def save_to_csv(current_infected, community_risk, label, filename):
    data_tuples = [
        f'({int(current_infected[i].item() // 10)}, {int(community_risk[i].item() * 10)})'
        for i in range(current_infected.shape[0])
    ]

    sorted_indices = sorted(range(len(data_tuples)), key=lambda i: data_tuples[i])
    sorted_data_tuples = [data_tuples[i] for i in range(current_infected.shape[0])]

    df_data = {
        'Infected and Risk': sorted_data_tuples,
        'Label': [label[sorted_indices[i]].item() for i in range(current_infected.shape[0])]
    }

    df = pd.DataFrame(df_data)
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")


def myopic_policy(campus_model, search_algorithm, alpha, num_iterations):
    DIM = 10  # Ensuring a 10x10 grid to get exactly 100 points
    model_name = "approxSI"
    y, x = torch.tensor(np.mgrid[0:DIM, 0:DIM].reshape(2, -1)).float() / (DIM - 1)

    students_per_course = campus_model.number_of_students_per_course()[0]
    total_students = torch.tensor(students_per_course)

    current_infected = (1 - y) * students_per_course
    community_risk = x  # 0 to 1

    if search_algorithm == 'greedy':
        best_params, best_actions, best_reward, metrics = greedy_search(current_infected, community_risk, alpha, num_iterations)
    elif search_algorithm == 'simulated_annealing':
        best_params, best_actions, best_reward, metrics = simulated_annealing(current_infected, community_risk, alpha, num_iterations)
    elif search_algorithm == 'gradient_search':
        best_params, best_actions, best_reward, metrics = gradient_search(current_infected, community_risk, alpha, num_iterations)
    else:
        raise ValueError("Unknown search algorithm. Choose 'greedy', 'simulated_annealing', or 'gradient_search'.")

    m1, c1, m2, c2 = best_params
    best_policy = get_policy_regions(current_infected, community_risk, m1, c1, m2, c2)

    save_to_csv(current_infected, community_risk, best_policy, f"policy_data_single_course_{search_algorithm}.csv")

    # Save metrics to CSV
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(f"search_metrics_{search_algorithm}.csv", index=False)
    print(f"Search metrics saved to search_metrics_{search_algorithm}.csv")

    return best_params, best_actions, best_policy, total_students, community_risk, current_infected


def run_experiment(campus_model, search_algorithm, alpha, num_iterations=100):
    best_params, best_actions, best_policy, total_students, community_risk, current_infected = myopic_policy(campus_model, search_algorithm, alpha, num_iterations)
    metrics_file = f"search_metrics_{search_algorithm}.csv"
    metrics_df = pd.read_csv(metrics_file)
    return best_params, best_actions, best_policy, total_students, community_risk, current_infected, metrics_df


def plot_metrics(metrics_file1, metrics_file2, metrics_file3=None):
    metrics1 = pd.read_csv(metrics_file1)
    metrics2 = pd.read_csv(metrics_file2)
    metrics3 = pd.read_csv(metrics_file3) if metrics_file3 else None

    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    # Plot Total Rewards over Iterations
    axs[0].plot(metrics1['iteration'], metrics1['reward'], label=metrics_file1.split('_')[2].split('.')[0].capitalize())
    axs[0].plot(metrics2['iteration'], metrics2['reward'], label=metrics_file2.split('_')[2].split('.')[0].capitalize())
    if metrics3 is not None:
        axs[0].plot(metrics3['iteration'], metrics3['reward'], label=metrics_file3.split('_')[2].split('.')[0].capitalize())
    axs[0].set_title('Total Reward over Iterations')
    axs[0].set_xlabel('Iterations')
    axs[0].set_ylabel('Total Reward')
    axs[0].legend()
    axs[0].xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Plot Number of Steps
    axs[1].plot(metrics1['iteration'], metrics1['steps'], label=metrics_file1.split('_')[2].split('.')[0].capitalize())
    axs[1].plot(metrics2['iteration'], metrics2['steps'], label=metrics_file2.split('_')[2].split('.')[0].capitalize())
    if metrics3 is not None:
        axs[1].plot(metrics3['iteration'], metrics3['steps'], label=metrics_file3.split('_')[2].split('.')[0].capitalize())
    axs[1].set_title('Number of Steps over Iterations')
    axs[1].set_xlabel('Iterations')
    axs[1].set_ylabel('Steps')
    axs[1].legend()
    axs[1].xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Plot Number of Iterations until Convergence
    metrics1_convergence = metrics1['iteration'].max()
    metrics2_convergence = metrics2['iteration'].max()
    if metrics3 is not None:
        metrics3_convergence = metrics3['iteration'].max()
        axs[2].bar(['Algorithm 1', 'Algorithm 2', 'Algorithm 3'], [metrics1_convergence, metrics2_convergence, metrics3_convergence])
    else:
        axs[2].bar(['Algorithm 1', 'Algorithm 2'], [metrics1_convergence, metrics2_convergence])
    axs[2].set_title('Iterations until Convergence')
    axs[2].set_xlabel('Algorithm')
    axs[2].set_ylabel('Iterations')

    plt.tight_layout()
    plt.savefig('performance_metrics_comparison.png')
    plt.show()

    # Create a separate plot for the table of metrics
    fig, ax = plt.subplots(figsize=(12, 5))  # Adjusted size for better readability
    table_data = pd.concat([metrics1.tail(1), metrics2.tail(1)])
    if metrics3 is not None:
        table_data = pd.concat([table_data, metrics3.tail(1)])

    # Drop the 'params' column
    table_data = table_data.drop(columns=['params', 'iteration', 'memory_used'])

    # Formatting large numbers in scientific notation and rounding decimals to 2 places
    for col in ['reward', 'steps']:
        if table_data[col].dtype == float or table_data[col].dtype == int:
            table_data[col] = table_data[col].apply(lambda x: f'{x:.2e}' if abs(x) > 1000 else f'{x:.2f}')

    # Round other columns to two decimal places
    for col in table_data.columns:
        if col not in ['reward', 'steps'] and (
                table_data[col].dtype == float or table_data[col].dtype == int):
            table_data[col] = table_data[col].round(2)

    # Add algorithm labels for clarity
    table_data.insert(0, 'Algorithm', [metrics_file1.split('_')[2].split('.')[0].capitalize(),
                                       metrics_file2.split('_')[2].split('.')[0].capitalize()] +
                                      ([metrics_file3.split('_')[2].split('.')[0].capitalize()] if metrics3 is not None else []))

    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=table_data.values, colLabels=table_data.columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    plt.savefig('metrics_table.png')
    plt.show()

def plot_policy_lines_and_markers(best_params, best_actions, best_policy, total_students, community_risk, current_infected, search_algorithm, model_name="approxSI"):
    m1, c1, m2, c2 = best_params

    # Plot only the lines
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.suptitle(f'Policy Lines Only - {search_algorithm}', fontsize=16)

    x_vals = np.linspace(0, 1, 100)
    y_vals_m1 = m1 * x_vals + c1
    y_vals_m2 = m2 * x_vals + c2

    ax.plot(x_vals, y_vals_m1, 'g-', linewidth=2, label=f'Line A: y = {m1:.2f}x + {c1:.2f}')
    ax.plot(x_vals, y_vals_m2, 'b-', linewidth=2, label=f'Line B: y = {m2:.2f}x + {c2:.2f}')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, total_students + 0.05 * total_students)
    ax.set_xlabel('Community Risk')
    ax.set_ylabel('Current Infected')
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(f"policy_lines_only_{search_algorithm}.png", bbox_inches='tight', dpi=300)
    plt.show()

    # Plot the lines with markers
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.suptitle(f'Policy Label-{model_name}-{search_algorithm}', fontsize=16)

    colors = {0: '#a9a9a9', 50: 'darkorange', 100: 'turquoise'}  # Grey, Light Blue, Light Green
    region_to_action = {1: int(best_actions[0]), 2: int(best_actions[1]), 3: int(best_actions[2])}

    # Debugging: Print the region-to-action mapping
    print(f"Region to Action Mapping: {region_to_action}")

    def safe_color_mapping(region):
        region_val = region.item()
        if region_val in region_to_action:
            action = region_to_action[region_val]
            return colors.get(action, '#FFFFFF')  # Default to white if action not in colors
        return '#FFFFFF'

    scatter = ax.scatter(community_risk.numpy(), current_infected.numpy(),
                         c=[safe_color_mapping(region) for region in best_policy], s=100, marker='s')
    ax.set_xlabel('Community Risk')
    ax.set_ylabel('Current Infected')
    ax.set_title(f'Total Students: {total_students}')
    ax.grid(False)
    max_val = total_students
    # Adjust y-axis limits to show full markers
    y_margin = max_val * 0.05  # 5% margin
    ax.set_ylim(-y_margin, max_val + y_margin)

    # Adjust x-axis limits to show full markers
    ax.set_xlim(-0.05, 1.05)

    # Plot the lines used for policy determination
    ax.plot(x_vals, y_vals_m1, 'g-', linewidth=2, label=f'Line A: y = {m1:.2f}x + {c1:.2f}')
    ax.plot(x_vals, y_vals_m2, 'b-', linewidth=2, label=f'Line B: y = {m2:.2f}x + {c2:.2f}')
    ax.legend(loc='upper right')

    # Create a custom legend
    unique_actions = sorted(set(region_to_action.values()))
    legend_elements = [mpatches.Patch(facecolor=colors[val], edgecolor='black', label=f'Allow {val}')
                       for val in unique_actions]
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.05),
               ncol=len(unique_actions), fontsize='large')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(f"label_{model_name}_{search_algorithm}.png", bbox_inches='tight', dpi=300)
    plt.show()


if __name__ == "__main__":
    # Uncomment and run the experiment
    campus_model = campus_model.CampusModel(num_courses=1, students_per_course=[100], initial_infection_rate=[0.2])
    alpha = 0.5
    num_iterations = 1000

    search_algorithm = 'greedy'  # Change to 'simulated_annealing' or 'greedy' or gradient_search to run other algorithms
    best_params, best_actions, best_policy, total_students, community_risk, current_infected, metrics = run_experiment(campus_model, search_algorithm, alpha, num_iterations)

    plot_policy_lines_and_markers(best_params, best_actions, best_policy, total_students, community_risk, current_infected, search_algorithm)

    # Optional: Call the plot_metrics function with the two CSV files
    # plot_metrics('search_metrics_greedy.csv', 'search_metrics_simulated_annealing.csv', 'search_metrics_gradient_search.csv')
