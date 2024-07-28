import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from scipy.stats import binom
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from campus_digital_twin import campus_model


def estimate_infected_students(current_infected, allowed_per_course, community_risk):
    const_1 = 0.005
    const_2 = 0.01

    # Expand dimensions to match
    current_infected = current_infected.unsqueeze(2)  # Shape: [100, num_courses, 1]
    community_risk = community_risk.unsqueeze(1).unsqueeze(2)  # Shape: [100, 1, 1]

    infected = ((const_1 * current_infected) * allowed_per_course +
                (const_2 * community_risk) * allowed_per_course ** 2)

    infected = torch.min(infected, allowed_per_course)
    return infected

def get_reward(allowed, new_infected, alpha: float):
    reward = (alpha * allowed) - ((1 - alpha) * new_infected)
    return reward

allowed = torch.tensor([0, 50, 100])  # Updated allowed values


def get_label(num_infected, community_risk, alpha, num_courses):
    label = torch.zeros((num_infected.shape[0], num_courses), dtype=torch.long)
    max_reward = torch.full((num_infected.shape[0], num_courses), -float('inf'))
    allowed_values = torch.zeros((num_infected.shape[0], num_courses))
    new_infected_values = torch.zeros((num_infected.shape[0], num_courses))
    reward_values = torch.zeros((num_infected.shape[0], num_courses))

    rewards_for_actions = torch.zeros((num_infected.shape[0], len(allowed), num_courses))

    allowed_per_course = allowed.view(1, 1, -1).expand(num_infected.shape[0], num_courses, -1)
    new_infected = estimate_infected_students(num_infected, allowed_per_course, community_risk)

    for i, a in enumerate(allowed):
        reward = get_reward(a, new_infected[:, :, i], alpha)
        rewards_for_actions[:, i, :] = reward
        mask = reward > max_reward
        label[mask] = i
        max_reward[mask] = reward[mask]
        allowed_values[mask] = a
        new_infected_values[mask] = new_infected[:, :, i][mask]
        reward_values[mask] = reward[mask]

    return label, allowed_values, new_infected_values, reward_values, rewards_for_actions


def save_to_csv(current_infected, community_risk, label, allowed_values, new_infected_values, reward_values,
                rewards_for_actions, filename):
    num_courses = current_infected.shape[1]
    data_tuples = [
        f'({int(current_infected[i, 0].item() // 10)}, {int(community_risk[i].item() * 10)})'
        for i in range(current_infected.shape[0])
    ]

    sorted_indices = sorted(range(len(data_tuples)), key=lambda i: data_tuples[i])
    sorted_data_tuples = [data_tuples[i] for i in sorted_indices]

    df_data = {
        'Infected and Risk': sorted_data_tuples,
    }

    for course in range(num_courses):
        df_data.update({
            f'Allowed Values Course {course + 1}': [allowed_values[sorted_indices[i], course].item() for i in
                                                    range(100)],
            f'New Infected Values Course {course + 1}': [new_infected_values[sorted_indices[i], course].item() for i in
                                                         range(100)],
            f'Reward Values Course {course + 1}': [reward_values[sorted_indices[i], course].item() for i in range(100)],
            f'Label Course {course + 1}': [label[sorted_indices[i], course].item() for i in range(100)],
            f'Reward 0 Course {course + 1}': [rewards_for_actions[sorted_indices[i], 0, course].item() for i in
                                              range(100)],
            f'Reward 50 Course {course + 1}': [rewards_for_actions[sorted_indices[i], 1, course].item() for i in
                                               range(100)],
            f'Reward 100 Course {course + 1}': [rewards_for_actions[sorted_indices[i], 2, course].item() for i in
                                                range(100)]
        })

    df = pd.DataFrame(df_data)
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

def myopic_policy(campus_model):
    alpha = 0.5
    DIM = 10  # Ensuring a 10x10 grid to get exactly 100 points
    model_name = "myopic_policy-approxSI"
    y, x = torch.tensor(np.mgrid[0:DIM, 0:DIM].reshape(2, -1)).float() / (DIM - 1)

    num_courses = campus_model.num_courses
    students_per_course = campus_model.number_of_students_per_course()
    initial_infection = campus_model.get_initial_infection()

    total_students_per_course = torch.tensor(students_per_course)

    current_infected = torch.zeros((DIM * DIM, num_courses))
    for course in range(num_courses):
        current_infected[:, course] = (1 - y) * students_per_course[course]

    community_risk = x  # 0 to 1

    label, allowed_values, new_infected_values, reward_values, rewards_for_actions = get_label(current_infected,
                                                                                               community_risk, alpha,
                                                                                               num_courses)

    save_to_csv(current_infected, community_risk, label, allowed_values, new_infected_values, reward_values,
                rewards_for_actions, "policy_data_multi_course.csv")

    # Create subplots for each course
    fig, axes = plt.subplots(1, num_courses, figsize=(5 * num_courses, 5), squeeze=False)
    fig.suptitle(f'Policy Label-{model_name} (alpha={alpha})', fontsize=16)

    colors = ['#FF9999', '#66B2FF', '#99FF99']  # Light Red, Light Blue, Light Green
    color_map = {0: colors[0], 1: colors[1], 2: colors[2]}

    for course in range(num_courses):
        ax = axes[0, course]
        scatter = ax.scatter(community_risk.numpy(), current_infected[:, course].numpy(),
                             c=[color_map[l.item()] for l in label[:, course]], s=100, marker='s')
        ax.set_xlabel('Community Risk')
        ax.set_ylabel('Current Infected')
        ax.set_title(f'Course {course + 1}\nTotal Students: {total_students_per_course[course]}')
        ax.grid(False)
        max_val = students_per_course[course]
        # Adjust y-axis limits to show full markers
        y_margin = max_val * 0.05  # 5% margin
        ax.set_ylim(-y_margin, max_val + y_margin)

        # Adjust x-axis limits to show full markers
        ax.set_xlim(-0.05, 1.05)

    # Create a custom legend
    legend_elements = [mpatches.Patch(facecolor=colors[i], edgecolor='black', label=f'Allow {v}')
                       for i, v in enumerate([0, 50, 100])]
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.05),
               ncol=3, fontsize='large')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, wspace=0.3)
    plt.savefig(f"label_{model_name}-{alpha}_multi_course.png", bbox_inches='tight', dpi=300)
    plt.show()

if __name__ == "__main__":
    # Example usage:
    campus_model = campus_model.CampusModel(num_courses=3, students_per_course=[580, 256, 100], initial_infection_rate=[0.2, 0.3, 0.1])
    myopic_policy(campus_model)