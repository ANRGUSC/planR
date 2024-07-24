import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from scipy.stats import binom

def estimate_infected_students(current_infected, allowed_per_course, community_risk):
    const_1 = 0.005
    const_2 = 0.01

    infected = ((const_1 * current_infected) * allowed_per_course +
                (const_2 * community_risk) * allowed_per_course ** 2)

    infected = torch.min(infected, allowed_per_course)
    return infected

def estimate_infected_students_sir(current_infected, allowed_per_course, community_risk):
    const_1 = 0.001
    const_2 = 0.01

    total_students = 100
    recovery_rate = 1.0

    susceptible = torch.max(torch.tensor(0.0), total_students - current_infected)

    new_infected_inside = (const_1 * current_infected * (susceptible / 100) * susceptible * allowed_per_course)
    new_infected_outside = const_2 * community_risk * allowed_per_course * allowed_per_course
    recovered = torch.max(torch.tensor(0.0), recovery_rate * current_infected)

    total_infected = new_infected_inside + new_infected_outside
    infected = torch.min(current_infected + total_infected - recovered, allowed_per_course)

    return infected

# Constants
ROOM_AREA = 669  # default: 869  # [SQFT] # for 100 students
ROOM_ACH = 12.12 / 3600  # [1/s]
ROOM_HEIGHT = 2.7  # m
HVAC_EFFICIENCY = 0.8
ACTIVE_INFECTED_TIME = 0.1
MAX_DURATION = 12 * 60  # minutes
BREATH_RATE = 2 * 10 ** -4  # Breathing rate of the occupants
ACTIVE_INFECTED_EMISSION = 40
PASSIVE_INFECTION_EMISSION = 1
D0_values = [500, 1000, 2000]  # Different values of D0 for testing
D0 = 1000
VACCINATION_EFFECT = 0.85
DOSE_VACCINATION_RATIO = 0.25
TRANSMISSION_VACCINATION_RATIO = 0.25
community_risk_values = [i / 10 for i in range(11)]  # Values from 0 to 1
allowed_values = [0, 50, 100]  # Updated values for allowed
total_students = 100

def calculate_indoor_infection_prob(room_capacity: int, initial_infection_prob: float):
    occupancy_density = room_capacity / (ROOM_AREA * 0.092903)
    dose_one_person = (
            occupancy_density * BREATH_RATE /
            (ROOM_HEIGHT * HVAC_EFFICIENCY * ROOM_ACH) *
            (ACTIVE_INFECTED_TIME * ACTIVE_INFECTED_EMISSION +
             (1 - ACTIVE_INFECTED_TIME) * PASSIVE_INFECTION_EMISSION) * MAX_DURATION
    )
    total_transmission_prob = 0
    for i in range(0, room_capacity):
        infection_prob = binom.pmf(i, room_capacity, initial_infection_prob)
        dose_total = i * dose_one_person
        # if dose_total > 0:
        transmission_prob = 1 - math.exp(-dose_total / D0)
        # else:
        #     transmission_prob = 0
        total_transmission_prob += infection_prob * transmission_prob

    return total_transmission_prob

def get_infected_students(current_infected_students, allowed_students_per_course, community_risk: float):
    susceptible = torch.max(torch.tensor(0.0), torch.tensor(total_students) - current_infected_students)
    recovery_rate = 0.1
    room_capacity = allowed_students_per_course
    initial_infection_prob = (current_infected_students / total_students * susceptible/total_students)

    infected_prob = calculate_indoor_infection_prob(room_capacity, initial_infection_prob)
    infected_prob = torch.tensor(infected_prob) if not isinstance(infected_prob, torch.Tensor) else infected_prob
    allowed_students_per_course = torch.tensor(allowed_students_per_course) if not isinstance(
        allowed_students_per_course, torch.Tensor) else allowed_students_per_course

    if torch.any(torch.isnan(infected_prob)):
        infected_prob = torch.tensor(0.0)  # or another default value
    if torch.any(torch.isnan(allowed_students_per_course)):
        allowed_students_per_course = torch.tensor(0)  # or another default value

    total_indoor_infected_allowed = (infected_prob * allowed_students_per_course).int()
    total_infected_allowed_outdoor = (community_risk * allowed_students_per_course).int()
    total_infected_allowed = torch.min(total_indoor_infected_allowed + total_infected_allowed_outdoor,
                                       allowed_students_per_course)

    recovered = torch.max(torch.tensor(0.0), recovery_rate * current_infected_students).int()
    infected_students = torch.min(current_infected_students + total_infected_allowed - recovered,
                                  allowed_students_per_course).int()
    return infected_students

def get_reward(allowed, new_infected, alpha: float):
    reward = (alpha * allowed) - ((1 - alpha) * new_infected)
    return reward

allowed = torch.tensor([0, 50, 100])  # Updated allowed values

def get_label(num_infected, community_risk, alpha):
    label = torch.zeros(num_infected.shape[0], dtype=torch.long)
    max_reward = torch.full(num_infected.shape, -float('inf'))
    allowed_values = torch.zeros(num_infected.shape[0])
    new_infected_values = torch.zeros(num_infected.shape[0])
    reward_values = torch.zeros(num_infected.shape[0])

    rewards_for_actions = torch.zeros((num_infected.shape[0], len(allowed)))

    for i, a in enumerate(allowed):
        new_infected = estimate_infected_students(num_infected, a, community_risk)
        reward = get_reward(a, new_infected, alpha)
        rewards_for_actions[:, i] = reward
        mask = reward > max_reward
        label[mask] = i
        max_reward[mask] = reward[mask]
        allowed_values[mask] = a
        new_infected_values[mask] = new_infected[mask]
        reward_values[mask] = reward[mask]

    return label, allowed_values, new_infected_values, reward_values, rewards_for_actions


def save_to_csv(current_infected, community_risk, label, allowed_values, new_infected_values, reward_values,
                rewards_for_actions, filename):
    data_tuples = [
        f'({int(current_infected[i].item() // 10)}, {int(community_risk[i].item() * 10)})'
        for i in range(current_infected.shape[0])
    ]

    sorted_indices = sorted(range(len(data_tuples)), key=lambda i: data_tuples[i])
    sorted_data_tuples = [f'({data_tuples[i][0]}, {data_tuples[i][1]})' for i in sorted_indices]

    df = pd.DataFrame({
        'Infected and Risk': [sorted_data_tuples[i] for i in range(100)],
        'Allowed Values': [allowed_values[sorted_indices[i]].item() for i in range(100)],
        'New Infected Values': [new_infected_values[sorted_indices[i]].item() for i in range(100)],
        'Reward Values': [reward_values[sorted_indices[i]].item() for i in range(100)],
        'Label': [label[sorted_indices[i]].item() for i in range(100)],
        'Reward 0': [rewards_for_actions[sorted_indices[i], 0].item() for i in range(100)],
        'Reward 50': [rewards_for_actions[sorted_indices[i], 1].item() for i in range(100)],
        'Reward 100': [rewards_for_actions[sorted_indices[i], 2].item() for i in range(100)]
    })
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")


def myopic_policy():
    alpha = 0.5
    DIM = 10  # Ensuring a 10x10 grid to get exactly 100 points
    model_name = "myopic_policy-approxSI"
    y, x = torch.tensor(np.mgrid[0:DIM, 0:DIM].reshape(2, -1)).float() / (DIM - 1)
    current_infected = (1 - y) * 100  # 0 to 100
    community_risk = x  # 0 to 1
    label, allowed_values, new_infected_values, reward_values, rewards_for_actions = get_label(current_infected,
                                                                                               community_risk, alpha)

    save_to_csv(current_infected, community_risk, label, allowed_values, new_infected_values, reward_values,
                rewards_for_actions, "policy_data.csv")

    # Define RGB colors for the labels
    colors = np.array([[1, 0, 0],  # Red for label 0
                       [0, 1, 0],  # Green for label 1
                       [0, 0, 1]])  # Blue for label 2

    # Create the scatter plot
    plt.figure(figsize=(10, 10))
    label_colors = colors[label.numpy().reshape(-1)]
    plt.scatter(community_risk.numpy(), current_infected.numpy(), c=label_colors, s=500,
                marker='s')  # 's' for square marker
    plt.xlabel('Community Risk')
    plt.ylabel('Current Infected')
    plt.title(f'Policy Label-{model_name} (alpha={alpha})')

    # Create a custom legend with square patches
    import matplotlib.patches as mpatches
    red_patch = mpatches.Patch(color='red', label='Allow 0', edgecolor='black')
    green_patch = mpatches.Patch(color='green', label='Allow 50', edgecolor='black')
    blue_patch = mpatches.Patch(color='blue', label='Allow 100', edgecolor='black')
    # Place the legend outside the plot
    plt.legend(handles=[red_patch, green_patch, blue_patch], loc='lower left', bbox_to_anchor=(1, 0),
               fontsize='x-large')


    plt.grid(False)  # Remove grid background
    plt.savefig(f"label_{model_name}-{alpha}.png")
    plt.show()


if __name__ == "__main__":
    myopic_policy()

