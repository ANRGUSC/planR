import math
import pandas as pd
from scipy.stats import binom
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
ROOM_AREA = 669  # [SQFT] for 100 students
ROOM_ACH = 12.12 / 3600  # [1/s]
ROOM_HEIGHT = 1.7  # m
HVAC_EFFICIENCY = 0.8
ACTIVE_INFECTED_TIME = 0.6
MAX_DURATION = 6 * 60  # minutes
BREATH_RATE = 2 * 10 ** -4  # Breathing rate of the occupants
ACTIVE_INFECTED_EMISSION = 10
PASSIVE_INFECTION_EMISSION = 1
D0_values = [1000]  # Different values of D0 for testing
VACCINATION_EFFECT = 0.85
DOSE_VACCINATION_RATIO = 0.25
TRANSMISSION_VACCINATION_RATIO = 0.25
community_risk_values = [i / 10 for i in range(11)]  # Values from 0 to 1
allowed_values = [0, 50, 100]  # Values for allowed
total_students = 100

def calculate_indoor_infection_prob(room_capacity: int, initial_infection_prob: float, D0: float):
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
        if dose_total > 0:
            transmission_prob = 1 - math.exp(-dose_total / D0)
        else:
            transmission_prob = 0
        total_transmission_prob += infection_prob * transmission_prob
    return total_transmission_prob

def get_infected_students(current_infected_students: list, allowed_students_per_course: list,
                          students_per_course: list, community_risk: float, D0: float):
    infected_students = []
    for n, f in enumerate(allowed_students_per_course):
        susceptible = max(0, total_students - current_infected_students[n])
        recovery_rate = 0.1
        room_capacity = allowed_students_per_course[n]
        initial_infection_prob = (current_infected_students[n] / students_per_course[n]) * susceptible / total_students * room_capacity

        infected_prob = calculate_indoor_infection_prob(room_capacity, initial_infection_prob, D0)
        if math.isnan(infected_prob):
            infected_prob = 0
        if math.isnan(allowed_students_per_course[n]):
            allowed_students_per_course[n] = 0
        total_indoor_infected_allowed = int(infected_prob * allowed_students_per_course[n])

        total_infected_allowed_outdoor = int(community_risk * room_capacity)
        total_infected_allowed = min(total_indoor_infected_allowed + total_infected_allowed_outdoor, room_capacity)
        recovered = max(int(recovery_rate * current_infected_students[n]), 0)
        new_infections = min(total_infected_allowed, susceptible)
        new_recoveries = max(int(recovery_rate * current_infected_students[n]), 0)
        net_infections = new_infections - new_recoveries

        updated_infected = max(0, net_infections)
        updated_infected = min(updated_infected, room_capacity)

        infected_students.append(updated_infected)
    return infected_students

# Data for plotting and CSV
results = []

for D0 in D0_values:
    for community_risk in community_risk_values:
        for allowed in allowed_values:
            current_infected_students = [10] * 1  # Example current infected students
            allowed_students_per_course = [allowed] * 1
            students_per_course = [total_students] * 1
            infected_students = get_infected_students(current_infected_students, allowed_students_per_course,
                                                      students_per_course, community_risk, D0)
            results.append({
                "D0": D0,
                "Community Risk": community_risk,
                "Allowed": allowed,
                "Infected Students": infected_students[0]
            })

# Convert results to DataFrame
df = pd.DataFrame(results)

# Save to CSV
df.to_csv("infection_results.csv", index=False)

# Plotting
fig, ax = plt.subplots(figsize=(12, 8))
for D0 in D0_values:
    subset = df[df["D0"] == D0]
    for allowed in allowed_values:
        filtered = subset[subset["Allowed"] == allowed]
        ax.plot(filtered["Community Risk"], filtered["Infected Students"], label=f"D0={D0}, Allowed={allowed}")

ax.set_xlabel("Community Risk")
ax.set_ylabel("Infected Students")
ax.set_title("Infected Students for Different Values of D0 and Allowed Students")
ax.legend()
plt.grid(True)
plt.show()

# Heatmap plotting
for D0 in D0_values:
    pivot_table = df[df["D0"] == D0].pivot("Community Risk", "Allowed", "Infected Students")
    plt.figure(figsize=(12, 10))
    sns.heatmap(pivot_table, annot=True, fmt="d", cmap="plasma", cbar=True)
    plt.title(f"Peak Infections for D0={D0}", pad=20)
    plt.xlabel("Allowed Students")
    plt.ylabel("Community Risk")
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.ylim(-0.5, 10.5)  # Ensure y-axis starts at 0
    plt.savefig(f"peak_infections_D0_{D0}.png")
    plt.show()
