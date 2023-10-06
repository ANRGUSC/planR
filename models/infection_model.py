# Contains logic for calculating infected students based on different models
import math
from scipy.stats import binom
import numpy as np

# Constants
ROOM_AREA = 469  # [SQFT]
ROOM_ACH = 12.12 / 3600  # [1/s]
ROOM_HEIGHT = 2.7  # m
HVAC_EFFICIENCY = 0.8
ACTIVE_INFECTED_TIME = 0.2
MAX_DURATION = 2 * 60  # minutes
BREATH_RATE = 2 * 10 ** -4  # Breathing rate of the occupants
ACTIVE_INFECTED_EMISSION = 40
PASSIVE_INFECTION_EMISSION = 1
D0 = 10  # Constant value for tuning the model.


def calculate_indoor_infection_prob(room_capacity: int, initial_infection_prob: float):
    occupancy_density = 1 / (ROOM_AREA / 0.092903)
    dose_one_person = (
        (1 - occupancy_density * BREATH_RATE) /
        (ROOM_HEIGHT * HVAC_EFFICIENCY * ROOM_ACH) *
        (ACTIVE_INFECTED_TIME * ACTIVE_INFECTED_EMISSION +
         (1 - ACTIVE_INFECTED_TIME) * PASSIVE_INFECTION_EMISSION) * MAX_DURATION
    )
    total_transmission_prob = 0
    for i in range(0, room_capacity):
        infection_prob = binom.pmf(i, room_capacity, initial_infection_prob)
        dose_total = i * dose_one_person
        transmission_prob = 1 - math.exp(-dose_total / D0)
        total_transmission_prob += infection_prob * transmission_prob

    return total_transmission_prob


def get_infected_students(current_infected_students: list, allowed_students_per_course: list,
                          students_per_course: list, initial_infection: list, community_risk: float):
    infected_students = []
    for n, f in enumerate(allowed_students_per_course):
        if f == 0:
            correction_factor = 1
            infected_students.append(int(community_risk * students_per_course[n] * correction_factor))
        else:
            asymptomatic_ratio = 0.5
            initial_infection_prob = current_infected_students[n] / students_per_course[n] * asymptomatic_ratio
            room_capacity = allowed_students_per_course[n]
            infected_prob = calculate_indoor_infection_prob(room_capacity, initial_infection_prob)
            total_indoor_infected_allowed = int(infected_prob * allowed_students_per_course[n])
            total_infected_allowed_outdoor = int(community_risk * allowed_students_per_course[n])
            total_infected_allowed = min(total_indoor_infected_allowed + total_infected_allowed_outdoor,
                                         allowed_students_per_course[n])

            infected_students.append(
                int(round(total_infected_allowed + community_risk * (students_per_course[n] - allowed_students_per_course[n]))))

    return infected_students

def get_infected_students_sir(current_infected, allowed_per_course, community_risk):
    # Simple approximation model that utilizes the community risk

    infected_students = []
    for i in range(len(allowed_per_course)):
        const_1 = 0.5
        const_2 = 0.25
        infected = int(((const_1 * current_infected[i]) * (allowed_per_course[i])) + (
                (const_2 * community_risk) * allowed_per_course[i] ** 2))

        infected = min(infected, allowed_per_course[i])

        infected_students.append(infected)

    infected_students = list(map(int, list(map(round, infected_students))))
    return infected_students