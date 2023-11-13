# Contains logic for calculating infected students based on different models
import math
from scipy.stats import binom
import numpy as np
from prettytable import PrettyTable
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools


# Constants
ROOM_AREA = 1750  # [SQFT] # for 100 students
ROOM_ACH = 12.12 / 3600  # [1/s]
ROOM_HEIGHT = 2.7  # m
HVAC_EFFICIENCY = 0.8
ACTIVE_INFECTED_TIME = 0.2
MAX_DURATION = 2 * 60  # minutes
BREATH_RATE = 2 * 10 ** -4  # Breathing rate of the occupants
ACTIVE_INFECTED_EMISSION = 40
PASSIVE_INFECTION_EMISSION = 1
D0 = 1000  # Constant value for tuning the model.


def calculate_indoor_infection_prob(room_capacity: int, initial_infection_prob: float):
    #occupancy_density = 1 / (ROOM_AREA / 0.092903) # the occupancy density is the number of people per square meter
    occupancy_density = (ROOM_AREA * 0.092903) / room_capacity  # revised occupancy density
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
    correction_factor = 0.8
    for n, f in enumerate(allowed_students_per_course):
        if f == 0:
            infected_students.append(int(community_risk * students_per_course[n] * correction_factor))
        else:
            asymptomatic_ratio = 1
            initial_infection_prob = current_infected_students[n] / (students_per_course[n] * asymptomatic_ratio)
            print("initial infection prob: ", initial_infection_prob)
            room_capacity = allowed_students_per_course[n]
            infected_prob = calculate_indoor_infection_prob(room_capacity, initial_infection_prob)
            print("infected prob: ", infected_prob, "allowed students per course: ", allowed_students_per_course[n])
            total_indoor_infected_allowed = int(infected_prob * allowed_students_per_course[n])
            total_infected_allowed_outdoor = int(community_risk * allowed_students_per_course[n]) * correction_factor
            total_infected_allowed = min(total_indoor_infected_allowed + total_infected_allowed_outdoor,
                                         allowed_students_per_course[n])

            infected_students.append(
                int(round(total_infected_allowed + community_risk * (students_per_course[n] - allowed_students_per_course[n]))))


    return infected_students


def get_infected_students_sir(current_infected, allowed_per_course, community_risk):
    # Simple approximation model that utilizes the community risk

    infected_students = []
    for i in range(len(allowed_per_course)):
        const_1 = 0.005 # reduce this to a smaller value
        const_2 = 0.01 #reduce this value to be very small 0.01, 0.02
        infected = int(((const_1 * current_infected[i]) * (allowed_per_course[i])) + (
                (const_2 * community_risk) * allowed_per_course[i] ** 2))

        infected = min(infected, allowed_per_course[i])

        infected_students.append(infected)

    infected_students = list(map(int, list(map(round, infected_students))))
    return infected_students
#
# # Create a table
# table = []
#
# def generate_data(current_infected_list, allowed_per_course_values, community_risk_values):
#     if not current_infected_list:
#         return
#
#     current_infected = current_infected_list[0]
#
#     for allowed_per_course, community_risk in itertools.product(allowed_per_course_values, community_risk_values):
#         # infected_students = get_infected_students_sir([current_infected], [allowed_per_course], community_risk)
#         infected_students = get_infected_students_sir([current_infected], [allowed_per_course], community_risk)
#
#         row = (current_infected, allowed_per_course, community_risk, infected_students[0])
#         table.append(row)
#
#         # Update the current_infected for the next iteration
#         current_infected = infected_students[0]
#
#     # Recursively call the function with the next values in the list
#     generate_data(current_infected_list[1:], allowed_per_course_values, community_risk_values)
#
# # Define lists for current_infected, allowed_per_course, and community_risk
# initial_current_infected_list = [20]
# allowed_per_course_values = [0, 50, 100]
# # community_risk_values = [0.01 + i * 0.005 for i in range(15)]  # 15 values from 0.01 to 0.1 in increments of 0.005
# community_risk_values = [0.1 + i * 0.06 for i in range(15)]
# # Start the recursive data generation
# generate_data(initial_current_infected_list, allowed_per_course_values, community_risk_values)
#
# # Create a PrettyTable object with column headers
# table_obj = PrettyTable()
# table_obj.field_names = ["Current Infected", "Allowed per Course", "Community Risk", "Infected Students"]
#
# # Populate the table with data
# for row in table:
#     table_obj.add_row(row)
#
# # Set the alignment for numeric columns (optional)
# table_obj.align["Current Infected"] = "r"
# table_obj.align["Allowed per Course"] = "r"
# table_obj.align["Community Risk"] = "r"
# table_obj.align["Infected Students"] = "r"
#
# # Print the table
# table_str = table_obj.get_string()
# with open("table-approximate-sir.txt", "w") as file:
#     file.write(table_str)
#
# # Extract the data from the table
# current_infected = [row[0] for row in table]
# allowed_per_course = [row[1] for row in table]
# community_risk = [row[2] for row in table]
# infected_students = [row[3] for row in table]
#
# # Create a 3D scatter plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# # Scatter plot with allowed_per_course as the color
# scatter = ax.scatter(community_risk, allowed_per_course, infected_students, c=allowed_per_course, cmap='plasma')
#
# # Add labels and title
# ax.set_xlabel('Community Risk')
# ax.set_ylabel('Allowed per Course')
# ax.set_zlabel('Infected Students')
# plt.title('Approximate SIR behavior const_1 = 0.02, const_2 = 0.01')
#
# # Add colorbar to show the mapping of colors to allowed_per_course values
# colorbar = plt.colorbar(scatter, label='Allowed per Course')
#
# # Save the figure
# plt.savefig('3d-scatter-indoor-current.png')




#
# # Prepare data
# current_infected_data = [row[0] for row in table]
# allowed_per_course_data = [row[1] for row in table]
# community_risk_data = [row[2] for row in table]
# infected_students_data = [row[3] for row in table]
#
# # Create a 3D scatter plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# # Scatter plot
# scatter = ax.scatter(current_infected_data, allowed_per_course_data, community_risk_data,
#                      c=infected_students_data, cmap='plasma', marker='o')
#
# # Adding labels and title
# ax.set_xlabel('Current Infected')
# ax.set_ylabel('Allowed Per Course')
# ax.set_zlabel('Community Risk')
# plt.title('Infected Students based on approximated sir model')
#
# # Adding a color bar
# cbar = plt.colorbar(scatter, ax=ax)
# cbar.set_label('Infected Students')
#
# # Save the figure
# plt.savefig('3d-scatter-sir.png')
#
# # Create an instance of PrettyTable
# pt = PrettyTable()
#
# # Define column headers
# pt.field_names = ["Current Infected", "Allowed Per Course", "Community Risk", "Infected Students"]
#
# # Add rows to the table
# for row in table:
#     pt.add_row(row)
#
# # Optionally, you can set the alignment and other formatting options
# pt.align = "r"
# pt.float_format = "0.2"
#
# # Print the table
# table_str = pt.get_string()
# with open("table-sir.txt", "w") as file:
#     file.write(table_str)