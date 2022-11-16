"""This module updates the state of the campus model by applying an infection
model every time an action is taken. The ac
"""

import math
from scipy.stats import binom
import copy
import numpy as np
from campus_digital_twin import campus_model as cm
import logging
import time
import random
import wandb

random.seed(10)


def calculate_indoor_infection_prob(room_capacity, initial_infection_prob):
    # print(initial_infection_prob)
    room_area = 469  # The area of the room in [SQFT]
    room_ach = 12.12 / 3600  # The air change rate of the room in [1/s]
    room_height = 2.7  # The height of the room
    hvac = 0.8  # The HVAC system efficiency of the room
    active_infected_time = 0.2  # The active time of the infected occupants
    vaccination_effect = 0  # The probability of the vaccine being effective on the vaccinated occupants
    dose_vaccinated_ratio = 1  # The ratio of the dose emitted via the infected vaccinated occupants
    transmission_vaccinated_ratio = 1  # The infection probability of the vaccinated occupants get infected
    max_duration = 2 * 60  # minutes
    f_in = 0
    f_out = 0
    breath_rate = 2 * 10 ** -4  # Breathing rate of the occupants
    active_infected_emission = 40  # The emission rate for the active infected occupants
    passive_infection_emission = 1  # The emission rate for the passive infected occupants
    D0 = 1000 # Constant value for tuning the model
    vaccination_ratio = 0

    occupancy_density = 1 / room_area / 0.092903
    dose_one_person = (1 - f_in) * (1 - f_out) * (occupancy_density * breath_rate) / \
                      (room_height * hvac * room_ach) * \
                      (active_infected_time * active_infected_emission + (1 - active_infected_time) *
                       passive_infection_emission) * max_duration
    dose_one_person = dose_one_person * (vaccination_ratio * vaccination_effect * dose_vaccinated_ratio +
                                         vaccination_ratio * (1 - vaccination_effect) +
                                         (1 - vaccination_ratio))

    total_transmission_prob = 0
    for i in range(0, room_capacity):
        infection_prob = binom.pmf(i, room_capacity, initial_infection_prob)
        dose_total = i * dose_one_person
        transmission_prob = 1 - math.exp(-dose_total / D0)
        total_transmission_prob += infection_prob * transmission_prob

    total_transmission_prob *= (vaccination_ratio * vaccination_effect * transmission_vaccinated_ratio +
                                vaccination_ratio * (1 - vaccination_effect) +
                                (1 - vaccination_ratio))

    return total_transmission_prob


def get_infected_students_sir(current_infected, allowed_per_course, community_risk):
    # Simple approximation model that utilizes the community risk
    logging.info(f'Allowed: {allowed_per_course}')
    logging.info(f'Infected: {current_infected}')
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


# Infection Model
def get_infected_students(current_infected, allowed_per_course, students_per_course, initial_infection, community_risk):
    """ Number of total infected students in week n+1 that could come in = asymptomatic ratio *
    number of students infected in week n + community risk * total number of students

        Number of allowed infected students = Number of total infected students *
        allowed students / total number of students

    Returns:
        A list of infected students per course at a given week
    """

    infected_students = []
    for n, f in enumerate(allowed_per_course):
        if f == 0:
            infected_students.append(0)

        else:
            asymptomatic_ratio = 0.5
            initial_infection_prob = current_infected[n]/students_per_course[n] * asymptomatic_ratio
            room_capacity = students_per_course[n]
            infected_prob = calculate_indoor_infection_prob(room_capacity, initial_infection_prob)
            total_indoor_infected_allowed = int(infected_prob * allowed_per_course[n])
            total_infected_allowed_outdoor = int(community_risk * allowed_per_course[n])
            total_infected_allowed = min(total_indoor_infected_allowed + total_infected_allowed_outdoor, allowed_per_course[n])
            #infected_students.append(int(total_infected_allowed ))

            infected_students.append(int(total_infected_allowed + community_risk*(students_per_course[n] - allowed_per_course[n])))

    # print("infected and allowed: ", infected_students, allowed_per_course, community_risk)
    return infected_students


class CampusState:
    """
    The state every week is represented as a list whose elements are
    the number of infected students and index, the course_id.\\

    Key Variables:
        student_status: list representing the percentage of infected students.
        community_risk: float value that is currently assumed to
        model: object representing campus data

    """
    model = cm.CampusModel()
    # counter = 0

    def __init__(self, initialized=False, student_status=model.number_of_infected_students_per_course(),
                 test_student_status = model.test_number_of_infected_students_per_course(),
                 community_risk=model.initial_community_risk(), current_time=0,
                 weeks=model.initial_community_risk(),
                 allowed_per_course=cm.CampusModel().number_of_students_per_course()[0]):

        self.initialized = initialized
        self.student_status = student_status
        self.test_student_status = test_student_status
        self.current_time = current_time
        self.community_risk = community_risk[self.current_time]
        self.allowed_students_per_course = allowed_per_course
        self.weeks = weeks
        self.states = []
        # CampusState.counter += 1

        print("Total infected students per course", self.student_status)
        print("Total students per course", self.allowed_students_per_course)

    def get_state(self):
        """
        Returns:
            A  infected students per course after allowing a certain
            percentage of students in a course
        """
        state = self.get_student_status()
        return state

    def set_state(self):
        self.student_status = self.model.number_of_infected_students_per_course().copy()
        return

    def get_course_infection_status(self):
        """Retrieves the number of students from the campus model.
        Return:
            None
        """
        self.model.number_of_infected_students_per_course()

    def get_student_status(self):
        """Get the state of the campus.
        Returns:
            observation
        """
        obs_state = copy.deepcopy(self.student_status)
        obs_state.append(int(self.community_risk * 100))
        # print("obs_state: ", obs_state)
        return obs_state

    def get_community_risk(self):
        return self.community_risk

    def set_community_risk_high(self):
        """Get the community risk value
        Returns:
            community_risk: Float
        """
        self.community_risk = random.uniform(0.6, 1.0)
        return self.community_risk

    def set_community_risk_low(self):
        """
        :type community_risk: int

        """

        self.community_risk = random.uniform(0.1, 0.5)
        # if self.current_time <= self.model.get_max_weeks():
        #     self.community_risk = self.model.initial_community_risk()[self.current_time]
        #
        # else:
        #     self.community_risk = self.model.initial_community_risk()[0]
        #self.current_time = self.current_time + 1
        return self.community_risk

    def get_observation(self):
        """Get the state of the campus.
        Returns:
            observation
        """
        observation = copy.deepcopy(self.student_status)
        observation.append(int(self.community_risk * 100))
        return observation

    def get_test_observations(self):
        test_observation = copy.deepcopy(self.test_student_status)
        test_observation.append(int(self.community_risk * 100))
        return test_observation



    def update_with_action(self, action):
        """Updates the campus state object with action.
        Args:
             action: A list with percentage of students to allow for each course.
        Returns:
            None
        """
        #print("action: ", action)
        if self.current_time < self.model.get_max_weeks():

            self.update_with_infection_model(action, self.community_risk)

        return None

    def update_with_infection_model(self, action, community_risk):
        """Updates the observation with the number of students infected per course.
        Args:
            action: a list with percentage of students to be allowed in a course
            community_risk: a float value that is provided by an external entity.
        Returns:
            None
        """
        allowed_students_per_course = []
        infected_students = self.student_status.copy()
        students_per_course = self.model.number_of_students_per_course()[0]
        initial_infection = self.model.number_of_infected_students_per_course()

        for i, action in enumerate(action):
            # calculate allowed per course

            allowed = math.ceil((self.model.number_of_students_per_course()[0][i] * action)/100)
            allowed_students_per_course.append(allowed)

        """
        Uncomment/comment to get infected students where one model uses an approximation model based on sir while the 
        other one uses one based on an indoor transmission risk model.
        """
        updated_infected = get_infected_students\
            (infected_students, allowed_students_per_course, students_per_course, initial_infection, community_risk)


        # infected = get_infected_students_sir\
        #     (infected_students, allowed_students_per_course, community_risk)
        self.allowed_students_per_course = allowed_students_per_course[:]
        self.student_status = updated_infected[:]
        if self.current_time >= 7:
            self.set_community_risk_low()
        else:
            self.set_community_risk_high()

        self.current_time = self.current_time + 1
        return None



        #return allowed_students_per_course, updated_infected

    def get_reward(self):
        """Calculate the reward given the current state.
        Returns: estimated reward per step.
        """

        current_infected_students = sum(copy.deepcopy(self.student_status))
        allowed_students = sum(self.allowed_students_per_course)
        alpha = 0.9
        reward = alpha * allowed_students - ((1-alpha) * current_infected_students)
        # diff = []
        # beta = 1 - alpha
        # for n, m in zip(self.student_status, self.allowed_students_per_course):
        #     d = (m * alpha) - (beta * n)
        #     diff.append(d)
        # reward = int(sum(diff))

        return int(reward)

    def reset(self):
        self.current_time = 0
        self.allowed_students_per_course = self.model.number_of_students_per_course()[0]
        #self.student_status = self.model.number_of_infected_students_per_course()[:]
        for i in range(len(self.allowed_students_per_course)):
            self.student_status[i] = int(random.random() * self.allowed_students_per_course[i])
        self.community_risk = random.random()

        return self.get_student_status()


