"""This module updates the state of the campus model by applying an infection
model every time an action is taken. The ac
"""
import math
from scipy.stats import binom
import copy
import numpy as np
from campus_digital_twin import campus_model as cm
import logging


def calculate_indoor_infection_prob(room_capacity, initial_infection_prob):
    # print(initial_infection_prob)
    room_area = 869  # The area of the room in [SQFT]
    room_ach = 12.12 / 3600  # The air change rate of the room in [1/s]
    room_height = 2.7  # The height of the room
    hvac = 0.8  # The HVAC system efficiency of the room
    active_infected_time = 0.9  # The active time of the infected occupants
    vaccination_effect = 0  # The probability of the vaccine being effective on the vaccinated occupants
    dose_vaccinated_ratio = 1  # The ratio of the dose emitted via the infected vaccinated occupants
    transmission_vaccinated_ratio = 1  # The infection probability of the vaccinated occupants get infected
    max_duration = 2 * 60  # minutes
    f_in = 0
    f_out = 0
    breath_rate = 2 * 10 ** -4  # Breathing rate of the occupants
    active_infected_emission = 40  # The emission rate for the active infected occupants
    passive_infection_emission = 1  # The emission rate for the passive infected occupants
    D0 = 1000  # Constant value for tuning the model
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
        const_1 = 0.025
        const_2 = 0.025
        infected = int(((const_1 * current_infected[i]) * (allowed_per_course[i])) + (
                (const_2 * community_risk) * allowed_per_course[i] ** 2))

        infected = min(infected, allowed_per_course[i])

        infected_students.append(infected)

    infected_students = list(map(int, list(map(round, infected_students))))
    return infected_students


# Infection Model
def get_infected_students(current_infected, allowed_per_course, students_per_course):
    """This function makes an assumption that each classroom has similar physical characteristics with
    varying room capacity. This change is room capacity is due to the actions suggested by the RL agent.


    Returns:
        A list of infected students per course at a given week
    """
    # print(current_infected, allowed_per_course)
    infected_students = []
    logging.info(f'Allowed: {allowed_per_course}')
    logging.info(f'Infected: {current_infected}')
    for n, f in enumerate(current_infected):
        initial_infection_prob = f / students_per_course[n]
        room_capacity = allowed_per_course[n]

        infected = calculate_indoor_infection_prob(room_capacity, initial_infection_prob)

        total_infected = (infected * room_capacity)

        infected_students.append(math.ceil(total_infected))

    return infected_students


class CampusState:
    """
    The state every week is represented as a list whose elements are
    the number of infected students and index, the course_id.

    Key Variables:
        student_status: list representing the percentage of infected students.
        community_risk: float value that is currently assumed to
        model: object representing campus data

    """
    model = cm.CampusModel()
    logging.info(f'Course capacity: {model.number_of_students_per_course()[0]}')
    counter = 0

    def __init__(self, initialized=False, student_status=model.number_of_infected_students_per_course(),
                 community_risk=model.initial_community_risk(), current_time=0,
                 allowed_per_course=model.number_of_students_per_course()[0]):
        self.initialized = initialized
        self.student_status = student_status
        self.current_time = current_time
        self.community_risk = community_risk[self.current_time]
        self.course_operation_status = None
        self.allowed_students_per_course = allowed_per_course
        self.states = []
        CampusState.counter += 1

    def get_state(self):
        """
        Returns:
            A  infected students per course after allowing a certain
            percentage of students in a course
        """
        state = self.get_student_status()
        return state

    def set_state(self):
        state = self.model.number_of_infected_students_per_course()
        state.append(int(self.community_risk * 100))
        return state

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
        return list(obs_state)

    def get_community_risk(self):
        """Get the community risk value
        Returns:
            community_risk: Float
        """
        return self.community_risk

    def set_community_risk(self, community_risk):
        """
        :type community_risk: int
        """
        self.community_risk = community_risk

    def get_observation(self):
        """Get the state of the campus.
        Returns:
            observation
        """
        observation = copy.deepcopy(self.student_status)
        observation.append(int(np.round(self.community_risk * 100)))
        return observation

    def update_with_action(self, action):
        """Updates the campus state object with action.
        Args:
             action: A list with percentage of students to allow for each course.
        Returns:
            None
        """
        self.update_with_infection_model(action, self.community_risk)
        self.current_time = self.current_time + 1
        self.set_community_risk(self.model.initial_community_risk()[self.current_time - 1])
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
        infected_students = copy.deepcopy(self.student_status)
        students_per_course = self.model.number_of_students_per_course()[0]

        for i, action in enumerate(action):
            # calculate allowed per course
            allowed = math.ceil(self.model.number_of_students_per_course()[0][i] * (action / 100))
            allowed_students_per_course.append(allowed)

        # infected = get_infected_students \
        #     (infected_students, allowed_students_per_course, students_per_course)

        infected = get_infected_students_sir\
            (infected_students, allowed_students_per_course, community_risk)

        self.allowed_students_per_course = allowed_students_per_course[:]
        self.student_status = infected

        return None

    def get_reward(self, alpha):
        """Calculate the reward given the current state.
        Returns:
            A list reward value
        """

        current_infected_students = sum(copy.deepcopy(self.student_status)) \
                                    / len(self.student_status)
        allowed_students = sum(copy.deepcopy(self.allowed_students_per_course)) \
                           / len(self.allowed_students_per_course)
        beta = 1 - alpha
        reward = int(alpha * allowed_students - beta * current_infected_students)
        return reward
