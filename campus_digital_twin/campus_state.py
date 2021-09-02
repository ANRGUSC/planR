"""This class represents the campus environment that includes a reward model.

In this environment, the actions that an agent is allowed to take
is to determine the percentage of students to be allowed to be in
a given course at a given week.

Each week the campus environment has a different state represented by
the number of infected students and the community risk value.

Usage example:
state = CampusState()
current_state = state.get_state()

"""
import copy
import numpy as np
from campus_digital_twin import campus_model as cm


# Infection Model
def get_infected_students(current_infected, allowed_per_course, community_risk):
    """Computes the number of infected students per course based on SIR model.
    This infection model can be replaced by another model.
    Args:
        current_infected: The number of infected students at a given week
        allowed_per_course: A list with total students allowed per course
        community_risk: A float value

    Returns:
        A list of infected students per course at a given week

    """

    infected_students = []
    for i in range(len(allowed_per_course)):
        const_1 = 0.025
        const_2 = 0.025

        infected = int(((const_1 * current_infected[i]) * (allowed_per_course[i])) + (
                (const_2 * community_risk) * allowed_per_course[i] ** 2))
        infected = min(infected, allowed_per_course[i])
        percentage_infected = int(infected / allowed_per_course[i] * 100) if \
            allowed_per_course[i] != 0 else 0
        infected_students.append(percentage_infected)

    return infected_students


class CampusState:
    """
    The state every week is represented as a list whose elements is
    the percentage of infected students and index, the course_id.

    Key Variables:
        initialized: boolean
        student_status: list
        current_time: integer
        community_risk: float
        model: object
        counter: integer

    """
    model = cm.CampusModel()
    counter = 0

    def __init__(self, initialized=False, student_status=model.percentage_of_infected_students(),
                 community_risk=model.initial_community_risk(), current_time=0):

        self.initialized = initialized
        self.student_status = student_status
        self.current_time = current_time
        self.community_risk = community_risk[self.current_time]
        self.course_operation_status = None
        self.allowed_students_per_course = []
        self.states = []
        CampusState.counter += 1

    def get_state(self):
        """Get the current state.
        Returns:
            The state is a list representing the percentage of infected students per course

        """
        status = self.get_student_status()
        return status

    def get_course_infection_status(self):
        """Retrieve the number of students from the campus model.
        Return:
            None

        """
        self.model.number_of_students_per_course()

    def get_student_status(self):
        """Get the state of the campus.
        Returns:
            observation
        """
        obs_state = copy.deepcopy(self.student_status)
        obs_state.append(int(self.community_risk * 100))
        return obs_state

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

        students_per_course = self.model.number_of_students_per_course()[0]

        for course, occupancy in enumerate(students_per_course):
            self.allowed_students_per_course.append \
                (int(action[course] / 100 * students_per_course[course]))
        self.update_with_infection_model(action, self.community_risk)
        self.current_time = self.current_time + 1
        self.set_community_risk(self.model.initial_community_risk()[self.current_time - 1])

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

        for course, occupancy in enumerate(students_per_course):  #
            allowed_students_per_course.append \
                (int(action[course] / 100 * students_per_course[course]))

        infected_students = get_infected_students \
            (infected_students, allowed_students_per_course, community_risk)
        self.allowed_students_per_course = allowed_students_per_course
        self.student_status = infected_students

    def get_reward(self, alpha):
        """Calculate the reward given the current state.
        Returns:
            A scalar reward value
        """

        current_infected_students = sum(copy.deepcopy(self.student_status)) \
                                    / len(self.student_status)
        allowed_students = sum(self.allowed_students_per_course) \
                           / len(self.allowed_students_per_course)
        # alpha = 0.85
        beta = 1 - alpha
        reward = alpha * allowed_students - beta * current_infected_students
        reward_list = [int(reward), self.allowed_students_per_course, self.student_status]
        return reward_list
