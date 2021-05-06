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
from campus_digital_twin import campus_model as cm
import numpy as np
import copy


class CampusState():
    """
    The state every week is represented as a list whose elements is
    the percentage of infected students and index, the course_id.

    Attributes:
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
        status = self.get_student_status()
        return status

    def get_course_infection_status(self):
        self.model.number_of_students_per_course()

    # Getters
    def get_student_status(self):
        obs_state = copy.deepcopy(self.student_status)
        obs_state.append(int(self.community_risk * 100))
        return obs_state

    def get_teacher_status(self):
        return self.teacher_status

    def get_course_quarantine_status(self):
        return self.course_quarantine_status

    def get_shut_down(self):
        return self.shut_down

    def get_community_risk(self):
        return self.community_risk

    # Setters

    def set_student_status(self, student_status):
        """
            :type student_status: list
        """
        self.student_status = student_status

    def set_teacher_status(self, teacher_status):
        """
            :type teacher_status: list
        """
        self.teacher_status = teacher_status

    def set_course_quarantine_status(self, course_quarantine_status):
        """
            :type course_quarantine_status: list
        """
        self.course_quarantine_status = course_quarantine_status

    def set_shut_down(self, shut_down):
        """
            :type shut_down: list
        """
        self.shut_down = shut_down

    def set_community_risk(self, community_risk):
        """
        :type community_risk: int
        """
        self.community_risk = community_risk

    def get_observation(self):
        observation = copy.deepcopy(self.student_status)
        observation.append(int(np.round(self.community_risk * 100)))
        return observation


    def update_with_action(self, action):


        students_per_course = self.model.number_of_students_per_course()[0]

        for course, occupancy in enumerate(students_per_course):
            self.allowed_students_per_course.append \
                (int(action[course] / 100 * students_per_course[course]))

        self.update_all(action, self.community_risk)
        self.current_time = self.current_time + 1
        self.set_community_risk(self.model.initial_community_risk()[self.current_time - 1])

    def update_with_infection_model(self, action, community_risk):
        allowed_students_per_course = []
        infected_students = copy.deepcopy(self.student_status)
        students_per_course = self.model.number_of_students_per_course()[0]

        for course, occupancy in enumerate(students_per_course):  #
            allowed_students_per_course.append \
                (int(action[course] / 100 * students_per_course[course]))

        infected_students = self.get_infected_students \
            (infected_students, allowed_students_per_course, community_risk)
        self.allowed_students_per_course = allowed_students_per_course
        self.student_status = infected_students

    def get_infected_students(self, current_infected, allowed_per_course, community_risk):

        infected_students = []
        for i in range(len(allowed_per_course)):
            c1 = 0.025
            c2 = 0.025
            infected = int(((c1 * current_infected[i]) * (allowed_per_course[i])) + (
                    (c2 * community_risk) * allowed_per_course[i] ** 2))
            infected = min(infected, allowed_per_course[i])
            percentage_infected = int(infected / allowed_per_course[i] * 100) if \
                allowed_per_course[i] != 0 else 0
            infected_students.append(percentage_infected)

        return infected_students

    def update_all(self, action, community_risk):
        return self.update_with_infection_model(action, community_risk)

    def get_reward(self):

        current_infected_students = sum(copy.deepcopy(self.student_status)) \
                                    / len(self.student_status)
        allowed_students = sum(self.allowed_students_per_course) \
                           / len(self.allowed_students_per_course)
        a = 0.85
        b = 1 - a
        reward = a * allowed_students - b * current_infected_students
        reward_list = [int(reward), self.allowed_students_per_course, self.student_status]
        return reward_list
