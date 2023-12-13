import math
import copy
import random
from enum import Enum
from models.infection_model import get_infected_students
from models.infection_model import get_infected_students_sir
seed_value = random.randint(1, 1000)
random.seed(seed_value)

# 100HIGH_COMMUNITY_RISK = 0.7
# LOW_COMMUNITY_RISK = 0.3

# Enum for Community Risk
class CommunityRisk(Enum):
    LOW = random.uniform(0.01, 0.055)
    HIGH = random.uniform(0.055, 0.1)

def map_value_to_range(old_value, old_min=0.01, old_max=0.1, new_min=0, new_max=100):
    """Map a value from the old range to the new range."""
    return (old_value - old_min) / (old_max - old_min) * (new_max - new_min) + new_min

class Simulation:
    def __init__(self, model):
        self.current_time = 0
        self.model = model
        # Handle multiple courses dynamically
        self.allowed_students_per_course =[]
        self.student_status = model.initial_infection
        self.state_transition = []
        self.community_risk = random.random()
        self.weekly_infected_students = []
        self.allowed = []
        self.infected = []
        print("initial infected students: ", self.student_status) #debug check

    def set_community_risk_high(self):
        self.community_risk = random.uniform(0.5, 1.0)
        return self.community_risk

    def set_community_risk_low(self):
        self.community_risk = random.uniform(0.0, 0.5)
        return self.community_risk

    def get_student_status(self):
        obs_state = copy.deepcopy(self.student_status)
        # fixme: this is a hack to get the community risk value
        # fixme: this should be removed once the community risk is added to the student_status
        # obs_state.append(int(map_value_to_range(self.community_risk)))
        obs_state.append(int(self.community_risk * 100))

        return obs_state

    def update_with_action(self, action):
        """Updates the campus state object with action.
        Args:
             action: A list with percentage of students to allow for each course.
        Returns:
            None
        """
        if self.current_time < self.model.get_max_weeks():
            self.apply_action(action, self.community_risk)

        return None

    def apply_action(self, action: list, community_risk: float):
        allowed_students_per_course = [
            math.ceil(students * action[i] / 100)
            for i, students in enumerate(self. model.number_of_students_per_course())
        ]
        initial_infection = self.model.get_initial_infection()
        # updated_infected = get_infected_students(self.student_status, allowed_students_per_course,
        #                       self.model.number_of_students_per_course(), initial_infection, community_risk)
        updated_infected = get_infected_students_sir(self.student_status, allowed_students_per_course, community_risk)

        # print("updated infected students: ", updated_infected) #debug check

        # self.state_transition.append((self.student_status, updated_infected))
        self.allowed_students_per_course = allowed_students_per_course
        self.student_status = updated_infected
        # print("allowed students per course: ", self.allowed_students_per_course) #debug check
        # print("student status: ", self.student_status) #debug check
        self.weekly_infected_students.append(sum(updated_infected))

        self.community_risk = random.uniform(0.1, 0.9)

        # if self.current_time >= 7:
        #     self.set_community_risk_low()
        #     # self.community_risk = self.community_risk * self.set_community_risk_low() * random.uniform(0.0, 0.1) + self.community_risk
        # else:
        #     self.set_community_risk_high()
        #     # self.community_risk = self.community_risk * self.set_community_risk_high() * random.uniform(0.0, 0.1) + self.community_risk

        self.current_time += 1

    def get_reward(self, alpha: float):
        current_infected_students = sum(self.student_status)
        allowed_students = sum(self.allowed_students_per_course)
        return int(alpha * allowed_students - ((1 - alpha) * current_infected_students))

    # def get_reward(self, alpha: float):
    #     current_infected_students = sum(self.student_status)
    #     allowed_students = sum(self.allowed_students_per_course)
    #     community_risk = self.community_risk  # Assuming this is an attribute of the class
    #     # Define thresholds
    #     threshold = 0.5
    #
    #     # Define reward values
    #     high_reward_value = 100  # High reward
    #     base_reward = allowed_students
    #
    #     # Evaluate the condition and assign rewards
    #     if community_risk < threshold and allowed_students == 100:
    #         base_reward = allowed_students
    #
    #     if community_risk > threshold and allowed_students == 0:
    #         base_reward = high_reward_value/10
    #     # Adjust reward based on infected students and allowed students
    #     reward = int(alpha * base_reward - ((1 - alpha) * current_infected_students))
    #
    #     return reward

    def is_episode_done(self):
        """
        Determines if the episode has reached its termination point.

        Returns:
            bool: True if the current time has reached the maximum allowed weeks, False otherwise.
        """
        return self.current_time == self.model.get_max_weeks()

    def reset(self):
        self.current_time = 0
        self.allowed_students_per_course = self.model.number_of_students_per_course()
        # print("allowed students per course: ", self.allowed_students_per_course) #debug check
        # self.student_status = [min(int(random.random() * students), 30) for students in self.allowed_students_per_course]
        self.student_status = [random.randint(1, 99) for _ in self.allowed_students_per_course]

        # print("initial infected students: ", self.student_status) #debug check
        self.community_risk = random.uniform(0.0, 1.0)
        return self.get_student_status()





