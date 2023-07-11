"""This class implements the campus_digital_twin environment

The campus environment is composed of the following:
   - Students taking courses.
   - Courses offered by the campus.
   - Community risk provided to the campus every week.

   Agents control the number of students allowed to sit on a course per week.
   Observations consists of an ordered list that contains the number of the
   infected students and the community risk value. Every week the agent proposes what
   percentage of students to allow on campus.

   Actions consists of 3 levels for each course. These levels correspond to:
    - 0%: schedule class online
    - 50%: schedule 50% of the class online
    - 100%: schedule the class offline

   An episode ends after 15 steps (Each step represents a week).
   We assume an episode represents a semester.

"""
import gymnasium as gym
# import gym
from campus_digital_twin import campus_state as cs
import numpy as np
import json
import logging
logging.basicConfig(filename="run.txt", level=logging.INFO)
def list_to_int(s, n):
    s.reverse()
    a = 0
    for i in range(len(s)):
        a = a + s[i] * pow(n, i)
    return a


# get the multidiscrete list from an index
def int_to_list(num, n, size):
    outlist = [0] * size
    i = 0
    while num != 0:
        bit = num % n
        outlist[i] = bit
        num = (int)(num / n)
        i = i + 1
    outlist.reverse()
    return outlist


def get_discrete_value(number):
    value = 0
    if number in range(0, 11):
        value = 0
    elif number in range(11, 21):
        value = 1
    elif number in range(21, 31):
        value = 2
    elif number in range(31, 41):
        value = 3
    elif number in range(41, 51):
        value = 4
    elif number in range(51, 61):
        value = 5
    elif number in range(61, 71):
        value = 6

    elif number in range(71, 81):
        value = 7
    elif number in range(81, 91):
        value = 8
    elif number in range(91, 101):
        value = 9
    return value


# convert actions to discrete values 0,1,2
def action_conv_disc(action_or_state):
    discaction = []
    for i in (action_or_state):
        action_val = get_discrete_value(i)
        discaction.append(action_val)
    return discaction


# convert list of discrete values to 0 to 100 range
def disc_conv_action(discaction):
    # action_taken = [discaction]
    # print("Action taken", action_taken)
    action = []
    for i in range(len(discaction)):
        action.append((int)(discaction[i] * 50))
    return action


class CampusGymEnv(gym.Env):
    """
    Observation:
        Type: Multidiscrete([0, 1 ..., n+1]) where n is the number of courses and the last item is the community risk value.
        Example observation: [20, 34, 20, 0.5]
    Actions:
        Type: Multidiscrete([0, 1 ... n]) where n is the number of courses.
        Example action: [0, 1, 1]
    Reward:
        Reward is returned from the campus environment
        as a scalar value.A high reward corresponds
        to an increase in the number of allowed students.
    Starting State:
        All observations are obtained from a static information provided by a campus model.

    Episode Termination:
        The campus environment stops running after n steps where n represents the duration of campus operation.
    """
    metadata = {'render.modes': ['bot']}

    def __init__(self):
        # Create a new campus state object
        self.csobject = cs.CampusState()
        total_courses = self.csobject.model.total_courses()

        # Set the infection levels and occupancy level to minimize space
        num_infec_levels = 10
        num_occup_levels = 3

        self.action_space = gym.spaces.MultiDiscrete \
            ([num_occup_levels for _ in range(total_courses)])
        self.observation_space = gym.spaces.MultiDiscrete \
            ([num_infec_levels for _ in range(total_courses + 1)])

    def step(self, action):
        """Take action.
        Args:
            action: Type (list)
        Returns:
            observation: Type(list)
            reward: Type(int)
            done: Type(bool)
        """
        # Remove alpha from list of action.
        # print(f'action: {action}')
        if not isinstance(action, list):
            action = [action, 0.5]

        alpha = action[-1]
        action.pop()
        self.csobject.update_with_action(action)

        observation = np.array(action_conv_disc(self.csobject.get_state()))
        reward = self.csobject.get_reward(alpha)
        done = False
        if self.csobject.current_time == self.csobject.model.get_max_weeks():
            done = True


        info = {"allowed": self.csobject.allowed_students_per_course, "infected": self.csobject.student_status, "reward": reward}
        # print(info)
        logging.info(info)
        self.reward = reward


        # return observation, reward, done, info
        return observation, reward, done, False, info

    def reset(self):
        """Reset the current time.
        Returns:
            state: Type(list)
        """
        # self.csobject.current_time = 0
        # if seed:
        #     super().reset(seed=seed)
        state = self.csobject.reset()
        str_state = "reset state: " + str(state)
        logging.info(str_state)
        dstate = action_conv_disc(state)
        # print(f'dstate: {dstate}')
        info = {}
        return np.array(dstate), info

    def render(self, mode='bot'):
        """Render the environment.
        Returns:
            state: Type(list)
        """
        print("Number of infected students: ", self.csobject.get_test_observations())
        return None
