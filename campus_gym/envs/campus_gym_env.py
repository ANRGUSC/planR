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
import itertools
import gymnasium as gym
from campus_digital_twin import campus_model, campus_state
import numpy as np
import logging
logging.basicConfig(filename="run.txt", level=logging.INFO)

def get_discrete_value(number):
    """
    Converts a given number to a discrete value based on its range.

    Parameters:
    number (int or float): The input number to be converted to a discrete value.

    Returns:
    int: A discrete value representing the range in which the input number falls.
         It returns a value between 0 and 9, inclusive.

    Example:
    get_discrete_value(25) returns 2
    get_discrete_value(99) returns 9
    """

    # Ensure the number is within the range [0, 100]
    number = min(99, max(0, number))

    # Perform integer division by 10 to get the discrete value
    # This will also ensure that the returned value is an integer
    return number // 10


def convert_actions_to_discrete(action_or_state):
    """
    Converts a list of state values to a list of discrete values [0, 1, 2].

    This function applies the get_discrete_value function to each element in the input list,
    converting them to discrete values and returning the new list of discrete values.

    Parameters:
    action_or_state (list of int or float): A list containing action or state values to be converted.

    Returns:
    list of int: A list containing the converted discrete values.

    Example:
    convert_actions_to_discrete([15, 25, 35]) returns [1, 2, 3]
    """

    # Use list comprehension to apply get_discrete_value to each element in action_or_state
    discrete_actions_list = [get_discrete_value(value) for value in action_or_state]

    return discrete_actions_list


def disc_conv_action(discrete_actions_list):
    """
    Converts a list of discrete action values to a list of actions in the range [0, 100].

    Parameters:
    discrete_actions_list (list of int): A list containing discrete action values.

    Returns:
    list of int: A list containing converted action values in the range [0, 100].

    Example:
    disc_conv_action([0, 1, 2]) returns [0, 50, 100]
    """

    # Use list comprehension to convert each discrete action value
    # in discrete_actions_list to the range [0, 100]
    return [(int)(val * 50) for val in discrete_actions_list]


class CampusGymEnv(gym.Env):
    """
        Defines a Gym environment representing a campus scenario where agents control
        the number of students allowed to sit on a course per week, considering the
        risk of infection and the reward is based on the number of allowed students.

        Observation:
            Type: Multidiscrete([0, 1 ..., n+1]) where n is the number of courses, and
            the last item is the community risk value.
            Example observation: [20, 34, 20, 0.5]

        Actions:
            Type: Multidiscrete([0, 1 ... n]) where n is the number of courses.
            Example action: [0, 1, 1]

        Reward:
            Reward is a scalar value returned from the campus environment.
            A high reward corresponds to an increase in the number of allowed students.

        Episode Termination:
            The episode terminates after n steps, representing the duration of campus operation.
        """
    metadata = {'render.modes': ['bot']}

    def __init__(self, alpha=0.38):

        # Initialize a new campus state object
        self.campus_state = campus_state.Simulation(model=campus_model.CampusModel())
        total_courses = campus_model.CampusModel().num_courses

        # Define action and observation spaces
        num_infection_levels = 10
        num_occupancy_levels = 3
        self.alpha = alpha
        self.action_space = gym.spaces.MultiDiscrete([num_occupancy_levels] * total_courses) # [3,3,3]
        self.observation_space = gym.spaces.MultiDiscrete([num_infection_levels] * (total_courses + 1))

        self.possible_actions = [list(range(0, (k))) for k in self.action_space.nvec]
        self.possible_states = [list(range(0, (k))) for k in self.observation_space.nvec]
        self.all_actions = [str(i) for i in list(itertools.product(*self.possible_actions))]
        self.all_states = [str(i) for i in list(itertools.product(*self.possible_states))]


    def step(self, action):
        """
            Execute one time step within the environment.
        """

        # Extract alpha from the list of action and update the campus state with the action

        alpha = self.alpha
        # print("alpha: ", alpha)
        list_action = list(eval(self.all_actions[action]))
        c_list_action = [i * 50 for i in list_action]

        self.campus_state.update_with_action(c_list_action)
        # Obtain observation, reward, and check if the episode is done
        # observation = np.array(convert_actions_to_discrete(self.campus_state.get_student_status()))
        observation = self.campus_state.get_student_status()
        reward = self.campus_state.get_reward(alpha)
        done = self.campus_state.is_episode_done()
        # done = self.campus_state.current_time == self.campus_state.model.get_max_weeks()
        info = {
            "allowed": self.campus_state.allowed_students_per_course,
            "infected": self.campus_state.student_status,
            "community_risk": self.campus_state.community_risk,
            "reward": reward
        }

        return observation, reward, done, False, info

    def reset(self):
        """
        Reset the state of the environment to an initial state.
        Returns:    observation (object): the initial observation.
        """
        state = self.campus_state.reset()
        logging.info(f"reset state: {state}")
        discrete_state = convert_actions_to_discrete(state)

        return np.array(state), {}


    def render(self, mode='bot'):
        """
        Render the environment's state.
        """
        weekly_infected_students = int(sum(self.campus_state.weekly_infected_students))/len(self.campus_state.weekly_infected_students)
        allowed_students_per_course = self.campus_state.allowed_students_per_course
        print("weekly_infected_students: ", weekly_infected_students, "allowed_students_per_course: ",
              allowed_students_per_course)

        return None
