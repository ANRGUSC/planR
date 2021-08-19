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
import gym
from campus_digital_twin import campus_state as cs


class CampusGymEnv(gym.Env):
    """
    Description:
        At a given step (week) the agent makes a decision on
        how many students to allow for a given course.
    Observation:
        Type: Multidiscrete(n+1) where n is the number of courses
        and the last item is the community risk value.

        Example observation: [20, 34, 20, 0.5]
    Actions:
        Type: Multidiscrete(n) where n is the number of courses.
    Reward:
        Reward is returned from the campus environment
        as a scalar value.A high reward corresponds
        to an increase in the number of allowed students.
    Starting State:
        All observations are obtained
        from a static information provided by a campus model.

    Episode Termination:
        The campus environment stops running
        after n steps where n represents
        the duration of campus operation.
    """
    metadata = {'render.modes': ['bot']}

    def __init__(self):
        self.csobject = cs.CampusState()
        num_classes = self.csobject.model.total_courses()
        num_infec_levels = 3
        num_occup_levels = 3
        self.action_space = gym.spaces.MultiDiscrete\
            ([num_occup_levels for _ in range(num_classes)])
        self.observation_space = gym.spaces.MultiDiscrete\
            ([num_infec_levels for _ in range(num_classes + 1)])
        self.state = self.csobject.get_observation()
        print("Initial State", self.state)

    def step(self, action):
        """Take action.
        Args:
            action: Type (list)
        Returns:
            observation: Type(list)
            reward: Type(int)
            done: Type(bool)
        """
        self.csobject.update_with_action(action)
        observation = self.csobject.get_state()
        reward = self.csobject.get_reward()
        done = False
        if self.csobject.current_time == self.csobject.model.get_max_weeks():
            done = True
            self.reset()
        info = {}

        return observation, reward, done, info

    def reset(self):
        """Reset the current time.
        Returns:
            state: Type(list)
        """
        self.csobject.current_time = 0
        return self.csobject.get_state()

    def render(self, mode='bot'):
        """Render the environment.
        Returns:
            state: Type(list)
        """
        return self.csobject.get_state()
