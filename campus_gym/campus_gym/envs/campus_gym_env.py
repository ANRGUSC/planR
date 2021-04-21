import gym
from gym import error, spaces, utils
from gym.utils import seeding
from campus_digital_twin import simulation_engine as sim
from campus_digital_twin import campus_state as cs
import numpy as np
#from campus_digital_twin import simulation_engine as se
# from campus_digital_twin import observations, simulation_engine, scheduler
class CampusGymEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.csobject = cs.CampusState()
        #self.csobject = sim.create_campus_state()
        num_classes = self.csobject.model.total_courses()
        num_infec_levels = 3
        num_occup_levels = 3
        self.action_space = gym.spaces.MultiDiscrete([num_occup_levels for _ in range(num_classes)])
        self.observation_space = gym.spaces.MultiDiscrete([num_infec_levels for _ in range(num_classes + 1)])
        self.state = self.csobject.get_observation()
        print("Initial State", self.state)

    def step(self, action):

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
        self.csobject.current_time = 0
        return self.csobject.get_state()

    def render(self, mode='bot', close=False):
        return self.csobject.get_state()

