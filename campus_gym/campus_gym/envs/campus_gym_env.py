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
        self.csobject = sim.create_campus_state()
        #print(self.csobject.model.number_of_students_per_course())

        num_classes = len(self.csobject.model.number_of_students_per_course()[0])

        print("num classes =  " + str(num_classes))
        num_infec_levels = 3
        num_occup_levels = 3
        #self.action_space = gym.spaces.Box(low=np.array([0 for _ in range(num_classes)]), high=np.array([100 for _ in range (num_classes)]), dtype=np.int)
        self.action_space = gym.spaces.MultiDiscrete([num_occup_levels for _ in range(num_classes)])
        self.observation_space = gym.spaces.MultiDiscrete([num_infec_levels for _ in range(num_classes + 1)])
        self.state = self.csobject.get_state()


    def step(self, action):

        self.csobject.update_with_action(action)
        observation = self.csobject.get_observation()
        reward = self.csobject.get_reward()
        done = False
        if self.csobject.current_time == self.csobject.model.get_max_weeks():
          done = True
        info = {}

        return observation, reward, done, info

    def reset(self):
        self.state = self.csobject.get_state()
        self.csobject.current_time = 0
        return self.csobject.get_observation()

    def render(self, mode='human', close=False):
        print("current time: " + str(self.csobject.current_time))
        print(self.csobject.get_state())
        return

