import gym
from gym import error, spaces, utils
from gym.utils import seeding
from campus_digital_twin import observations, simulation_engine, scheduler
class CampusGymEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.action_space = gym.spaces.Dict(scheduler.CourseRoomScheduler().get_schedule())
        self.observation_space = gym.spaces.Dict(observations.observations)
    def step(self):
        state = self.observation_space
        action = self.action_space
        """
        This is an example:
        - 
        """
        if action == [10, 20, 30, 40]:
            reward = 1
        else:
            reward = -1
        done = True
        info = {}

        return state, reward, done, info

    def reset(self):
        state = simulation_engine.create_campus_state()
        return state

    # def render(self, mode='human'):
    #   return
    # def close(self):
    #   return