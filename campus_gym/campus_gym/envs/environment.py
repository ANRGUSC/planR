import gym
from gym.envs.registration import register
import sys
sys.path.append('../../..')
sys.path.append('../../../campus_digital_twin')


class Environment():
    def __init__(self):
        pass

    def CampusGymStudentInfection(self):
        register(
            id='campus-v0',
            entry_point='campus_gym_env:CampusGymEnv',
        )
        env = gym.make('campus-v0')  # try for different environements
        return env



