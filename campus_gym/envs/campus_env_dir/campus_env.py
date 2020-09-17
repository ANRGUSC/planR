import gym
class CampusEnv(gym.Env):
    def __init__(self):
        print('Environment initialized')
    def step(self):
        print('Step successful')
    def reset(self):
        pring('Environment reset')