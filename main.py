import gym

from gym.envs.registration import register
import sys
sys.path.append('campus_gym/campus_gym/envs')
register(
    id='campus-v0',
    entry_point='campus_gym_env:CampusGymEnv',
)

env = gym.make('campus-v0')

# Generate simulation parameters
# Generate csv files from the simulation parameters
#






