import gym
from gym.envs.registration import register
import sys
sys.path.append('../../..')
sys.path.append('../../../campus_digital_twin')

register(
    id='campus-v0',
    entry_point='campus_gym_env:CampusGymEnv',
)

env = gym.make('campus-v0') # try for different environements
observation = env.reset()
print("state at the start")
env.render()
print("---")

for t in range(3):
        print("time step: "+str(t))
#        print observation
        action = env.action_space.sample()
        print("action: "+str(action))
        observation, reward, done, info = env.step(action)
        print("observation: "+str(observation.nvec))
        print("reward: "+str(reward))
        print("current state:")
        env.render()
        print("---")

        if done:
            print("Finished after {} timesteps".format(t+1))
            break