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
# initialize Q-table with rows [total state_apaces (total_observation_space)] and  columns [total_action_space]
print(env.observation_space)
print(env.action_space)
for t in range(15):
        print("time step: "+str(t))
#        print observation
        action = env.action_space.sample()
        print("action: "+str(action))
        observation, reward, done, info = env.step(action)
        print("observation: "+str(observation))
        print("reward: "+str(reward))
        print("current state:")
        env.render()
        print("---")

        if done:
            print("Finished after {} timesteps".format(t+1))
            break