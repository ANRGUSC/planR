import gym
from gym.envs.registration import register
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN

register(
        id='campus-v0',
        entry_point='campus_gym_env:CampusGymEnv',
    )
env = gym.make('campus-v0')

model = DQN(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=25)
model.save("deepq_cartpole")

# del model # remove to demonstrate saving and loading
#
# model = DQN.load("deepq_cartpole")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()