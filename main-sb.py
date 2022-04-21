# from campus_digital_twin import campus_state
# cs = campus_state.CampusState()
# #cs.current_time = 0
#
# while cs.current_time < 15:
#
#     action = [50] # choices are 0, 50, 100
#     cs.update_with_action(action)
#     print(cs.get_student_status(), cs.current_time)
#     #cs.current_time = cs.current_time + 1



import gym
import campus_gym
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
import wandb
from wandb.integration.sb3 import WandbCallback
wandb.init(project="campus-plan", entity="leezo", sync_tensorboard=True)

env = gym.make('CampusGymEnv-v0')
model = A2C('MlpPolicy', env, verbose=1, tensorboard_log="./a2c_campus_tensorboard/")
model.learn(total_timesteps=1000, callback=WandbCallback(gradient_save_freq=100, verbose=2))


# obs = env.reset()
# for i in range(15):
#     action, _state = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#     env.render()
#     if done:
#       obs = env.reset()