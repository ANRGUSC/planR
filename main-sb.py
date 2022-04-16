"""
Uncomment for running the environment with dummy data. No gym implementation.
To change from one action to many actions one has to modify the campus data generator to match a preferred data structure.
In this case, it is a 1 classroom, 1 course, 100 total students, 20 initially infected students model.
"""
from campus_digital_twin import campus_state
cs = campus_state.CampusState()
cs.current_time = 0

while cs.current_time < 15:

    action = [50] # choices are 0, 50, 100
    cs.update_with_action(action)

    print(cs.get_student_status(), cs.current_time)
    cs.current_time = cs.current_time + 1



# import gym
# import campus_gym
# from stable_baselines3 import A2C
# env = gym.make('CampusGymEnv-v0')
# model = A2C('MlpPolicy', env, verbose=1, tensorboard_log="./a2c_campus_tensorboard/")
# model.learn(total_timesteps=1000)
#
# obs = env.reset()
# for i in range(100):
#     action, _state = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#     env.render()
#     if done:
#       obs = env.reset()