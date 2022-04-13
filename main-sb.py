# from campus_digital_twin import campus_state
# cs = campus_state.CampusState()
# cs.current_time = 0
#
# while(cs.current_time < 15):
#     cs.current_time = cs.current_time + 1
#     print( cs.get_student_status(), cs.current_time)
#     action = [50]
#     cs.update_with_action(action)

    # count += 1
    # if (cs.current_time == 15):
    #     break

#for i in range(len(cs.weeks)):





import gym
import campus_gym
from stable_baselines3 import A2C
env = gym.make('CampusGymEnv-v0')
print(env)
model = A2C('MlpPolicy', env, verbose=1, tensorboard_log="./a2c_campus_tensorboard/")
model.learn(total_timesteps=1000)

obs = env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()