import os
import gym
import numpy as np
import campus_gym
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
# import wandb
# from wandb.integration.sb3 import WandbCallback
# wandb.init(project="campus-plan", entity="leezo", sync_tensorboard=True)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # print(self.training_env.get_attr('reward')[0])
        self.logger.record('reward', self.training_env.get_attr('reward')[0])

        return True
def evaluate(model, num_episodes=100, deterministic=True):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_episodes: (int) number of episodes to evaluate it
    :return: (float) Mean reward for the last num_episodes
    """
    # This function will only work for a single Environment
    env = model.get_env()
    all_episode_rewards = []
    for i in range(num_episodes):
        episode_rewards = []
        done = False
        obs = env.reset()
        while not done:
            # _states are only useful when using LSTM policies
            action, _states = model.predict(obs, deterministic=deterministic)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            obs, reward, done, info = env.step(action)
            episode_rewards.append(reward)

        all_episode_rewards.append(sum(episode_rewards))

    mean_episode_reward = np.mean(all_episode_rewards)
    print("Mean reward:", mean_episode_reward, "Num episodes:", num_episodes)

    return mean_episode_reward
env = gym.make('CampusGymEnv-v0')
# eval_callback = EvalCallback(eval_env, best_model_save_path="./logs/",
#                              log_path="./logs/", eval_freq=2000,
#                              deterministic=True, render=False)
model = A2C('MlpPolicy', env, verbose=1, tensorboard_log="./a2c_campus_tensorboard/")
model.learn(total_timesteps=3000, callback=TensorboardCallback())



obs = env.reset()
for i in range(15):
    action, _state = model.predict(obs, deterministic=True)
    print(f'action: {action}')
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()
