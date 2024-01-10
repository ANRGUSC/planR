import time
import gymnasium as gym
import numpy as np
import tianshou as ts
import wandb
import tensorflow as tf
from .utilities import load_config

class DDPGAgent:
    def __init__(self, env, run_name, shared_config_path, agent_config_path=None, override_config=None):
        self.env = env
        self.run_name = run_name
        self.moving_average_window = 100
        self.shared_config_path = load_config(shared_config_path)
        
        if agent_config_path:
            self.agent_config = load_config(agent_config_path)
        else:
            self.agent_config = {}
        if override_config:
            self.agent_config.update(override_config)
        self.observation_dim = env.observation_space.shape
        self.action_dim = env.action_space.shape
        self.batch_size = 32
        self.observation_ph = tf.placeholder(tf.float32, shape=(None,) + self.observation_dim)
        self.action_ph = tf.placeholder(tf.float32, shape=(None,) + self.action_dim)

        def my_network():
            net = tf.layers.dense(self.observation_ph, 32, activation=tf.nn.relu)
            net = tf.layers.dense(net, 32, activation=tf.nn.relu)
            action = tf.layers.dense(net, self.action_dim[0], activation=None)

            action_value_input = tf.concat([self.observation_ph, self.action_ph], axis=1)
            net = tf.layers.dense(action_value_input, 64, activation=tf.nn.relu)
            net = tf.layers.dense(net, 64, activation=tf.nn.relu)
            action_value = tf.layers.dense(net, 1, activation=None)

            return action, action_value
        
        self.actor = ts.policy.Deterministic(my_network, observation_placeholder=self.observation_ph, has_old_net=True)
        self.critic = ts.policy.Deterministic(my_network, observation_placeholder=self.observation_ph, action_placeholder=self.action_ph, has_old_net=True)

        self.soft_update_op = ts.get_soft_update_op(1e-2, [self.actor, self.critic])

        self.critc_loss = ts.losses.value_mse(self.critic)
        self.critic_optimizer = tf.train.AdamOptimizer(1e-3)
        self.critic_train_op = self.critic_optimizer.minimize(self.critc_loss, var_list=list(self.critic.trainable_variables))

        self.dpg_grade_vars = ts.opt.DPG(self.actor, self.critic)
        self.actor_optimizer = tf.train.AdamOptimizer(1e-3)
        self.actor_train_op = self.actor_optimizer.apply_gradients(self.dpg_grade_vars)

        self.data_buffer = ts.data.VanillaReplayBuffer(10000, step=1)

        self.process_functions = [ts.data.advantage_estimation.ddpg_return(self.actor,self.critic)]

        self.data_collector = ts.data.DataCollector(
            env= self.env,
            policy=self.actor,
            data_buffer=self.data_buffer,
            process_functions=self.process_functions,
            managed_networks=[self.actor, self.critic]
        )

    def train(self, alpha):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            rewards_per_episode = []
            sess.run(tf.global_variables_initializer())
            
            self.actor.sync_weights()
            self.critic.sync_weights()

            start_time = time.time()
            self.data_collector.collect(num_timesteps=5000)
            for i in range(int(1e8)):
                self.data_collector.collect(num_timesteps=1, episode_cutoff=200)
                rewards_per_episode.append(self.data_collector.get_statistics()['episode_reward'])
                if i >= self.moving_average_window -1:
                    window_rewards = rewards_per_episode[max(0, i - self.moving_average_window + 1):i + 1]
                    moving_avg = np.mean(window_rewards)
                    std_dev = np.std(window_rewards)
                    wandb.log({
                        'Moving Average': moving_avg,
                        'Standard Deviation': std_dev,
                        'average_return': self.data_collector.get_statistics()['episode_reward'],
                        'step': i
                    })
                feed_dict = self.data_collector.next_batch(self.batch_size)
                sess.run(self.critic_train_op, feed_dict=feed_dict)

                self.data_collector.denoise_action(feed_dict)

                sess.run(self.actor_train_op, feed_dict=feed_dict)

                sess.run(self.soft_update_op)

                if i % 1000 == 0:
                    print('Step {}, elapsed time: {}'.format(i, time.time() - start_time))
                    ts.data.test_policy_in_env(self.actor,self.env, num_epsisodes=5, episode_cutoff=200)

    def test(self, episodes):
        pass