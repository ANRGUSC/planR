import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from datetime import datetime
import logging
from collections import deque
import random
import itertools
from tqdm import tqdm
from .utilities import load_config
from .visualizer import visualize_all_states, visualize_q_table, visualize_variance_in_rewards_heatmap, \
    visualize_explained_variance, visualize_variance_in_rewards, visualize_infected_vs_community_risk_table, \
    states_visited_viz
import wandb
from torch.optim.lr_scheduler import StepLR
import math
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import torch.nn.functional as F

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Set seed for reproducibility
set_seed(100)  # Replace 42 with your desired seed value



class ExplorationRateDecay:
    def __init__(self, max_episodes, min_exploration_rate, initial_exploration_rate):
        self.max_episodes = max_episodes
        self.min_exploration_rate = min_exploration_rate
        self.initial_exploration_rate = initial_exploration_rate
        self.current_decay_function = 1  # Variable to switch between different decay functions

    def set_decay_function(self, decay_function_number):
        self.current_decay_function = decay_function_number

    def get_exploration_rate(self, episode):
        if self.current_decay_function == 1:  # Exponential Decay
            exploration_rate = self.initial_exploration_rate * np.exp(-episode / self.max_episodes)

        elif self.current_decay_function == 2:  # Linear Decay
            exploration_rate = self.initial_exploration_rate - (
                        self.initial_exploration_rate - self.min_exploration_rate) * (episode / self.max_episodes)

        elif self.current_decay_function == 3:  # Polynomial Decay
            exploration_rate = self.initial_exploration_rate * (1 - episode / self.max_episodes) ** 2

        elif self.current_decay_function == 4:  # Inverse Time Decay
            exploration_rate = self.initial_exploration_rate / (1 + episode)

        elif self.current_decay_function == 5:  # Sine Wave Decay
            exploration_rate = self.min_exploration_rate + 0.5 * (
                        self.initial_exploration_rate - self.min_exploration_rate) * (
                                           1 + np.sin(np.pi * episode / self.max_episodes))

        elif self.current_decay_function == 6:  # Logarithmic Decay
            exploration_rate = self.initial_exploration_rate - (
                        self.initial_exploration_rate - self.min_exploration_rate) * np.log(episode + 1) / np.log(
                self.max_episodes + 1)

        elif self.current_decay_function == 7:  # Hyperbolic Tangent Decay
            exploration_rate = self.min_exploration_rate + 0.5 * (
                        self.initial_exploration_rate - self.min_exploration_rate) * (
                                           1 - np.tanh(episode / self.max_episodes))
        elif self.current_decay_function == 8:  # Square Root Decay
            exploration_rate = self.initial_exploration_rate * (1 - np.sqrt(episode / self.max_episodes))
        elif self.current_decay_function == 9:  # Stepwise Decay
            steps = 10
            step_size = (self.initial_exploration_rate - self.min_exploration_rate) / steps
            exploration_rate = self.initial_exploration_rate - (episode // (self.max_episodes // steps)) * step_size
        elif self.current_decay_function == 10:  # Inverse Square Root Decay
            exploration_rate = self.initial_exploration_rate / np.sqrt(episode + 1)
        elif self.current_decay_function == 11:  # Sigmoid Decay
            # exploration_rate = self.min_exploration_rate + (
            #             self.initial_exploration_rate - self.min_exploration_rate) / (
            #                                1 + np.exp(episode - self.max_episodes / 2))
            midpoint = self.max_episodes / 2
            smoothness = self.max_episodes / 10  # Adjust this divisor to change smoothness
            exploration_rate = self.min_exploration_rate + (
                    self.initial_exploration_rate - self.min_exploration_rate) / (
                                       1 + np.exp((episode - midpoint) / smoothness))

        elif self.current_decay_function == 12:  # Quadratic Decay
            exploration_rate = self.initial_exploration_rate * (1 - (episode / self.max_episodes) ** 2)
        elif self.current_decay_function == 13:  # Cubic Decay
            exploration_rate = self.initial_exploration_rate * (1 - (episode / self.max_episodes) ** 3)
        elif self.current_decay_function == 14:  # Sine Squared Decay
            exploration_rate = self.min_exploration_rate + (
                        self.initial_exploration_rate - self.min_exploration_rate) * np.sin(
                np.pi * episode / self.max_episodes) ** 2
        elif self.current_decay_function == 15:  # Cosine Squared Decay
            exploration_rate = self.min_exploration_rate + (
                        self.initial_exploration_rate - self.min_exploration_rate) * np.cos(
                np.pi * episode / self.max_episodes) ** 2
        elif self.current_decay_function == 16:  # Double Exponential Decay
            exploration_rate = self.initial_exploration_rate * np.exp(-np.exp(episode / self.max_episodes))
        elif self.current_decay_function == 17:  # Log-Logistic Decay
            exploration_rate = self.min_exploration_rate + (
                        self.initial_exploration_rate - self.min_exploration_rate) / (1 + np.log(episode + 1))
        elif self.current_decay_function == 18:  # Harmonic Series Decay
            exploration_rate = self.min_exploration_rate + (
                        self.initial_exploration_rate - self.min_exploration_rate) / (
                                           1 + np.sum(1 / np.arange(1, episode + 2)))
        elif self.current_decay_function == 19:  # Piecewise Linear Decay
            if episode < self.max_episodes / 2:
                exploration_rate = self.initial_exploration_rate - (
                            self.initial_exploration_rate - self.min_exploration_rate) * (
                                               2 * episode / self.max_episodes)
            else:
                exploration_rate = self.min_exploration_rate
        elif self.current_decay_function == 20:  # Custom Polynomial Decay
            p = 3  # Change the power for different polynomial behaviors
            exploration_rate = self.initial_exploration_rate * (1 - (episode / self.max_episodes) ** p)
        else:
            raise ValueError("Invalid decay function number")

        return exploration_rate


class DeepQNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super(DeepQNetwork, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.out = nn.Linear(hidden_dim, out_dim)

        # Add this line to initialize weights to zero
        self.apply(self._initialize_weights)  # <- This line

    def forward(self, x):
        h_prime = self.encoder(x)
        Q_values = self.out(h_prime)
        return Q_values

    def _initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
class DQNCustomAgent:
    def __init__(self, env, run_name, shared_config_path, agent_config_path=None, override_config=None):
        # Load Shared Config
        self.shared_config = load_config(shared_config_path)

        # Load Agent Specific Config if path provided
        if agent_config_path:
            self.agent_config = load_config(agent_config_path)
        else:
            self.agent_config = {}

        # If override_config is provided, merge it with the loaded agent_config
        if override_config:
            self.agent_config.update(override_config)

        # Access the results directory from the shared_config
        self.results_directory = self.shared_config['directories']['results_directory']

        # Create a unique subdirectory for each run to avoid overwriting results
        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.agent_type = "dqn_custom"
        self.run_name = run_name
        self.results_subdirectory = os.path.join(self.results_directory, self.agent_type, self.run_name, self.timestamp)
        if not os.path.exists(self.results_subdirectory):
            os.makedirs(self.results_subdirectory, exist_ok=True)
        self.model_directory = self.shared_config['directories']['model_directory']
        self.model_subdirectory = os.path.join(self.model_directory, self.agent_type, self.run_name, self.timestamp)
        if not os.path.exists(self.model_subdirectory):
            os.makedirs(self.model_subdirectory, exist_ok=True)

        # Set up logging to the correct directory
        log_file_path = os.path.join(self.results_subdirectory, 'agent_log.txt')
        logging.basicConfig(filename=log_file_path, level=logging.INFO)

        # Initialize wandb
        wandb.init(project=self.agent_type, name=self.run_name)

        # Initialize the neural network
        self.input_dim = len(env.reset()[0])
        self.output_dim = env.action_space.nvec[0]
        self.hidden_dim = self.agent_config['agent']['hidden_units']
        self.model = DeepQNetwork(self.input_dim, self.hidden_dim, self.output_dim).float()
        self.target_model = DeepQNetwork(self.input_dim, self.hidden_dim, self.output_dim).float()
        self.target_model.load_state_dict(self.model.state_dict())

        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.agent_config['agent']['learning_rate'])


        # Initialize agent-specific configurations and variables
        self.env = env
        self.max_episodes = self.agent_config['agent']['max_episodes']
        self.discount_factor = self.agent_config['agent']['discount_factor']
        self.exploration_rate = self.agent_config['agent']['exploration_rate']
        self.min_exploration_rate = self.agent_config['agent']['min_exploration_rate']
        self.exploration_decay_rate = self.agent_config['agent']['exploration_decay_rate']
        self.target_network_frequency = self.agent_config['agent']['target_network_frequency']

        # Replay memory
        self.replay_memory = deque(maxlen=self.agent_config['agent']['replay_memory_capacity'])
        self.batch_size = self.agent_config['agent']['batch_size']

        self.possible_actions = [list(range(0, (k))) for k in self.env.action_space.nvec]
        self.all_actions = [str(i) for i in list(itertools.product(*self.possible_actions))]

        # moving average for early stopping criteria
        self.moving_average_window = 100  # Number of episodes to consider for moving average
        self.stopping_criterion = 0.01  # Threshold for stopping
        self.prev_moving_avg = -float(
            'inf')  # Initialize to negative infinity to ensure any reward is considered an improvement in the first episode.

        # Hidden State
        self.hidden_state = None
        self.reward_window = deque(maxlen=self.moving_average_window)
        # Initialize the learning rate scheduler
        self.scheduler = StepLR(self.optimizer, step_size=100, gamma=0.9)
        self.learning_rate_decay = self.agent_config['agent']['learning_rate_decay']

        self.softmax_temperature = self.agent_config['agent']['softmax_temperature']

        self.state_visit_counts = {}
        self.noise_scale = 0.1
        self.noise_decay_rate = 0.9999
        self.noise_frequency = 1000
        self.aggressiveness = 0.8

        self.decay_handler = ExplorationRateDecay(self.max_episodes, self.min_exploration_rate, self.exploration_rate)
        self.decay_function = self.agent_config['agent']['e_decay_function']

    def update_replay_memory(self, state, action, reward, next_state, done):
        state = np.array(state, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        action = int(action)
        reward = float(reward)
        done = bool(done)
        self.replay_memory.append((state, action, reward, next_state, done))

    def compute_q_value(self, states, actions):
        q_values = self.model(states)
        return q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    def loss_function(self, action_probabilities, rewards):
        discounted_rewards = torch.zeros_like(rewards)
        discounted_rewards[-1] = rewards[-1]
        for i in range(len(rewards) - 1):
            curr = discounted_rewards[-(i + 1)]
            discounted_rewards[-(i + 2)] = rewards[-(i + 2)] + curr * self.discount_factor
        return (-torch.log(action_probabilities) * discounted_rewards).sum()

    def compute_intrinsic_reward(self, state):
        state_tuple = tuple(state.tolist())
        if state_tuple not in self.state_visit_counts:
            self.state_visit_counts[state_tuple] = 0
        self.state_visit_counts[state_tuple] += 1
        return np.exp(-self.aggressiveness * self.state_visit_counts[state_tuple])

    def boltzmann_policy(self, Q_values, temperature):
        max_Q = torch.max(Q_values)
        exp_Q = torch.exp((Q_values - max_Q) / temperature)
        prob_dist = exp_Q / torch.sum(exp_Q)
        return prob_dist

    def epsilon_greedy_policy(self, Q_values, epsilon):
        num_actions = Q_values.size(-1)  # Get the number of actions from Q_values

        if self.detect_nan(Q_values, "Q_values"):
            print(f"NaN Q-values detected. Returning uniform distribution.")
            return torch.ones(num_actions, dtype=torch.float32) / num_actions

        if np.random.random() < epsilon:
            # Explore: equal probability for all actions
            prob_dist = torch.ones(num_actions, dtype=torch.float32) / num_actions
            # print(f"Exploring: epsilon = {epsilon}")
        else:
            # Exploit: return softmax of Q-values
            prob_dist = F.softmax(Q_values.squeeze(), dim=-1)
            # print(f"Exploiting: epsilon = {epsilon}, Q_values = {Q_values}")

        return prob_dist
    def soft_update(self, target_model, online_model, tau):
        for target_param, online_param in zip(target_model.parameters(), online_model.parameters()):
            target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)

    def detect_nan(self, tensor, tensor_name):
        if torch.isnan(tensor).any():
            print(f"NaN detected in {tensor_name}")
            return True
        return False

    def train(self, alpha):
        pbar = tqdm(total=self.max_episodes, desc="Training Progress", leave=True)

        # Initialize accumulators for visualization
        actual_rewards = []
        predicted_rewards = []
        rewards_per_episode = []
        visited_state_counts = {}
        explained_variance_per_episode = []

        for episode in range(self.max_episodes):
            self.decay_handler.set_decay_function(self.decay_function)
            # state, _ = self.env.reset()
            # state = torch.FloatTensor(state).float
            terminated = False
            action_probabilities = []
            rewards = []
            visited_states = []  # List to store visited states
            loss = torch.tensor(0.0)
            predicted_rewards_episode = []
            current_state, _ = self.env.reset()
            current_state = torch.tensor(current_state, dtype=torch.float32).unsqueeze(0)

            while not terminated:
                current_state = (current_state - current_state.mean()) / (current_state.std() + 1e-8)
                Q_values = self.model(current_state)

                if self.detect_nan(Q_values, "Q_values"):
                    print(f"State that caused NaN: {current_state}")
                    print(f"Model parameters: {list(self.model.parameters())}")
                    break  # Exit the episode if NaN is detected

                prob_dist = self.epsilon_greedy_policy(Q_values, self.exploration_rate)
                action = prob_dist.argmax().item()
                # print(f"State: {current_state}, Action: {action}, prob_dist: {prob_dist}")

                next_state, reward, terminated, _, info = self.env.step(([action * 50], alpha))
                next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

                # print(f"Reward: {reward}")
                rewards.append(reward)
                predicted_rewards_episode.append(prob_dist[action].item())

                state_tuple = tuple(current_state.squeeze().tolist())
                if state_tuple not in self.state_visit_counts:
                    self.state_visit_counts[state_tuple] = 0
                self.state_visit_counts[state_tuple] += 1

                visited_states.append(state_tuple)

                self.update_replay_memory(current_state.squeeze().numpy(), action, reward, next_state.squeeze().numpy(),
                                          terminated)

                current_state = next_state

            # Training step
            if len(self.replay_memory) >= self.batch_size:
                minibatch = random.sample(self.replay_memory, self.batch_size)
                states, actions, rewards_batch, next_states, dones = map(np.array, zip(*minibatch))
                states = torch.FloatTensor(states).float()
                actions = torch.LongTensor(actions)
                rewards_batch = torch.FloatTensor(rewards_batch).float()
                next_states = torch.FloatTensor(next_states).float()
                dones = torch.FloatTensor(dones).float()

                curr_Q = self.compute_q_value(states, actions)
                with torch.no_grad():
                    next_Q_values = self.model(next_states)
                    next_actions = next_Q_values.argmax(1)
                    next_Q_values_target = self.target_model(next_states)
                    next_Q = next_Q_values_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
                expected_Q = rewards_batch + self.discount_factor * next_Q * (1 - dones)

                # entropy = -torch.sum(prob_dist * torch.log(prob_dist + 0.1))

                # Add entropy term to the loss
                # loss = nn.MSELoss()(curr_Q, expected_Q) - 0.01 * entropy
                loss = F.smooth_l1_loss(curr_Q, expected_Q)


                self.optimizer.zero_grad()
                loss.backward()

                # Add gradient clipping here
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()
                # self.scheduler.step()

                # Calculate TD error
                td_error = torch.abs(curr_Q - expected_Q).mean().item()

                # Log metrics to wandb
                mean_q_value = curr_Q.mean().item()
                wandb.log({
                    "loss": loss.item(),
                    "learning_rate": self.optimizer.param_groups[0]['lr'],
                    "exploration_rate": self.exploration_rate,
                    "mean_q_value": mean_q_value,
                    "td_error": td_error,
                    "avg_reward": torch.tensor(rewards, dtype=torch.float32).mean().item(),
                    "total_reward": torch.tensor(rewards, dtype=torch.float32).sum().item(),
                })

            # Update accumulators for visualization
            actual_rewards.append(rewards)
            predicted_rewards.append([prob.item() for prob in action_probabilities])
            rewards_per_episode.append(torch.tensor(rewards, dtype=torch.float32).mean().item())

            # Update the state visit counts
            for state in visited_states:
                state_str = str(state)  # Convert the state to a string to use it as a dictionary key
                if state_str in visited_state_counts:
                    visited_state_counts[state_str] += 1
                else:
                    visited_state_counts[state_str] = 1

            # Append the episode reward to the reward window for moving average calculation
            self.reward_window.append(sum(rewards))
            moving_avg_reward = np.mean(self.reward_window)

            # Calculate explained variance for the current episode
            explained_variance = calculate_explained_variance(rewards, predicted_rewards_episode)
            explained_variance_per_episode.append(explained_variance)


            # if episode % 100 == 0:
            self.exploration_rate = self.decay_handler.get_exploration_rate(episode)

            # Perform soft update
            self.soft_update(self.target_model, self.model, tau=0.01)

            pbar.update(1)
            pbar.set_description(
                f"Loss {float(loss.item())} Avg R {float(torch.tensor(rewards, dtype=torch.float32).mean().numpy())}")

        pbar.close()

        # After training, save the model
        model_file_path = os.path.join(self.model_subdirectory, 'model.pt')
        torch.save(self.model.state_dict(), model_file_path)

        # Visualization and logging
        saved_model = load_saved_model(self.model_directory, self.agent_type, self.run_name, self.timestamp,
                                       self.input_dim, self.hidden_dim, self.output_dim)
        value_range = range(0, 101, 10)
        all_states = [np.array([i, j]) for i in value_range for j in value_range]
        all_states_path = visualize_all_states(saved_model, all_states, self.run_name, self.max_episodes, alpha,
                                               self.results_subdirectory)
        wandb.log({"All_States_Visualization": [wandb.Image(all_states_path)]})


        states = list(visited_state_counts.keys())
        visit_counts = list(visited_state_counts.values())
        states_visited_path = states_visited_viz(states, visit_counts, alpha, self.results_subdirectory)
        wandb.log({"States Visited": [wandb.Image(states_visited_path)]})

        avg_rewards = [sum(lst) / len(lst) for lst in actual_rewards]
        explained_variance_path = os.path.join(self.results_subdirectory, 'explained_variance.png')
        visualize_explained_variance(explained_variance_per_episode, explained_variance_path)
        wandb.log({"Explained Variance": [wandb.Image(explained_variance_path)]})

        #
        # file_path_variance = visualize_variance_in_rewards(avg_rewards, self.results_subdirectory, self.max_episodes)
        # wandb.log({"Variance in Rewards": [wandb.Image(file_path_variance)]})


        return self.model

    def train_single_run(self, seed, alpha):
        set_seed(seed)
        # Reset relevant variables for each run
        self.replay_memory = deque(maxlen=self.agent_config['agent']['replay_memory_capacity'])
        self.reward_window = deque(maxlen=self.moving_average_window)
        self.model = DeepQNetwork(self.input_dim, self.hidden_dim, self.output_dim)
        self.target_model = DeepQNetwork(self.input_dim, self.hidden_dim, self.output_dim)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.agent_config['agent']['learning_rate'])
        self.scheduler = StepLR(self.optimizer, step_size=100, gamma=self.learning_rate_decay)

        self.run_rewards_per_episode = []  # Store rewards per episode for this run

        pbar = tqdm(total=self.max_episodes, desc=f"Training Run {seed}", leave=True)
        decay_rate = np.log(self.min_exploration_rate / self.exploration_rate) / self.max_episodes

        for episode in range(self.max_episodes):
            state, _ = self.env.reset()
            state = torch.FloatTensor(state)
            terminated = False

            hidden_state = torch.zeros(self.hidden_dim)
            rewards = []
            loss = torch.tensor(0.0)  # Initialize loss here

            while not terminated:
                Q_values = self.model(state)
                prob_dist = torch.nn.functional.softmax(Q_values, dim=-1)

                if np.random.random() > self.exploration_rate:
                    action = prob_dist.argmax()
                else:
                    action = np.random.choice(len(prob_dist), p=prob_dist.detach().numpy())

                next_state, reward, terminated, _, info = self.env.step(([action * 50], alpha))
                # Compute novelty reward
                state_tuple = tuple(state.tolist())
                if state_tuple not in self.state_visit_counts:
                    self.state_visit_counts[state_tuple] = 0
                novelty_reward = 1.0 / (1 + self.state_visit_counts[state_tuple])
                self.state_visit_counts[state_tuple] += 1

                # Add novelty reward to the actual reward
                total_reward = reward + novelty_reward
                rewards.append(total_reward)

                self.update_replay_memory(state, action, total_reward, next_state, terminated)
                state = torch.FloatTensor(next_state)

            # Training step
            if len(self.replay_memory) >= self.batch_size:
                minibatch = random.sample(self.replay_memory, self.batch_size)
                states, actions, rewards_batch, next_states, dones = map(np.array, zip(*minibatch))
                states = torch.FloatTensor(states)
                actions = torch.LongTensor(actions)
                rewards_batch = torch.FloatTensor(rewards_batch)
                next_states = torch.FloatTensor(next_states)
                dones = torch.FloatTensor(dones)

                curr_Q = self.compute_q_value(states, actions)
                with torch.no_grad():
                    next_Q_values = self.target_model(next_states)
                    next_Q = next_Q_values.max(1)[0]
                expected_Q = rewards_batch + self.discount_factor * next_Q * (1 - dones.unsqueeze(1))
                loss = nn.MSELoss()(curr_Q, expected_Q)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step()

                # Calculate TD error
                td_error = torch.abs(curr_Q - expected_Q).mean().item()

                # Log metrics to wandb
                mean_q_value = curr_Q.mean().item()
                # wandb.log({
                #     "loss": loss.item(),
                #     "learning_rate": self.optimizer.param_groups[0]['lr'],
                #     "exploration_rate": self.exploration_rate,
                #     "mean_q_value": mean_q_value,
                #     "td_error": td_error,
                #     "avg_reward": torch.tensor(rewards, dtype=torch.float32).mean().item(),
                #     "total_reward": torch.tensor(rewards, dtype=torch.float32).sum().item(),
                #     "episode": episode
                # })

            self.run_rewards_per_episode.append(rewards)  # Store raw rewards for this episode
            if episode % 100 == 0:
                self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate - (
                        1.0 - self.min_exploration_rate) / self.max_episodes)
            if episode % self.target_network_frequency == 0:
                self.target_model.load_state_dict(self.model.state_dict())

            pbar.update(1)
            pbar.set_description(f"Run {seed} - Loss {float(loss.item())} Avg Reward {float(np.mean(rewards))}")

        pbar.close()
        model_file_path = os.path.join(self.model_subdirectory, f'model_run_{seed}.pt')
        torch.save(self.model.state_dict(), model_file_path)
        return self.run_rewards_per_episode

    def compute_tolerance_interval(self, data, alpha, beta):
        """
        Compute the (alpha, beta)-tolerance interval for a given data sample.

        Parameters:
        data (list or numpy array): The data sample.
        alpha (float): The nominal error rate (e.g., 0.05 for 95% confidence level).
        beta (float): The proportion of future samples to be captured (e.g., 0.9 for 90% of the population).

        Returns:
        (float, float): The lower and upper bounds of the tolerance interval.
        """
        n = len(data)
        if n == 0:
            return np.nan, np.nan  # Handle case with no data

        sorted_data = np.sort(data)

        # Compute the number of samples that do not belong to the middle beta proportion
        nu = stats.binom.ppf(1 - alpha, n, beta)
        nu = int(nu)

        if nu >= n:
            return sorted_data[0], sorted_data[-1]  # If nu is greater than available data points, return full range

        # Compute the indices for the lower and upper bounds
        l = int(np.floor(nu / 2))
        u = int(np.ceil(n - nu / 2))

        return sorted_data[l], sorted_data[u]

    def visualize_tolerance_interval_curve(self, returns_per_episode, alpha, beta, output_path, metric='mean'):
        """
        Visualize the (alpha, beta)-tolerance interval curve over episodes for mean or median performance.

        Parameters:
        returns_per_episode (list): The list of returns per episode across multiple runs.
        alpha (float): The nominal error rate (e.g., 0.05 for 95% confidence level).
        beta (float): The proportion of future samples to be captured (e.g., 0.9 for 90% of the population).
        output_path (str): The file path to save the plot.
        metric (str): The metric to visualize ('mean' or 'median').
        """
        num_episodes = len(returns_per_episode[0])
        lower_bounds = []
        upper_bounds = []
        central_tendency = []
        episodes = list(range(num_episodes))  # Assume all runs have the same number of episodes)

        for episode in episodes:
            returns_at_episode = [returns[episode] for returns in
                                  returns_per_episode]  # Shape: (num_runs, episode_length)
            returns_at_episode = [item for sublist in returns_at_episode for item in sublist]  # Flatten to 1D

            if metric == 'mean':
                performance = np.mean(returns_at_episode)
            elif metric == 'median':
                performance = np.median(returns_at_episode)
            else:
                raise ValueError("Invalid metric specified. Use 'mean' or 'median'.")

            central_tendency.append(performance)
            lower, upper = self.compute_tolerance_interval(returns_at_episode, alpha, beta)
            lower_bounds.append(lower)
            upper_bounds.append(upper)

        lower_bounds = np.array(lower_bounds)
        upper_bounds = np.array(upper_bounds)
        central_tendency = np.array(central_tendency)

        # Smoothing the curve
        spline_points = 300  # Number of points for spline interpolation
        episodes_smooth = np.linspace(episodes[0], episodes[-1], spline_points)
        central_tendency_smooth = make_interp_spline(episodes, central_tendency)(episodes_smooth)
        lower_bounds_smooth = make_interp_spline(episodes, lower_bounds)(episodes_smooth)
        upper_bounds_smooth = make_interp_spline(episodes, upper_bounds)(episodes_smooth)

        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")

        # Plot central tendency
        sns.lineplot(x=episodes_smooth, y=central_tendency_smooth, color='blue',
                     label=f'{metric.capitalize()} Performance')

        # Fill between for tolerance interval
        plt.fill_between(episodes_smooth, lower_bounds_smooth, upper_bounds_smooth, color='lightblue', alpha=0.2,
                         label=f'Tolerance Interval (α={alpha}, β={beta})')

        plt.title(f'Tolerance Interval Curve for {metric.capitalize()} Performance')
        plt.xlabel('Episode')
        plt.ylabel('Return')
        plt.legend()
        plt.savefig(output_path)
        plt.close()

    def compute_confidence_interval(self, data, alpha):
        """
        Compute the confidence interval for a given data sample using the Student t-distribution.

        Parameters:
        data (list or numpy array): The data sample.
        alpha (float): The nominal error rate (e.g., 0.05 for 95% confidence interval).

        Returns:
        (float, float): The lower and upper bounds of the confidence interval.
        """
        n = len(data)
        mean = np.mean(data)
        std_err = np.std(data, ddof=1) / np.sqrt(n)
        t_value = stats.t.ppf(1 - alpha / 2, df=n - 1)
        margin_of_error = t_value * std_err
        return mean - margin_of_error, mean + margin_of_error

    def visualize_confidence_interval(self, returns, alpha, output_path):
        """
        Visualize the confidence interval over episodes.

        Parameters:
        returns (list): The list of returns per episode across multiple runs.
        alpha (float): The nominal error rate (e.g., 0.05 for 95% confidence interval).
        output_path (str): The file path to save the plot.
        """
        means = []
        lower_bounds = []
        upper_bounds = []
        episodes = list(range(len(returns[0])))  # Assume all runs have the same number of episodes

        for episode in episodes:
            episode_returns = [returns[run][episode] for run in range(len(returns))]
            mean = np.mean(episode_returns)
            lower, upper = self.compute_confidence_interval(episode_returns, alpha)
            means.append(mean)
            lower_bounds.append(lower)
            upper_bounds.append(upper)

        means = np.array(means)
        lower_bounds = np.array(lower_bounds)
        upper_bounds = np.array(upper_bounds)

        # Smoothing the curve
        spline_points = 300  # Number of points for spline interpolation
        episodes_smooth = np.linspace(episodes[0], episodes[-1], spline_points)
        means_smooth = make_interp_spline(episodes, means)(episodes_smooth)
        lower_bounds_smooth = make_interp_spline(episodes, lower_bounds)(episodes_smooth)
        upper_bounds_smooth = make_interp_spline(episodes, upper_bounds)(episodes_smooth)

        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")

        # Plot mean performance
        sns.lineplot(x=episodes_smooth, y=means_smooth, label='Mean Performance', color='blue')

        # Fill between for confidence interval
        plt.fill_between(episodes_smooth, lower_bounds_smooth, upper_bounds_smooth, color='lightblue', alpha=0.2,
                         label=f'Confidence Interval (α={alpha})')

        plt.title(f'Confidence Interval Curve for Mean Performance')
        plt.xlabel('Episode')
        plt.ylabel('Return')
        plt.legend()
        plt.savefig(output_path)
        plt.close()

    def visualize_boxplot_confidence_interval(self, returns, alpha, output_path):
        """
        Visualize the confidence interval using box plots.

        Parameters:
        returns (list): The list of returns per episode across multiple runs.
        alpha (float): The nominal error rate (e.g., 0.05 for 95% confidence interval).
        output_path (str): The file path to save the plot.
        """
        episodes = list(range(len(returns[0])))  # Assume all runs have the same number of episodes
        returns_transposed = np.array(returns).T.tolist()  # Transpose to get returns per episode

        plt.figure(figsize=(12, 8))
        sns.boxplot(data=returns_transposed, whis=[100 * alpha / 2, 100 * (1 - alpha / 2)], color='lightblue')
        plt.title(f'Box Plot of Returns with Confidence Interval (α={alpha})')
        plt.xlabel('Episode')
        plt.ylabel('Return')
        plt.xticks(ticks=range(len(episodes)), labels=episodes)
        plt.savefig(output_path)
        plt.close()

    def multiple_runs(self, num_runs, alpha_t, beta_t):
        returns_per_episode = []

        for run in range(num_runs):
            returns = self.train_single_run(run, alpha_t)
            returns_per_episode.append(returns)

        # Ensure returns_per_episode is correctly structured
        returns_per_episode = np.array(returns_per_episode)  # Shape: (num_runs, max_episodes, episode_length)

        output_path_mean = os.path.join(self.results_subdirectory, 'tolerance_interval_mean.png')
        output_path_median = os.path.join(self.results_subdirectory, 'tolerance_interval_median.png')

        self.visualize_tolerance_interval_curve(returns_per_episode, alpha_t, beta_t, output_path_mean, 'mean')
        self.visualize_tolerance_interval_curve(returns_per_episode, alpha_t, beta_t, output_path_median, 'median')

        wandb.log({"Tolerance Interval Mean": [wandb.Image(output_path_mean)]})
        wandb.log({"Tolerance Interval Median": [wandb.Image(output_path_median)]})

        # Confidence Intervals
        confidence_alpha = 0.05  # 95% confidence interval
        confidence_output_path = os.path.join(self.results_subdirectory, 'confidence_interval.png')
        self.visualize_confidence_interval(returns_per_episode, confidence_alpha, confidence_output_path)
        wandb.log({"Confidence Interval": [wandb.Image(confidence_output_path)]})

        # Box Plot Confidence Intervals
        boxplot_output_path = os.path.join(self.results_subdirectory, 'boxplot_confidence_interval.png')
        self.visualize_boxplot_confidence_interval(returns_per_episode, confidence_alpha, boxplot_output_path)
        wandb.log({"Box Plot Confidence Interval": [wandb.Image(boxplot_output_path)]})


def load_saved_model(model_directory, agent_type, run_name, timestamp, input_dim, hidden_dim, action_space_nvec):
    """
    Load a saved DeepQNetwork model from the subdirectory.

    Args:
    model_directory: Base directory where models are stored.
    agent_type: Type of the agent, used in directory naming.
    run_name: Name of the run, used in directory naming.
    timestamp: Timestamp of the model saving time, used in directory naming.
    input_dim: Input dimension of the model.
    hidden_dim: Hidden layer dimension.
    action_space_nvec: Action space vector size.

    Returns:
    model: The loaded DeepQNetwork model, or None if loading failed.
    """
    # Construct the model subdirectory path
    model_subdirectory = os.path.join(model_directory, agent_type, run_name, timestamp)

    # Construct the model file path
    model_file_path = os.path.join(model_subdirectory, 'model.pt')

    # Check if the model file exists
    if not os.path.exists(model_file_path):
        print(f"Model file not found in {model_file_path}")
        return None

    # Initialize a new model instance
    model = DeepQNetwork(input_dim, hidden_dim, action_space_nvec)

    # Load the saved model state into the model instance
    model.load_state_dict(torch.load(model_file_path))
    model.eval()  # Set the model to evaluation mode

    return model

def calculate_explained_variance(actual_rewards, predicted_rewards):
    actual_rewards = np.array(actual_rewards)
    predicted_rewards = np.array(predicted_rewards)
    variance_actual = np.var(actual_rewards, ddof=1)
    variance_unexplained = np.var(actual_rewards - predicted_rewards, ddof=1)
    explained_variance = 1 - (variance_unexplained / variance_actual)
    return explained_variance


def visualize_explained_variance(explained_variance_per_episode, output_path):
    plt.figure(figsize=(10, 6))
    plt.plot(explained_variance_per_episode, label='Explained Variance')
    plt.xlabel('Episode')
    plt.ylabel('Explained Variance')
    plt.title('Explained Variance over Episodes')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()
