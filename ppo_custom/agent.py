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
from torch.distributions.categorical import Categorical
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # utils.seed_everything(seed)

# Set seed for reproducibility
set_seed(100)  # Replace 42 with your desired seed value


class PPOMemory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []

    def generate_batches(self, batch_size):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+batch_size] for i in batch_start]

        states = torch.stack(self.states).detach().cpu().numpy()
        actions = torch.stack(self.actions).detach().cpu().numpy()
        probs = torch.stack(self.probs).detach().cpu().numpy()
        vals = torch.stack(self.vals).detach().cpu().numpy()
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)

        return states, actions, probs, vals, rewards, dones, batches

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []


class ActorCriticNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ActorCriticNetwork, self).__init__()

        # Shared layers
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)
        )

        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        shared_output = self.shared_layers(x)
        action_probs = self.actor(shared_output)
        state_value = self.critic(shared_output)
        return action_probs, state_value


class PPOCustomAgent:
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
        self.agent_type = "ppo_custom"
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

        self.model = ActorCriticNetwork(self.input_dim, self.hidden_dim, self.output_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.agent_config['agent']['learning_rate'])

        # Initialize agent-specific configurations and variables
        self.env = env
        self.max_episodes = self.agent_config['agent']['max_episodes']
        self.discount_factor = self.agent_config['agent']['discount_factor']

        self.possible_actions = [list(range(0, (k))) for k in self.env.action_space.nvec]
        self.all_actions = [str(i) for i in list(itertools.product(*self.possible_actions))]

        self.state_visit_counts = {}

        # PPO specific parameters
        self.n_epochs = self.agent_config['agent']['n_epochs']
        self.clip_param = self.agent_config['agent']['clip_param']
        self.gae_lambda = self.agent_config['agent']['gae_lambda']
        self.batch_size = self.agent_config['agent']['batch_size']
        self.value_loss_coef = self.agent_config['agent']['value_loss_coef']
        self.entropy_coef = self.agent_config['agent']['entropy_coef']

        self.memory = PPOMemory()

    def compute_gae(self, rewards, values, dones, next_value):
        advantages = []
        gae = 0
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.discount_factor * next_value * (1 - dones[step]) - values[step]
            gae = delta + self.discount_factor * self.gae_lambda * (1 - dones[step]) * gae
            advantages.insert(0, gae)
            next_value = values[step]
        return advantages

    def get_final_performance(self):
        # Return the average reward of the last few episodes, or any other metric you prefer
        return np.mean(self.run_rewards_per_episode[-10:])

    def train(self, alpha):
        torch.autograd.set_detect_anomaly(True)

        pbar = tqdm(total=self.max_episodes, desc="Training Progress", leave=True)

        actual_rewards = []
        explained_variance_per_episode = []
        visited_state_counts = {}
        self.run_rewards_per_episode = []

        for episode in range(self.max_episodes):
            state, _ = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32)
            episode_reward = 0
            done = False

            while not done:
                action_probs, value = self.model(state)
                action_dist = Categorical(action_probs)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)

                next_state, reward, done, _, _ = self.env.step(([action.item() * 50], alpha))
                next_state = torch.tensor(next_state, dtype=torch.float32)

                state_tuple = tuple(state.detach().numpy())
                visited_state_counts[state_tuple] = visited_state_counts.get(state_tuple, 0) + 1

                self.memory.states.append(state)
                self.memory.actions.append(action)
                self.memory.probs.append(log_prob)
                self.memory.vals.append(value)
                self.memory.rewards.append(reward)
                self.memory.dones.append(done)

                state = next_state
                episode_reward += reward

            _, next_value = self.model(next_state)
            returns = self.compute_gae(self.memory.rewards, self.memory.vals, self.memory.dones, next_value)

            states, actions, old_probs, vals, rewards, dones, batches = \
                self.memory.generate_batches(self.batch_size)

            values = torch.tensor(vals)
            returns = torch.tensor(returns)
            advantages = torch.tensor(returns) - values

            # Initialize loss tracking variables
            epoch_actor_loss = 0
            epoch_critic_loss = 0
            epoch_entropy = 0
            epoch_total_loss = 0

            for _ in range(self.n_epochs):
                for batch in batches:
                    states_batch = torch.tensor(states[batch], dtype=torch.float)
                    old_probs_batch = torch.tensor(old_probs[batch])
                    actions_batch = torch.tensor(actions[batch])
                    advantages_batch = advantages[batch]
                    returns_batch = returns[batch]

                    new_probs, critic_value = self.model(states_batch)
                    critic_value = critic_value.squeeze()

                    new_probs_batch = Categorical(new_probs)
                    new_log_probs = new_probs_batch.log_prob(actions_batch)

                    prob_ratio = (new_log_probs - old_probs_batch).exp()
                    weighted_probs = advantages_batch * prob_ratio
                    weighted_clipped_probs = torch.clamp(prob_ratio, 1 - self.clip_param,
                                                         1 + self.clip_param) * advantages_batch
                    actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                    critic_loss = F.mse_loss(critic_value, returns_batch)

                    entropy = new_probs_batch.entropy().mean()
                    total_loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * entropy

                    self.optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                    self.optimizer.step()

                    # Accumulate losses
                    epoch_actor_loss += actor_loss.item()
                    epoch_critic_loss += critic_loss.item()
                    epoch_entropy += entropy.item()
                    epoch_total_loss += total_loss.item()

            # Average losses over epochs and batches
            num_updates = self.n_epochs * len(batches)
            avg_actor_loss = epoch_actor_loss / num_updates
            avg_critic_loss = epoch_critic_loss / num_updates
            avg_entropy = epoch_entropy / num_updates
            avg_total_loss = epoch_total_loss / num_updates

            self.memory.clear_memory()

            actual_rewards.append(episode_reward)
            explained_variance = self.calculate_explained_variance(returns, vals)
            explained_variance_per_episode.append(explained_variance)
            self.run_rewards_per_episode.append(episode_reward)

            # Log to wandb
            wandb.log({
                'episode': episode,
                'episodic_return': episode_reward,
                'actor_loss': avg_actor_loss,
                'critic_loss': avg_critic_loss,
                'entropy': avg_entropy,
                'total_loss': avg_total_loss,
                'explained_variance': explained_variance,
            })

            # print(f"Episode {episode}, Reward: {episode_reward:.2f}, Loss: {avg_total_loss:.4f}")
            pbar.update(1)

        pbar.close()
        model_file_path = os.path.join(self.model_subdirectory, 'model.pt')
        torch.save(self.model.state_dict(), model_file_path)

        self.visualize_and_log_results(actual_rewards, explained_variance_per_episode, visited_state_counts, alpha)

        return self.model

    def compute_returns(self, rewards, dones, next_value):
        returns = []
        R = next_value
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.discount_factor * R * (1 - dones[step])
            returns.insert(0, R)
        return torch.cat(returns)

    def visualize_and_log_results(self, actual_rewards, explained_variance_per_episode, visited_state_counts, alpha):
        saved_model = load_saved_model(self.model_directory, self.agent_type, self.run_name, self.timestamp,
                                       self.input_dim, self.hidden_dim, self.output_dim)
        value_range = range(0, 101, 10)
        all_states = [np.array([i, j]) for i in value_range for j in value_range]
        all_states_path = visualize_all_states(saved_model, all_states, self.run_name, self.max_episodes, alpha,
                                               self.results_subdirectory)
        wandb.log({"All_States_Visualization": [wandb.Image(all_states_path)]})

        # avg_rewards = [np.mean(rewards) for rewards in actual_rewards]
        # explained_variance_path = os.path.join(self.results_subdirectory, 'explained_variance.png')
        # visualize_explained_variance(explained_variance_per_episode, explained_variance_path)
        # wandb.log({"Explained Variance": [wandb.Image(explained_variance_path)]})

        # Visualize visited states
        states = list(visited_state_counts.keys())
        visit_counts = list(visited_state_counts.values())
        states_visited_path = states_visited_viz(states, visit_counts, alpha, self.results_subdirectory)
        wandb.log({"States Visited": [wandb.Image(states_visited_path)]})

    def calculate_explained_variance(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        if len(y_true) != len(y_pred):
            min_length = min(len(y_true), len(y_pred))
            y_true = y_true[:min_length]
            y_pred = y_pred[:min_length]

        var_y = np.var(y_true)
        return np.mean(1 - np.var(y_true - y_pred) / var_y) if var_y != 0 else 0.0

    def train_single_run(self, seed, alpha):
        set_seed(seed)
        # self.replay_memory = deque(maxlen=self.agent_config['agent']['replay_memory_capacity'])
        self.model = ActorCriticNetwork(self.input_dim, self.hidden_dim, self.output_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.agent_config['agent']['learning_rate'])

        self.run_rewards_per_episode = []  # Store rewards per episode for this run

        torch.autograd.set_detect_anomaly(True)

        pbar = tqdm(total=self.max_episodes, desc="Training Progress", leave=True)

        actual_rewards = []
        explained_variance_per_episode = []
        visited_state_counts = {}

        for episode in range(self.max_episodes):
            state, _ = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32)
            episode_reward = 0
            done = False

            while not done:
                action_probs, value = self.model(state)
                action_dist = Categorical(action_probs)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)

                next_state, reward, done, _, _ = self.env.step(([action.item() * 50], alpha))
                next_state = torch.tensor(next_state, dtype=torch.float32)

                state_tuple = tuple(state.detach().numpy())
                visited_state_counts[state_tuple] = visited_state_counts.get(state_tuple, 0) + 1

                self.memory.states.append(state)
                self.memory.actions.append(action)
                self.memory.probs.append(log_prob)
                self.memory.vals.append(value)
                self.memory.rewards.append(reward)
                self.memory.dones.append(done)

                state = next_state
                episode_reward += reward

            _, next_value = self.model(next_state)
            returns = self.compute_gae(self.memory.rewards, self.memory.vals, self.memory.dones, next_value)

            states, actions, old_probs, vals, rewards, dones, batches = \
                self.memory.generate_batches(self.batch_size)

            values = torch.tensor(vals)
            returns = torch.tensor(returns)
            advantages = torch.tensor(returns) - values

            # Initialize loss tracking variables
            epoch_actor_loss = 0
            epoch_critic_loss = 0
            epoch_entropy = 0
            epoch_total_loss = 0

            for _ in range(self.n_epochs):
                for batch in batches:
                    states_batch = torch.tensor(states[batch], dtype=torch.float)
                    old_probs_batch = torch.tensor(old_probs[batch])
                    actions_batch = torch.tensor(actions[batch])
                    advantages_batch = advantages[batch]
                    returns_batch = returns[batch]

                    new_probs, critic_value = self.model(states_batch)
                    critic_value = critic_value.squeeze()

                    new_probs_batch = Categorical(new_probs)
                    new_log_probs = new_probs_batch.log_prob(actions_batch)

                    prob_ratio = (new_log_probs - old_probs_batch).exp()
                    weighted_probs = advantages_batch * prob_ratio
                    weighted_clipped_probs = torch.clamp(prob_ratio, 1 - self.clip_param,
                                                         1 + self.clip_param) * advantages_batch
                    actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                    critic_loss = F.mse_loss(critic_value, returns_batch)

                    entropy = new_probs_batch.entropy().mean()
                    total_loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * entropy

                    self.optimizer.zero_grad()
                    total_loss.backward()
                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                    self.optimizer.step()

                    # Accumulate losses
                    epoch_actor_loss += actor_loss.item()
                    epoch_critic_loss += critic_loss.item()
                    epoch_entropy += entropy.item()
                    epoch_total_loss += total_loss.item()

            # Average losses over epochs and batches
            num_updates = self.n_epochs * len(batches)
            avg_total_loss = epoch_total_loss / num_updates

            self.memory.clear_memory()

            actual_rewards.append(episode_reward)
            explained_variance = self.calculate_explained_variance(returns, vals)
            explained_variance_per_episode.append(explained_variance)
            self.run_rewards_per_episode.append(episode_reward)


            print(f"Episode {episode}, Reward: {episode_reward:.2f}, Loss: {avg_total_loss:.4f}")
            pbar.update(1)

        pbar.close()
        # After training, save the model
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
        returns_per_episode = np.array(returns_per_episode)

        if returns_per_episode.size == 0:
            print("No data to visualize. Skipping tolerance interval curve.")
            return

        num_episodes = returns_per_episode.shape[1]
        if num_episodes == 0:
            print("No episodes in the runs. Skipping tolerance interval curve.")
            return

        lower_bounds = []
        upper_bounds = []
        central_tendency = []
        episodes = list(range(num_episodes))

        for episode in episodes:
            returns_at_episode = returns_per_episode[:, episode]

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

        # Check for NaNs and Infs
        if np.any(np.isnan(lower_bounds)) or np.any(np.isnan(upper_bounds)) or np.any(np.isnan(central_tendency)):
            print("Warning: Array contains NaNs. Skipping tolerance interval curve.")
            return
        if np.any(np.isinf(lower_bounds)) or np.any(np.isinf(upper_bounds)) or np.any(np.isinf(central_tendency)):
            print("Warning: Array contains Infs. Skipping tolerance interval curve.")
            return

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

    def multiple_runs(self, num_runs, alpha_t, beta_t):
        returns_per_episode = []

        for run in range(num_runs):
            returns = self.train_single_run(run, alpha_t)
            returns_per_episode.append(returns)

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

def load_saved_model(model_directory, agent_type, run_name, timestamp, input_dim, hidden_dim, action_space_nvec):
    model_subdirectory = os.path.join(model_directory, agent_type, run_name, timestamp)
    model_file_path = os.path.join(model_subdirectory, 'model.pt')

    if not os.path.exists(model_file_path):
        print(f"Model file not found in {model_file_path}")
        return None

    model = ActorCriticNetwork(input_dim, hidden_dim, action_space_nvec)
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


