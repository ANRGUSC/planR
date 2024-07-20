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
    visualize_explained_variance, visualize_variance_in_rewards, visualize_infected_vs_community_risk_table, states_visited_viz
import wandb


class DeepQNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super(DeepQNetwork, self).__init__()
        self.hidden_dim = hidden_dim  # Add this line to store the hidden dimension

        self.encoder = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, h):
        h_prime = self.encoder(torch.cat((x, h), dim=-1))
        Q_values = self.out(h_prime)
        # Compute the Q-values for each action dimension
        return Q_values, h_prime


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
        self.model = DeepQNetwork(self.input_dim, self.hidden_dim, self.output_dim)
        self.target_model = DeepQNetwork(self.input_dim, self.hidden_dim, self.output_dim)
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

    def update_replay_memory(self, state, action, reward, next_state, done):
        # Convert to appropriate types before appending to replay memory
        state = np.array(state, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        action = int(action)
        reward = float(reward)
        done = bool(done)
        self.replay_memory.append((state, action, reward, next_state, done))

    def compute_q_value(self, states, actions):
        q_values, _ = self.model(states, torch.zeros(states.size(0), self.hidden_dim))
        return q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    def loss_function(self, action_probabilities, rewards):
        discounted_rewards = torch.zeros_like(rewards)
        discounted_rewards[-1] = rewards[-1]
        for i in range(len(rewards) - 1):
            curr = discounted_rewards[-(i + 1)]
            discounted_rewards[-(i + 2)] = rewards[-(i + 2)] + curr * self.discount_factor
        # p is small, d is large -> loss will be large
        # p is large, d is large -> loss will be smaller
        # p is small, d is small -> loss will be small
        # p is large, d is small -> loss will be larger
        return (-torch.log(action_probabilities) * discounted_rewards).sum()

    def train(self, alpha):
        pbar = tqdm(total=self.max_episodes)
        decay_rate = np.log(self.min_exploration_rate / self.exploration_rate) / self.max_episodes

        # Initialize accumulators for visualization
        actual_rewards = []
        predicted_rewards = []
        rewards_per_episode = []
        visited_state_counts = {}
        last_episode = {'infected': [], 'allowed': [], 'community_risk': []}

        for episode in range(self.max_episodes):
            state, _ = self.env.reset()
            state = torch.FloatTensor(state)
            terminated = False

            hidden_state = torch.zeros(self.hidden_dim)

            action_probabilities = []
            rewards = []
            visited_states = []  # List to store visited states
            loss = torch.tensor(0.0)

            while not terminated:
                # Action Distribution
                Q_values, hidden_state = self.model(state, hidden_state)
                prob_dist = torch.nn.functional.softmax(Q_values, dim=-1)

                if np.random.random() > self.exploration_rate:
                    action = prob_dist.argmax()
                else:
                    action = np.random.choice(len(prob_dist), p=prob_dist.detach().numpy())

                action_probabilities.append(prob_dist[action])
                next_state, reward, terminated, _, info = self.env.step(([action * 50], alpha))
                rewards.append(reward)

                visited_states.append(state.tolist())  # Add the current state to the list of visited states

                self.update_replay_memory(state, action, reward, next_state, terminated)

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
                    next_Q_values, _ = self.target_model(next_states, torch.zeros(next_states.size(0), self.hidden_dim))
                    next_Q = next_Q_values.max(1)[0]
                expected_Q = rewards_batch + self.discount_factor * next_Q * (1 - dones.unsqueeze(1))
                loss = nn.MSELoss()(curr_Q, expected_Q)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Log metrics to wandb
                mean_q_value = curr_Q.mean().item()
                wandb.log({
                    "loss": loss.item(),
                    "learning_rate": self.optimizer.param_groups[0]['lr'],
                    "exploration_rate": self.exploration_rate,
                    "mean_q_value": mean_q_value,
                    "episode": episode,
                    "avg_reward": torch.tensor(rewards, dtype=torch.float32).mean().item(),
                    "total_reward": torch.tensor(rewards, dtype=torch.float32).sum().item()
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

            # Decay the exploration rate
            # self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * np.exp(decay_rate))
            self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate - (
                    1.0 - self.min_exploration_rate) / self.max_episodes)


            # Update target network
            if episode % self.target_network_frequency == 0:
                self.target_model.load_state_dict(self.model.state_dict())

            pbar.update(1)
            pbar.set_description(
                f"Loss {float(loss.item())} Avg R {float(torch.tensor(rewards, dtype=torch.float32).mean().numpy())}")

        # After training, save the model
        model_file_path = os.path.join(self.model_subdirectory, 'model.pt')
        torch.save(self.model.state_dict(), model_file_path)

        # Visualization and logging
        states = list(visited_state_counts.keys())
        visit_counts = list(visited_state_counts.values())
        states_visited_path = states_visited_viz(states, visit_counts, alpha, self.results_subdirectory)
        wandb.log({"States Visited": [wandb.Image(states_visited_path)]})

        avg_rewards = [sum(lst) / len(lst) for lst in actual_rewards]
        explained_variance_path = visualize_explained_variance(actual_rewards, predicted_rewards,
                                                               self.results_subdirectory, self.max_episodes)
        wandb.log({"Explained Variance": [wandb.Image(explained_variance_path)]})

        file_path_variance = visualize_variance_in_rewards(avg_rewards, self.results_subdirectory, self.max_episodes)
        wandb.log({"Variance in Rewards": [wandb.Image(file_path_variance)]})

        saved_model = load_saved_model(self.model_directory, self.agent_type, self.run_name, self.timestamp,
                                       self.input_dim, self.hidden_dim, self.output_dim)
        value_range = range(0, 101, 10)
        all_states = [np.array([i, j]) for i in value_range for j in value_range]
        all_states_path = visualize_all_states(saved_model, all_states, self.run_name, self.max_episodes, alpha,
                                               self.results_subdirectory)
        wandb.log({"All_States_Visualization": [wandb.Image(all_states_path)]})

        return self.model


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
