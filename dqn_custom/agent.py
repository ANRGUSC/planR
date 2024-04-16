import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR, StepLR
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
seed = 100
random.seed(seed)
torch.manual_seed(seed)

class DeepQNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DeepQNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        # Basic network layers
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # Adjusted to only take `input_dim`
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # Creating a separate output layer for each action dimension
        self.output_layers = nn.ModuleList([nn.Linear(hidden_dim, n) for n in output_dim])

    def forward(self, x):
        h = self.net(x)
        # Compute the Q-values for each action dimension
        return [layer(F.relu(h)) for layer in self.output_layers]


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
        # Initialize the neural network
        self.input_dim = len(env.reset()[0])
        self.output_dim = env.action_space.nvec
        self.hidden_dim = self.agent_config['agent']['hidden_units']
        self.model = DeepQNetwork(self.input_dim, self.hidden_dim, self.output_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.agent_config['agent']['learning_rate'])

        # Initialize agent-specific configurations and variables
        self.env = env
        self.max_episodes = self.agent_config['agent']['max_episodes']
        self.discount_factor = self.agent_config['agent']['discount_factor']
        self.exploration_rate = self.agent_config['agent']['exploration_rate']
        self.min_exploration_rate = self.agent_config['agent']['min_exploration_rate']
        self.exploration_decay_rate = self.agent_config['agent']['exploration_decay_rate']
        self.learning_rate = self.agent_config['agent']['learning_rate']

        # Parameters for adjusting learning rate over time
        self.learning_rate_decay = self.agent_config['agent']['learning_rate_decay']
        self.min_learning_rate = self.agent_config['agent']['min_learning_rate']
        self.scheduler = StepLR(self.optimizer, step_size=100, gamma=0.95)  # Initialize the scheduler


        # Replay memory
        self.replay_memory = deque(maxlen=self.agent_config['agent']['replay_memory_capacity'])
        self.batch_size = self.agent_config['agent']['batch_size']
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # self.scheduler = ExponentialLR(self.optimizer, gamma=self.learning_rate_decay)

        self.possible_actions = [list(range(0, (k))) for k in self.env.action_space.nvec]
        self.all_actions = [str(i) for i in list(itertools.product(*self.possible_actions))]

        # moving average for early stopping criteria
        self.moving_average_window = 100  # Number of episodes to consider for moving average
        self.stopping_criterion = 0.01  # Threshold for stopping
        self.prev_moving_avg = -float(
            'inf')  # Initialize to negative infinity to ensure any reward is considered an improvement in the first episode.
        self.loss_sum = 0.0
        self.loss_count = 0


    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return [np.random.randint(0, n) for n in self.output_dim]
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = self.model(state)
        return [torch.argmax(q).item() for q in q_values]

    def softmax_act(self, state):
        if np.random.rand() < self.exploration_rate:
            # Return random actions for each dimension
            return [np.random.randint(0, n) for n in self.output_dim]

        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = self.model(state)

        # Convert Q-values into actions using softmax
        actions = []
        for q in q_values:
            q = q.squeeze()
            probabilities = F.softmax(q, dim=0).detach().numpy()
            if not np.isclose(probabilities.sum(), 1):
                print("Probabilities do not sum to 1:", probabilities.sum())
            if np.any(np.isnan(probabilities)):
                print("NaN values in probabilities")
            action = np.random.choice(np.arange(len(probabilities)), p=probabilities)
            actions.append(action)
        # exit()

        return actions

    def remember(self, state, action, reward, next_state, done):
        self.replay_memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size, episode):
        if len(self.replay_memory) < batch_size:
            return  # Not enough samples in the replay buffer to perform a training step

        minibatch = random.sample(self.replay_memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.stack(states)
        next_states = torch.stack(next_states)
        actions = torch.tensor(actions, dtype=torch.long)  # ensure actions are correctly shaped
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Fetch the Q-values from the model
        current_q_values_list = self.model(states)
        next_q_values_list = self.model(next_states)

        # Remove the unnecessary middle dimension and stack along the batch axis
        current_q_values = torch.cat([q.squeeze(1) for q in current_q_values_list], dim=0)
        next_q_values = torch.cat([q.squeeze(1) for q in next_q_values_list], dim=0)

        # Gather the selected Q-values based on actions
        current_q_values = current_q_values.gather(1, actions)


        # Take max along the actions dimension for each next state Q-values
        max_next_q_values = next_q_values.max(1, keepdim=True)[0]

        # Compute the target Q values
        targets = rewards.unsqueeze(1) + self.discount_factor * max_next_q_values * (1 - dones.unsqueeze(1))

        # Compute the loss (Mean Squared Error)
        loss = nn.MSELoss()(current_q_values, targets)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # Update loss tracking
        self.loss_sum += loss.item()
        self.loss_count += 1

        # Calculate and log the moving average of the loss every 100 episodes
        if episode % 100 == 0 and self.loss_count > 0:
            moving_average_loss = self.loss_sum / self.loss_count
            wandb.log({"Moving Average (Loss)": moving_average_loss}, step=episode)
            # Reset tracking after logging
            self.loss_sum = 0.0
            self.loss_count = 0

    def train(self, alpha):
        actual_rewards = []
        predicted_rewards = []
        rewards_per_episode = []
        visited_state_counts = {}  # Dictionary to keep track of state visits

        for episode in tqdm(range(self.max_episodes)):
            state, _ = self.env.reset()
            # print("State: ", state)
            state = torch.FloatTensor(state).view(-1, self.input_dim)
            total_reward = 0
            done = False
            episode_rewards = []
            episode_predicted_rewards = []

            while not done:
                # action = self.act(state)
                action = self.softmax_act(state)
                # Perform action in the environment
                c_list_action = [i * 50 for i in action]
                action_alpha_list = [*c_list_action, alpha]

                next_state, reward, done,_, info = self.env.step(action_alpha_list)
                next_state = torch.FloatTensor(next_state).view(-1, self.input_dim)

                self.remember(state, action, reward, next_state, done)
                self.replay(self.batch_size, episode)  # Direct learning from the memory after each step

                state = next_state
                episode_rewards.append(reward)
                total_reward += reward
                # print(info)

            # Log episode statistics
            rewards_per_episode.append(int(total_reward/15))
            actual_rewards.append(episode_rewards)
            # Log the moving average of rewards to monitor training progress
            if episode >= self.moving_average_window - 1:
                window_rewards = rewards_per_episode[-self.moving_average_window:]
                moving_avg = np.mean(window_rewards)
                std_dev = np.std(window_rewards)
                wandb.log({
                    'Moving Average (Reward)': moving_avg,
                    'Standard Deviation': std_dev,
                    'Raw Reward': int(total_reward/15),
                    'step': episode  # Ensure the x-axis is labeled correctly as 'Episodes'
                })

            # Adjust the exploration rate
                # Adjust the exploration rate
            # self.exploration_rate = max(self.agent_config['agent']['min_exploration_rate'],
            #                                 self.exploration_rate - self.exploration_decay_rate)
            self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay_rate)
            # self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate - (self.exploration_rate - self.min_exploration_rate) / self.max_episodes)

            # decay = (1 - episode / self.max_episodes) ** 2
            # self.learning_rate = max(self.min_learning_rate, self.learning_rate * decay)
            self.scheduler.step()

        model_name = self.run_name + ".pt"
        # Save the model after training is completed
        model_file_path = os.path.join(self.model_subdirectory, model_name)
        torch.save(self.model.state_dict(), model_file_path)
        # Inside the train method, after training the agent:
        saved_model = load_saved_model(self.model_directory, self.agent_type, self.run_name, self.timestamp, self.input_dim, self.hidden_dim, self.output_dim, model_name)
        value_range = range(0, 101, 10)
        # Generate all combinations of states
        all_states = [np.array([i, j]) for i in value_range for j in value_range]
        all_states_path = visualize_all_states(saved_model, all_states, self.run_name,
                                                   self.max_episodes, alpha,
                                                   self.results_subdirectory)
        wandb.log({"All_States_Visualization": [wandb.Image(all_states_path)]})

        return self.model

def load_saved_model(model_directory, agent_type, run_name, timestamp, input_dim, hidden_dim, action_space_nvec, model_name):
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
    model_file_path = os.path.join(model_subdirectory, model_name)

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