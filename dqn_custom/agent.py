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
    def __init__(self, input_dim, hidden_dim, action_space_nvec):
        super(DeepQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        # Creating a separate output layer for each action dimension
        self.output_layers = nn.ModuleList([nn.Linear(hidden_dim, n) for n in action_space_nvec])

    def forward(self, x):
        x = self.relu(self.fc1(x))
        # Compute the Q-values for each action dimension
        return [layer(x) for layer in self.output_layers]


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
        self.model = DeepQNetwork(self.input_dim, self.hidden_dim, env.action_space.nvec)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.agent_config['agent']['learning_rate'])

        # Initialize agent-specific configurations and variables
        self.env = env
        self.max_episodes = self.agent_config['agent']['max_episodes']
        self.discount_factor = self.agent_config['agent']['discount_factor']
        self.exploration_rate = self.agent_config['agent']['exploration_rate']
        self.min_exploration_rate = self.agent_config['agent']['min_exploration_rate']
        self.exploration_decay_rate = self.agent_config['agent']['exploration_decay_rate']

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
        # self.state_action_visits = np.zeros((rows, columns))
        # self.state_action_visits = np.zeros((self.env.observation_space.nvec, self.env.action_space.nvec))

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        output = self.model(state_tensor)
        # Select the action with the highest Q-value in each dimension
        return [torch.argmax(q_values).item() for q_values in output]

    def compute_q_value(self, states, actions):
        states = torch.FloatTensor(states)
        q_values = self.model(states)
        # Extract the Q-value for the taken action in each dimension
        return torch.stack([q_values[i].gather(1, actions[:, i].unsqueeze(1)) for i in range(len(q_values))], dim=1)

    def update_replay_memory(self, state, action, reward, next_state, done):
        self.replay_memory.append((state, action, reward, next_state, done))

    def train_step(self):
        if len(self.replay_memory) < self.batch_size:
            return

        minibatch = random.sample(self.replay_memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards))
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones))

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Compute current and next Q-values
        curr_Q = self.compute_q_value(states, actions)
        next_Q = torch.stack([q_values.max(1)[0] for q_values in self.model(next_states)], dim=1)

        expected_Q = rewards + self.discount_factor * next_Q * (1 - dones.unsqueeze(1))
        loss = nn.MSELoss()(curr_Q, expected_Q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def train(self, alpha):
        actual_rewards = []
        predicted_rewards = []
        rewards = []
        rewards_per_episode = []
        last_episode = {}
        # Initialize visited state counts dictionary
        visited_state_counts = {}
        for episode in tqdm(range(self.max_episodes)):
            state, _ = self.env.reset()
            # print("state after reset", state, state.shape)
            total_reward = 0
            c_state = state[0]
            terminated = False
            e_return = []
            e_allowed = []
            e_infected_students = []
            state_transitions = []
            total_reward = 0
            e_predicted_rewards = []
            e_community_risk = []
            last_episode['infected'] = e_infected_students
            last_episode['allowed'] = e_allowed
            last_episode['community_risk'] = e_community_risk

            while not terminated:
                # state_tensor = torch.FloatTensor(state.flatten()).unsqueeze(0)
                # Select an action
                print("State: ", state, type(state))
                if np.random.random() > self.exploration_rate:
                    # Select an action for each dimension independently
                    # action_values = self.model(state_tensor)

                    action_values = self.model(torch.FloatTensor(state))
                    action = [torch.argmax(values).item() for values in action_values]
                    # Retrieve the predicted reward (Q-value) for the selected action
                    predicted_reward = [values.max().item() for values in action_values]
                else:
                    action = self.env.action_space.sample()
                    # For a random action, we don't have a predicted reward from the model
                    predicted_reward = [0] * len(action)  # or set to None, based on how you want to handle it

                e_predicted_rewards.append(predicted_reward)

                print("action: ", action)
                # list_action = list(eval(self.all_actions[action]))
                c_list_action = [i * 50 for i in action]
                action_alpha_list = [*c_list_action, alpha]
                next_state, reward, terminated, _, info = self.env.step(action_alpha_list)
                self.update_replay_memory(state, action, reward, next_state, terminated)
                state = next_state
                total_reward += reward

                self.train_step()
                # Update other accumulators...
                week_reward = int(reward)
                total_reward += week_reward
                e_return.append(week_reward)
                e_allowed.append(info['allowed'])
                e_infected_students.append(info['infected'])
                e_community_risk.append(info['community_risk'])
                converted_state = str(state)
                if converted_state not in visited_state_counts:
                    visited_state_counts[converted_state] = 1
                else:
                    visited_state_counts[converted_state] += 1
                print(info)

            avg_episode_return = sum(e_return) / len(e_return)
            rewards_per_episode.append(avg_episode_return)
            actual_rewards.append(e_return)
            predicted_rewards.append(e_predicted_rewards)

            self.exploration_rate *= self.exploration_decay_rate
            self.exploration_rate = max(self.exploration_rate, self.min_exploration_rate)

            if episode >= self.moving_average_window - 1:
                window_rewards = rewards_per_episode[max(0, episode - self.moving_average_window + 1):episode + 1]
                moving_avg = np.mean(window_rewards)
                std_dev = np.std(window_rewards)

                # Store the current moving average for comparison in the next episode
                self.prev_moving_avg = moving_avg

                # Log the moving average and standard deviation along with the episode number
                wandb.log({
                    'Moving Average': moving_avg,
                    'Standard Deviation': std_dev,
                    'average_return': total_reward/len(e_return),
                    'step': episode  # Ensure the x-axis is labeled correctly as 'Episodes'
                })

        # After training, save the model

        model_file_path = os.path.join(self.model_subdirectory, 'model.pt')
        torch.save(self.model.state_dict(), model_file_path)
        states = list(visited_state_counts.keys())
        visit_counts = list(visited_state_counts.values())
        states_visited_path = states_visited_viz(states, visit_counts, alpha, self.results_subdirectory)
        wandb.log({"States Visited": [wandb.Image(states_visited_path)]})

        avg_rewards = [sum(lst) / len(lst) for lst in actual_rewards]
        print("avg rewards", avg_rewards)
        # Pass actual and predicted rewards to visualizer
        explained_variance_path = visualize_explained_variance(actual_rewards, predicted_rewards,
                                                               self.results_subdirectory, self.max_episodes)
        wandb.log({"Explained Variance": [wandb.Image(explained_variance_path)]})

        file_path_variance = visualize_variance_in_rewards(avg_rewards, self.results_subdirectory, self.max_episodes)
        wandb.log({"Variance in Rewards": [wandb.Image(file_path_variance)]})

        # # Inside the train method, after training the agent:
        saved_model = load_saved_model(self.model_directory, self.agent_type, self.run_name, self.timestamp, self.input_dim, self.hidden_dim, self.output_dim)
        value_range = range(0, 101, 10)
        # Generate all combinations of states
        all_states = [np.array([i, j]) for i in value_range for j in value_range]
        all_states_path = visualize_all_states(saved_model, all_states, self.run_name,
                                               self.max_episodes, alpha,
                                               self.results_subdirectory)
        wandb.log({"All_States_Visualization": [wandb.Image(all_states_path)]})

        file_path_heatmap = visualize_variance_in_rewards_heatmap(rewards_per_episode, self.results_subdirectory,
                                                                  bin_size=10)  # 25 for 2500 episodes, 10 for 1000 episodes
        wandb.log({"Variance in Rewards Heatmap": [wandb.Image(file_path_heatmap)]})

        print("infected: ", last_episode['infected'], "allowed: ", last_episode['allowed'], "community_risk: ",
              last_episode['community_risk'])
        file_path_infected_vs_community_risk = visualize_infected_vs_community_risk_table(last_episode, alpha,
                                                                                          self.results_subdirectory)
        wandb.log({"Infected vs Community Risk": [wandb.Image(file_path_infected_vs_community_risk)]})

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




