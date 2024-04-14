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
seed = 100
random.seed(seed)
torch.manual_seed(seed)

class DeepQNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_space_nvec):
        super(DeepQNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # Creating a separate output layer for each action dimension
        self.output_layers = nn.ModuleList([nn.Linear(hidden_dim, n) for n in action_space_nvec])

    def forward(self, x, h):
        x_h = torch.cat((x,h),dim=-1)
        h = self.net(x_h)
        # Compute the Q-values for each action dimension
        return [layer(torch.nn.functional.relu(h)) for layer in self.output_layers], h


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
        self.learning_rate = self.agent_config['agent']['learning_rate']

        # Parameters for adjusting learning rate over time
        self.learning_rate_decay = self.agent_config['agent']['learning_rate_decay']
        self.min_learning_rate = self.agent_config['agent']['min_learning_rate']

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

        self.exploration_decay_rate = (self.exploration_rate - self.min_exploration_rate) / self.max_episodes

    # def select_action(self, state):
    #     if np.random.rand() < self.exploration_rate:
    #         # Exploration: Randomly select an action for each action dimension
    #         action = [random.randint(0, n - 1) for n in self.output_dim]
    #     else:
    #         # Exploitation: Select the action with the highest Q-value for each action dimension
    #         state_tensor = torch.FloatTensor(state).unsqueeze(0)
    #         q_values = self.model(state_tensor)
    #         action = [torch.argmax(values).item() for values in q_values]
    #     return action

    def compute_q_value(self, states, actions, hidden_state):
        states = torch.FloatTensor(states)
        q_values, _ = self.model(states, hidden_state)
        # Extract the Q-value for the taken action in each dimension
        return torch.stack([q_values[i].gather(1, actions[:, i].unsqueeze(1)) for i in range(len(q_values))], dim=1)

    def update_replay_memory(self, state, action, reward, next_state, hidden_state, done):
        self.replay_memory.append((state, action, reward, next_state, hidden_state, done))

    def train_step(self):
        # print("Replay Memory Length: ", len(self.replay_memory), "Batch Size: ", self.batch_size)
        if len(self.replay_memory) < self.batch_size:
            return

        minibatch = random.sample(self.replay_memory, self.batch_size)
        states, actions, rewards, next_states, hidden_states, dones = zip(*minibatch)
        states = torch.cat(states,dim=0)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        hidden_states = torch.cat(hidden_states,dim=0)
        dones = torch.FloatTensor(dones)

        # Compute current Q-values
        curr_Q = self.compute_q_value(states, actions, hidden_states)

        # Compute next Q-values from the model; get max Q-value at the next state
        next_Q_values, _ = self.model(next_states, hidden_states)
        max_next_Q = torch.stack([q_values.max(1)[0] for q_values in next_Q_values], dim=1)

        # Compute the expected Q values
        expected_Q = rewards.unsqueeze(1) + self.discount_factor * max_next_Q * (1 - dones.unsqueeze(1))

        # Calculate loss
        loss = nn.MSELoss()(curr_Q, expected_Q)
        wandb.log({"Loss": loss.item()})
        print("Loss: ", loss.item())

        # Log and optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, alpha):
        actual_rewards = []
        predicted_rewards = []
        rewards = []
        rewards_per_episode = []
        visited_state_counts = {}  # Dictionary to keep track of state visits

        for episode in tqdm(range(self.max_episodes)):
            state, _ = self.env.reset()
            state = torch.FloatTensor(state).view(-1,self.input_dim)
            hidden_state = torch.zeros((len(state),self.hidden_dim)).float()
            total_reward = 0
            terminated = False
            episode_rewards = []
            episode_predicted_rewards = []

            while not terminated:
                # state_key = str(state)  # Convert state to a string to use as a key in the dictionary
                # if state_key in visited_state_counts:
                #     visited_state_counts[state_key] += 1
                # else:
                #     visited_state_counts[state_key] = 1

                q_values, hidden_state = self.model(state, hidden_state)
                if random.uniform(0, 1) > self.exploration_rate:
                    # Exploitation: Select the action with the highest Q-value for each action dimension
                    action = [torch.argmax(values).item() for values in q_values]
                    # predicted_reward = [values.max().item() for values in q_values]
                else:
                    # Exploration: Random action
                    # p_values = torch.softmax(torch.stack(q_values,dim=0),dim=-1)
                    # action = [random.choice([i for i in range(len(p))],weights=p.detach().numpy()) for p in p_values]
                    action = [random.randint(0, n - 1) for n in self.output_dim]
                    # predicted_reward = [0] * len(action)

                # episode_predicted_rewards.append(predicted_reward)

                # Perform action in the environment
                c_list_action = [i * 50 for i in action]
                action_alpha_list = [*c_list_action, alpha]
                next_state, reward, terminated, _, info = self.env.step(action_alpha_list)

                # Store the transition in replay memory
                self.update_replay_memory(state, action, reward, next_state, hidden_state, terminated)

                # Update state and accumulate reward
                state = torch.FloatTensor(next_state).view(-1,self.input_dim)
                episode_rewards.append(reward)
                total_reward += reward

            # After each episode, perform a training step
            self.train_step()
            self.replay_memory = []

            # Log episode statistics
            avg_episode_return = sum(episode_rewards) / len(episode_rewards)
            rewards_per_episode.append(avg_episode_return)
            actual_rewards.append(episode_rewards)
            predicted_rewards.append(episode_predicted_rewards)

            # Adjust exploration rate and learning rate
            self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate - (
                        self.exploration_rate - self.min_exploration_rate) / self.max_episodes)
            decay = (1 - episode / self.max_episodes) ** 2
            self.learning_rate = max(self.min_learning_rate, self.learning_rate * decay)

            # Log and manage moving average for early stopping or adjustments
            if episode >= self.moving_average_window - 1:
                window_rewards = rewards_per_episode[max(0, episode - self.moving_average_window + 1):episode + 1]
                moving_avg = np.mean(window_rewards)
                std_dev = np.std(window_rewards)
                wandb.log({
                    'Moving Average': moving_avg,
                    'Standard Deviation': std_dev,
                    'average_return': avg_episode_return
                })
        # After training, save the model

        model_file_path = os.path.join(self.model_subdirectory, 'model.pt')
        torch.save(self.model.state_dict(), model_file_path)
        # states = list(visited_state_counts.keys())
        # visit_counts = list(visited_state_counts.values())
        # states_visited_path = states_visited_viz(states, visit_counts, alpha, self.results_subdirectory)
        # wandb.log({"States Visited": [wandb.Image(states_visited_path)]})
        #
        # avg_rewards = [sum(lst) / len(lst) for lst in actual_rewards]
        #
        # explained_variance_path = visualize_explained_variance(actual_rewards, predicted_rewards,
        #                                                        self.results_subdirectory, self.max_episodes)
        # wandb.log({"Explained Variance": [wandb.Image(explained_variance_path)]})
        #
        # file_path_variance = visualize_variance_in_rewards(avg_rewards, self.results_subdirectory, self.max_episodes)
        # wandb.log({"Variance in Rewards": [wandb.Image(file_path_variance)]})

        # # Inside the train method, after training the agent:
        saved_model = load_saved_model(self.model_directory, self.agent_type, self.run_name, self.timestamp, self.input_dim, self.hidden_dim, self.output_dim)
        value_range = range(0, 101, 10)
        # Generate all combinations of states
        all_states = [np.array([i, j]) for i in value_range for j in value_range]
        all_states_path = visualize_all_states(saved_model, all_states, self.run_name,
                                               self.max_episodes, alpha,
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