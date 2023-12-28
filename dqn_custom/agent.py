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
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        agent_type = "dqn_custom"
        self.results_subdirectory = os.path.join(self.results_directory, agent_type, run_name, timestamp)
        os.makedirs(self.results_subdirectory, exist_ok=True)

        # Set up logging to the correct directory
        log_file_path = os.path.join(self.results_subdirectory, 'agent_log.txt')
        logging.basicConfig(filename=log_file_path, level=logging.INFO)
        # Initialize the neural network
        input_dim = len(env.reset()[0])
        output_dim = np.prod(env.action_space.nvec)
        hidden_dim = self.agent_config['agent']['hidden_units']
        self.model = DeepQNetwork(input_dim, hidden_dim, env.action_space.nvec)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.agent_config['agent']['learning_rate'])

        # Initialize agent-specific configurations and variables
        self.env = env
        self.run_name = run_name
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
        for episode in range(self.max_episodes):
            state, _ = self.env.reset()
            print("state after reset", state, state.shape)
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
                if np.random.random() > self.exploration_rate:
                    # Select an action for each dimension independently
                    # action_values = self.model(state_tensor)

                    action_values = self.model(torch.FloatTensor(state))
                    action = [torch.argmax(values).item() for values in action_values]
                else:
                    action = self.env.action_space.sample()
                print("action: ", action)
                # list_action = list(eval(self.all_actions[action]))
                c_list_action = [i * 50 for i in action]
                action_alpha_list = [*c_list_action, alpha]
                next_state, reward, terminated, _, info = self.env.step(action_alpha_list)
                self.update_replay_memory(state, action, reward, next_state, terminated)
                state = next_state
                total_reward += reward

                self.train_step()

            self.exploration_rate *= self.exploration_decay_rate
            self.exploration_rate = max(self.exploration_rate, self.min_exploration_rate)

            print(f"Episode: {episode}, Total reward: {total_reward}, Exploration rate: {self.exploration_rate}")

        # After training, save the model
        torch.save(self.model.state_dict(), 'dqn_model.pth')


