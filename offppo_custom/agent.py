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
import math
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

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

class ActorCriticNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super(ActorCriticNetwork, self).__init__()
        self.hidden_dim = hidden_dim  # Add this line to store the hidden dimension

        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        shared_output = self.shared_layers(x)
        policy_dist = self.actor(shared_output)
        value = self.critic(shared_output)
        return policy_dist, value

class OffPPOCustomAgent:
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
        self.agent_type = "offppo_custom"
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
        self.target_model = ActorCriticNetwork(self.input_dim, self.hidden_dim, self.output_dim)  # Add target model
        self.target_model.load_state_dict(self.model.state_dict())  # Initialize target model
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.agent_config['agent']['learning_rate'])

        # Initialize agent-specific configurations and variables
        self.env = env
        self.max_episodes = self.agent_config['agent']['max_episodes']
        self.discount_factor = self.agent_config['agent']['discount_factor']
        self.epsilon = self.agent_config['agent']['epsilon']  # Clipping parameter for PPO
        self.lmbda = self.agent_config['agent']['lambda']  # GAE parameter
        self.min_exploration_rate = self.agent_config['agent']['min_exploration_rate']
        self.exploration_rate = self.agent_config['agent']['exploration_rate']
        self.target_network_frequency = self.agent_config['agent']['target_network_frequency']

        # Replay memory
        self.replay_memory = deque(maxlen=self.agent_config['agent']['replay_memory_capacity'])
        self.batch_size = self.agent_config['agent']['batch_size']

        self.possible_actions = [list(range(0, (k))) for k in self.env.action_space.nvec]
        self.all_actions = [str(i) for i in list(itertools.product(*self.possible_actions))]

        # Moving average for early stopping criteria
        self.moving_average_window = 100  # Number of episodes to consider for moving average
        self.stopping_criterion = 0.01  # Threshold for stopping
        self.prev_moving_avg = -float('inf')  # Initialize to negative infinity to ensure any reward is considered an improvement in the first episode.

        # Hidden State
        self.hidden_state = None
        self.reward_window = deque(maxlen=self.moving_average_window)
        self.scheduler = StepLR(self.optimizer, step_size=100, gamma=0.9)


    def update_replay_memory(self, state, action, reward, next_state, done, log_prob, value):
        # Convert to appropriate types before appending to replay memory
        state = np.array(state, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        action = int(action)
        reward = float(reward)
        done = bool(done)
        self.replay_memory.append((state, action, reward, next_state, done, log_prob, value))

    def compute_advantages(self, rewards, values, dones):
        advantages = []
        returns = []
        advantage = 0
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = values[i + 1] if i + 1 < len(values) else values[i]
            else:
                next_value = values[i + 1]

            td_error = rewards[i] + self.discount_factor * next_value * (1 - dones[i]) - values[i]
            advantage = td_error + self.discount_factor * self.lmbda * (1 - dones[i]) * advantage
            advantages.insert(0, advantage)
            returns.insert(0, advantage + values[i])

        return torch.FloatTensor(advantages), torch.FloatTensor(returns)

    def train(self, alpha):
        pbar = tqdm(total=self.max_episodes, desc="Training Progress", leave=True)

        # Initialize accumulators for visualization
        actual_rewards = []
        predicted_rewards = []
        rewards_per_episode = []
        visited_state_counts = {}

        for episode in range(self.max_episodes):
            state, _ = self.env.reset()
            state = torch.FloatTensor(state)
            terminated = False

            episode_rewards = []
            episode_log_probs = []
            episode_values = []
            episode_dones = []
            visited_states = []  # List to store visited states

            while not terminated:
                policy_dist, value = self.model(state)

                # Check for NaN or Inf values immediately
                if torch.any(torch.isnan(policy_dist)) or torch.any(torch.isinf(policy_dist)):
                    print(f"NaN or Inf detected in policy_dist at episode {episode}")
                    print(f"policy_dist: {policy_dist}")
                    print(f"value: {value}")
                    raise ValueError("policy_dist contains NaN or Inf values")

                # Clamp policy_dist to ensure all elements are within valid range
                policy_dist = torch.clamp(policy_dist, min=1e-9, max=1.0 - 1e-9)

                # Normalize policy_dist to ensure it sums to 1
                policy_dist = policy_dist / policy_dist.sum()

                action = torch.multinomial(policy_dist, 1).item()
                log_prob = torch.log(policy_dist[action] + 1e-9)  # Add epsilon to avoid log(0)
                next_state, reward, terminated, _, info = self.env.step(([action * 50], alpha))
                episode_rewards.append(reward)
                episode_log_probs.append(log_prob)  # Do not detach yet
                episode_values.append(value)  # Do not detach yet
                episode_dones.append(terminated)

                visited_states.append(state.tolist())  # Add the current state to the list of visited states

                self.update_replay_memory(state, action, reward, next_state, terminated, log_prob, value)

                state = torch.FloatTensor(next_state)

            # Compute advantages and returns
            _, last_value = self.model(state)
            episode_values.append(last_value)
            advantages, returns = self.compute_advantages(episode_rewards, episode_values, episode_dones)

            # Ensure returns and episode_values are the same length
            if len(returns) != len(episode_values) - 1:
                raise ValueError(
                    f"Mismatch in lengths: returns({len(returns)}) vs episode_values({len(episode_values) - 1})")

            # Initialize losses
            policy_loss = torch.tensor(0.0)
            value_loss = torch.tensor(0.0)
            loss = torch.tensor(0.0)

            # Training step
            if len(self.replay_memory) >= self.batch_size:
                minibatch = random.sample(self.replay_memory, self.batch_size)
                states, actions, rewards_batch, next_states, dones, log_probs, values = zip(*minibatch)

                # Convert lists to tensors
                states = torch.stack([torch.FloatTensor(s) for s in states])
                actions = torch.LongTensor(actions)
                old_log_probs = torch.stack([lp.detach() for lp in log_probs])
                values = torch.stack([v.detach() for v in values])  # Detach values here

                # Recompute advantages for the sampled minibatch
                advantages_batch = []
                returns_batch = []
                for r, v, d in zip(rewards_batch, values, dones):
                    adv, ret = self.compute_advantages([r], [v], [d])
                    advantages_batch.append(adv)
                    returns_batch.append(ret)

                advantages_batch = torch.cat(advantages_batch).detach()
                returns_batch = torch.cat(returns_batch).detach()

                # Ensure the shapes match
                if states.size(0) != advantages_batch.size(0):
                    raise ValueError(
                        f"Shape mismatch: states({states.size(0)}) vs advantages({advantages_batch.size(0)})")

                policy_dist, values = self.model(states)
                log_probs = torch.log(
                    policy_dist.gather(1, actions.unsqueeze(1)).squeeze(1) + 1e-9)  # Add epsilon to avoid log(0)
                ratios = torch.exp(log_probs - old_log_probs)

                surr1 = ratios * advantages_batch
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * advantages_batch
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.MSELoss()(values.squeeze(), returns_batch)  # Ensure values are squeezed
                loss = policy_loss + value_loss

                self.optimizer.zero_grad()
                loss.backward(retain_graph=False)  # Ensure retain_graph is not used

                # Apply gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()
                self.scheduler.step()  # Update learning rate

                # Update target network
                if episode % self.target_network_frequency == 0:
                    self.target_model.load_state_dict(self.model.state_dict())

            # Update accumulators for visualization
            actual_rewards.append(episode_rewards)
            predicted_rewards.append([prob.item() for prob in episode_log_probs])
            rewards_per_episode.append(np.mean(episode_rewards))

            # Update the state visit counts
            for state in visited_states:
                state_str = str(state)  # Convert the state to a string to use it as a dictionary key
                if state_str in visited_state_counts:
                    visited_state_counts[state_str] += 1
                else:
                    visited_state_counts[state_str] = 1

            # Append the episode reward to the reward window for moving average calculation
            self.reward_window.append(sum(episode_rewards))
            moving_avg_reward = np.mean(self.reward_window)

            # Prepare the log data
            log_data = {
                "moving_avg_reward": moving_avg_reward,
                "avg_reward": np.mean(episode_rewards),
                "total_reward": np.sum(episode_rewards),
                "exploration_rate": self.exploration_rate
            }

            # Add loss values to log data only if they were computed
            if len(self.replay_memory) >= self.batch_size:
                log_data.update({
                    "policy_loss": policy_loss.item(),
                    "value_loss": value_loss.item(),
                    "loss": loss.item(),
                    "learning_rate": self.optimizer.param_groups[0]['lr']
                })

            # Log all metrics to wandb once
            wandb.log(log_data)

            # Epsilon decay
            # self.exploration_rate = max(self.min_exploration_rate,
            #                             self.exploration_rate - (1 - self.min_exploration_rate) / self.max_episodes)
            self.epsilon = max(self.min_exploration_rate,
                               self.min_exploration_rate + (1.0 - self.min_exploration_rate) * (
                                       1 - (episode / self.max_episodes) ** 2))

            pbar.update(1)
            pbar.set_description(
                f"Policy Loss {float(policy_loss.item())} Value Loss {float(value_loss.item())} Avg R {float(np.mean(episode_rewards))}")

        pbar.close()

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
    Load a saved ActorCriticNetwork model from the subdirectory.

    Args:
    model_directory: Base directory where models are stored.
    agent_type: Type of the agent, used in directory naming.
    run_name: Name of the run, used in directory naming.
    timestamp: Timestamp of the model saving time, used in directory naming.
    input_dim: Input dimension of the model.
    hidden_dim: Hidden layer dimension.
    action_space_nvec: Action space vector size.

    Returns:
    model: The loaded ActorCriticNetwork model, or None if loading failed.
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
    model = ActorCriticNetwork(input_dim, hidden_dim, action_space_nvec)

    # Load the saved model state into the model instance
    model.load_state_dict(torch.load(model_file_path))
    model.eval()  # Set the model to evaluation mode

    return model
