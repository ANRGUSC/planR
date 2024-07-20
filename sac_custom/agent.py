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
from torch.optim.lr_scheduler import StepLR
from torch.distributions import Normal
from torch.nn.utils import clip_grad_norm_

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
set_seed(100)

class ActorNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super(ActorNetwork, self).__init__()
        self.hidden_dim = hidden_dim

        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

        # Initialize weights
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        shared_output = self.shared_layers(x)
        logits = self.actor(shared_output)
        return logits

    def get_action(self, x):
        logits = self(x)
        policy_dist = torch.softmax(logits, dim=-1)
        logger.info(f"Policy Distribution in get_action: {policy_dist}")
        action = torch.multinomial(policy_dist, 1)
        log_prob = torch.log(policy_dist.gather(1, action) + 1e-9)
        value = None  # Assuming value is not needed or computed here
        return action, log_prob, policy_dist, value

    def print_gradients(self):
        for name, param in self.named_parameters():
            if param.grad is not None:
                print(f"Gradients for {name}: {param.grad}")
            else:
                print(f"No gradients for {name}")

class QNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + action_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, action):
        # Ensure action has the same number of dimensions as x
        if len(action.shape) == 1:
            action = action.unsqueeze(-1)
        return self.net(torch.cat([x, action], dim=-1))

class ValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ValueNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)

class SACCustomAgent:
    def __init__(self, env, run_name, shared_config_path, agent_config_path=None, override_config=None):
        self.shared_config = load_config(shared_config_path)

        if agent_config_path:
            self.agent_config = load_config(agent_config_path)
        else:
            self.agent_config = {}

        if override_config:
            self.agent_config.update(override_config)

        self.results_directory = self.shared_config['directories']['results_directory']
        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.agent_type = "sac_custom"
        self.run_name = run_name
        self.results_subdirectory = os.path.join(self.results_directory, self.agent_type, self.run_name, self.timestamp)
        os.makedirs(self.results_subdirectory, exist_ok=True)
        self.model_directory = self.shared_config['directories']['model_directory']
        self.model_subdirectory = os.path.join(self.model_directory, self.agent_type, self.run_name, self.timestamp)
        os.makedirs(self.model_subdirectory, exist_ok=True)

        log_file_path = os.path.join(self.results_subdirectory, 'agent_log.txt')
        logging.basicConfig(filename=log_file_path, level=logging.INFO)

        wandb.init(project=self.agent_type, name=self.run_name)

        self.input_dim = len(env.reset()[0])
        self.action_dim = env.action_space.shape[0]
        self.hidden_dim = self.agent_config['agent']['hidden_units']

        self.actor = ActorNetwork(self.input_dim, self.hidden_dim, self.action_dim)
        self.q1 = QNetwork(self.input_dim, self.hidden_dim, self.action_dim)
        self.q2 = QNetwork(self.input_dim, self.hidden_dim, self.action_dim)
        self.value = ValueNetwork(self.input_dim, self.hidden_dim)
        self.target_value = ValueNetwork(self.input_dim, self.hidden_dim)
        self.target_value.load_state_dict(self.value.state_dict())
        self.target_q1 = QNetwork(self.input_dim, self.hidden_dim, self.action_dim)  # Add this line
        self.target_q2 = QNetwork(self.input_dim, self.hidden_dim, self.action_dim)  # Add this line

        # Initialize target networks with the same weights
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.agent_config['agent']['learning_rate'])
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=self.agent_config['agent']['learning_rate'])
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=self.agent_config['agent']['learning_rate'])
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=self.agent_config['agent']['learning_rate'])

        self.env = env
        self.max_episodes = self.agent_config['agent']['max_episodes']
        self.discount_factor = self.agent_config['agent']['discount_factor']
        self.exploration_rate = self.agent_config['agent']['exploration_rate']
        self.min_exploration_rate = self.agent_config['agent']['min_exploration_rate']
        self.exploration_decay_rate = self.agent_config['agent']['exploration_decay_rate']
        self.target_network_frequency = self.agent_config['agent']['target_network_frequency']
        self.alpha = self.agent_config['agent']['alpha']
        self.tau = self.agent_config['agent']['tau']  # Add this line

        self.replay_memory = deque(maxlen=self.agent_config['agent']['replay_memory_capacity'])
        self.batch_size = self.agent_config['agent']['batch_size']

        self.possible_actions = [list(range(0, (k))) for k in self.env.action_space.shape]
        self.all_actions = [str(i) for i in list(itertools.product(*self.possible_actions))]

        self.moving_average_window = 100
        self.stopping_criterion = 0.01
        self.prev_moving_avg = -float('inf')

        self.hidden_state = None
        self.reward_window = deque(maxlen=self.moving_average_window)
        self.scheduler = StepLR(self.actor_optimizer, step_size=100, gamma=0.9)

        # Additional attributes
        self.qf1 = self.q1  # Ensuring qf1 is correctly referenced
        self.qf2 = self.q2  # Ensuring qf2 is correctly referenced
        self.q_optimizer = optim.Adam(itertools.chain(self.q1.parameters(), self.q2.parameters()),
                                      lr=self.agent_config['agent']['learning_rate'])
        self.autotune = self.agent_config.get('autotune', False)
        self.target_entropy = -np.prod(env.action_space.shape)
        self.log_alpha = torch.tensor(np.log(self.alpha)).requires_grad_()
        self.a_optimizer = optim.Adam([self.log_alpha], lr=self.agent_config['agent']['learning_rate'])
        self.max_grad_norm = self.agent_config['agent']['max_grad_norm']

    def update_replay_memory(self, state, action, reward, next_state, done, log_prob, value):
        # Convert to appropriate types before appending to replay memory
        state = np.array(state, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        action = int(action)
        reward = float(reward)
        done = bool(done)
        self.replay_memory.append((state, action, reward, next_state, done, log_prob, value))

    def compute_q_values(self, states, actions):
        q1_values = self.q1(states, actions)
        q2_values = self.q2(states, actions)
        return q1_values, q2_values

    def compute_targets(self, rewards, next_states, dones):
        with torch.no_grad():
            next_states = torch.FloatTensor(next_states)
            target_values = self.target_value(next_states).squeeze(-1)
            targets = rewards + (1 - dones) * self.discount_factor * target_values
        return targets

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def train(self, alpha):
        pbar = tqdm(total=self.max_episodes, desc="Training Progress", leave=True)
        decay_rate = np.log(self.min_exploration_rate / self.exploration_rate) / self.max_episodes

        actual_rewards = []
        predicted_rewards = []
        rewards_per_episode = []
        visited_state_counts = {}
        # Initialize accumulators for visualization
        actor_loss = None
        q_loss = None
        total_steps = 0
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
                total_steps += 1
                # print("Calling get_action method...")
                action, log_prob, policy_dist, value = self.actor.get_action(state.unsqueeze(0))

                # Check for NaN or Inf values immediately
                if torch.any(torch.isnan(policy_dist)) or torch.any(torch.isinf(policy_dist)):
                    print(f"NaN or Inf detected in policy_dist at episode {episode}")
                    print(f"policy_dist: {policy_dist}")
                    raise ValueError("policy_dist contains NaN or Inf values")

                # Clamp policy_dist to ensure all elements are within valid range
                policy_dist = torch.clamp(policy_dist, min=1e-9, max=1.0 - 1e-9)

                # Normalize policy_dist to ensure it sums to 1
                policy_dist = policy_dist / policy_dist.sum()

                next_state, reward, terminated, _, info = self.env.step(([action * 50], alpha))
                # print(f"Action: {action.item()}, Reward: {reward}")

                episode_rewards.append(reward)
                episode_log_probs.append(log_prob.item())  # Store the log_prob
                episode_values.append(value.item() if value is not None else 0)  # Store the value
                episode_dones.append(terminated)

                visited_states.append(state.tolist())  # Add the current state to the list of visited states

                self.update_replay_memory(state, action, reward, next_state, terminated, log_prob.item(),
                                          value.item() if value is not None else 0)

                state = torch.FloatTensor(next_state)

            # Training step
            if len(self.replay_memory) >= self.batch_size:
                minibatch = random.sample(self.replay_memory, self.batch_size)
                states, actions, rewards_batch, next_states, dones, log_probs, values = zip(*minibatch)
                states = torch.FloatTensor(states)
                actions = torch.FloatTensor(actions).view(-1, 1)  # Ensure actions have the correct shape
                rewards_batch = torch.FloatTensor(rewards_batch)  # Corrected this line
                next_states = torch.FloatTensor(next_states)
                dones = torch.FloatTensor(dones)

                # SAC training logic here
                with torch.no_grad():
                    next_actions, next_log_probs, next_policy_dist, next_value = self.actor.get_action(next_states)
                    next_q1_values = self.target_q1(next_states, next_actions)  # Use target network
                    next_q2_values = self.target_q2(next_states, next_actions)  # Use target network
                    min_next_q_values = torch.min(next_q1_values, next_q2_values) - self.alpha * next_log_probs
                    q_targets = rewards_batch + self.discount_factor * (1 - dones) * min_next_q_values

                q1_values = self.q1(states, actions)
                q2_values = self.q2(states, actions)
                q1_loss = nn.functional.mse_loss(q1_values, q_targets)
                q2_loss = nn.functional.mse_loss(q2_values, q_targets)
                q_loss = q1_loss + q2_loss

                print(f"Q1 Loss: {q1_loss.item()}, Q2 Loss: {q2_loss.item()}, Q Loss: {q_loss.item()}")

                self.q_optimizer.zero_grad()
                q_loss.backward()
                # Add gradient clipping here
                clip_grad_norm_(self.q1.parameters(), self.max_grad_norm)
                clip_grad_norm_(self.q2.parameters(), self.max_grad_norm)
                self.q_optimizer.step()

                # Update actor network
                actions, log_probs, policy_dist, value = self.actor.get_action(states)
                q1_values = self.q1(states, actions)
                q2_values = self.q2(states, actions)
                min_q_values = torch.min(q1_values, q2_values)
                actor_loss = (self.alpha * log_probs - min_q_values).mean()

                print(f"Actor Loss: {actor_loss.item()}")

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                # self.actor.print_gradients()  # Add this line
                # Add gradient clipping here
                clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                if self.autotune:
                    alpha_loss = -(self.log_alpha.exp() * (log_probs + self.target_entropy).detach()).mean()
                    print(f"Alpha Loss: {alpha_loss.item()}")
                    self.a_optimizer.zero_grad()
                    alpha_loss.backward()
                    self.a_optimizer.step()
                    self.alpha = self.log_alpha.exp()

                # Soft update target networks
                if total_steps % self.target_network_frequency == 0:
                    self.soft_update(self.q1, self.target_q1)
                    self.soft_update(self.q2, self.target_q2)

            log_data = {
                "moving_avg_reward": np.mean(self.reward_window),
                "episode": episode,
                "avg_reward": np.mean(episode_rewards),
                "total_reward": np.sum(episode_rewards),
                "exploration_rate": self.exploration_rate
            }

            if actor_loss is not None and q_loss is not None:
                log_data.update({
                    "actor_loss": actor_loss.item(),
                    "critic_loss": q_loss.item()
                })

            wandb.log(log_data)

            pbar.update(1)
            pbar.set_description(
                f"Actor Loss {float(actor_loss.item()) if actor_loss is not None else 'N/A'} "
                f"Critic Loss {float(q_loss.item()) if q_loss is not None else 'N/A'} "
                f"Avg R {float(np.mean(episode_rewards))}")

        pbar.close()

        model_file_path = os.path.join(self.model_subdirectory, 'model.pt')
        torch.save(self.actor.state_dict(), model_file_path)

        states = list(visited_state_counts.keys())
        visit_counts = list(visited_state_counts.values())
        states_visited_path = states_visited_viz(states, visit_counts, alpha, self.results_subdirectory)
        wandb.log({"States Visited": [wandb.Image(states_visited_path)]})

        avg_rewards = [sum(lst) / len(lst) for lst in actual_rewards]
        explained_variance_path = visualize_explained_variance(actual_rewards, predicted_rewards,
                                                               self.results_subdirectory, self.max_episodes)
        wandb.log({"Explained Variance": [wandb.Image(explained_variance_path)]})

        if avg_rewards:
            file_path_variance = visualize_variance_in_rewards(avg_rewards, self.results_subdirectory,
                                                               self.max_episodes)
            wandb.log({"Variance in Rewards": [wandb.Image(file_path_variance)]})
        else:
            print("Warning: No rewards to visualize")

        saved_model = load_saved_model(self.model_directory, self.agent_type, self.run_name, self.timestamp,
                                       self.input_dim, self.hidden_dim, self.action_dim)
        if saved_model is not None:
            value_range = range(0, 101, 10)
            all_states = [np.array([i, j]) for i in value_range for j in value_range]
            all_states_path = visualize_all_states(saved_model, all_states, self.run_name, self.max_episodes, alpha,
                                                   self.results_subdirectory)
            wandb.log({"All_States_Visualization": [wandb.Image(all_states_path)]})
        else:
            print("Warning: Could not load saved model for visualization")

        return self.actor



def load_saved_model(model_directory, agent_type, run_name, timestamp, input_dim, hidden_dim, action_dim):
    model_subdirectory = os.path.join(model_directory, agent_type, run_name, timestamp)
    model_file_path = os.path.join(model_subdirectory, 'model.pt')

    if not os.path.exists(model_file_path):
        print(f"Model file not found in {model_file_path}")
        return None

    model = ActorNetwork(input_dim, hidden_dim, action_dim)
    model.load_state_dict(torch.load(model_file_path))
    model.eval()

    return model
