import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import numpy as np
import itertools
from .utilities import load_config
from .visualizer import visualize_all_states, visualize_q_table, visualize_variance_in_rewards_heatmap, \
    visualize_explained_variance, visualize_variance_in_rewards, visualize_infected_vs_community_risk_table, states_visited_viz
import os
import io
import json
import logging
from datetime import datetime
from tqdm import tqdm
import wandb
import random
import pandas as pd
import csv
import math
import collections


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.size = 0
        self.data_pointer = 0

    def add(self, priority, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_idx, priority)

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

        self.size = min(self.size + 1, self.capacity)

    def update(self, tree_idx, priority):
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)

    def _propagate(self, tree_idx, change):
        parent = (tree_idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def get_leaf(self, value):
        parent = 0
        while True:
            left_child = 2 * parent + 1
            right_child = left_child + 1
            if left_child >= len(self.tree):
                leaf = parent
                break
            else:
                if value <= self.tree[left_child]:
                    parent = left_child
                else:
                    value -= self.tree[left_child]
                    parent = right_child
        data_idx = leaf - self.capacity + 1
        return leaf, self.tree[leaf], self.data[data_idx]

    @property
    def total_priority(self):
        return self.tree[0]

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.epsilon = 1e-6

    def add(self, error, sample):
        priority = self._get_priority(error)
        self.tree.add(priority, sample)

    def sample(self, batch_size):
        batch = []
        idxs = []
        segment = self.tree.total_priority / batch_size

        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, priority, data = self.tree.get_leaf(s)
            batch.append(data)
            idxs.append(idx)

        return idxs, batch

    def update(self, idx, error):
        priority = self._get_priority(error)
        self.tree.update(idx, priority)

    def _get_priority(self, error):
        return (error + self.epsilon) ** self.alpha


class QLearningAgent:
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
        self.results_subdirectory = os.path.join(self.results_directory, "q_learning", run_name, timestamp)
        os.makedirs(self.results_subdirectory, exist_ok=True)

        # Set up logging to the correct directory
        log_file_path = os.path.join(self.results_subdirectory, 'agent_log.txt')
        logging.basicConfig(filename=log_file_path, level=logging.INFO)
        # Initialize agent-specific configurations and variables
        self.env = env
        self.run_name = run_name
        self.max_episodes = self.agent_config['agent']['max_episodes']
        self.learning_rate = self.agent_config['agent']['learning_rate']
        self.discount_factor = self.agent_config['agent']['discount_factor']
        self.exploration_rate = self.agent_config['agent']['exploration_rate']
        self.min_exploration_rate = self.agent_config['agent']['min_exploration_rate']
        self.exploration_decay_rate = self.agent_config['agent']['exploration_decay_rate']

        # Parameters for adjusting learning rate over time
        self.learning_rate_decay = self.agent_config['agent']['learning_rate_decay']
        self.min_learning_rate = self.agent_config['agent']['min_learning_rate']

        # Initialize q table
        rows = np.prod(env.observation_space.nvec)
        columns = np.prod(env.action_space.nvec)
        self.q_table = np.zeros((rows, columns))
        # self.q_table = np.random.uniform(low=200, high=300, size=(rows, columns))
        # self.q_table = np.full((rows, columns), 300)

        # Initialize other required variables and structures
        self.training_data = []
        self.possible_actions = [list(range(0, (k))) for k in self.env.action_space.nvec]
        self.possible_states = [list(range(0, (k))) for k in self.env.observation_space.nvec]
        self.all_actions = [str(i) for i in list(itertools.product(*self.possible_actions))]
        self.all_states = [str(i) for i in list(itertools.product(*self.possible_states))]

        self.states = list(itertools.product(*self.possible_states))

        # Initialize state visit counts for count-based exploration
        self.state_visits = np.zeros(rows)

        # moving average for early stopping criteria
        self.moving_average_window = 100  # Number of episodes to consider for moving average
        self.stopping_criterion = 0.01  # Threshold for stopping
        self.prev_moving_avg = -float('inf')  # Initialize to negative infinity to ensure any reward is considered an improvement in the first episode.
        self.state_action_visits = np.zeros((rows, columns))

    def save_q_table(self):
        policy_dir = self.shared_config['directories']['policy_directory']
        if not os.path.exists(policy_dir):
            os.makedirs(policy_dir)

        file_path = os.path.join(policy_dir, f'q_table_{self.run_name}.npy')
        np.save(file_path, self.q_table)
        print(f"Q-table saved to {file_path}")

    # def _policy(self, mode, state):
    #     """Define the policy of the agent."""
    #     global action
    #     if mode == 'train':
    #         if random.uniform(0, 1) > self.exploration_rate:
    #             # print("Non random action selected", self.exploration_rate)
    #             dstate = str(tuple(state))
    #             action = np.argmax(self.q_table[self.all_states.index(dstate)])
    #
    #         else:
    #             sampled_actions = str(tuple(self.env.action_space.sample().tolist()))
    #             # print("sampled action", sampled_actions, self.exploration_rate)
    #
    #             action = self.all_actions.index(sampled_actions)
    #             # print("Action chosen", action)
    #
    #     elif mode == 'test':
    #         dstate = str(tuple(state))
    #         action = np.argmax(self.q_table[self.all_states.index(dstate)])
    #
    #     return action

    def _policy(self, mode, state):
        """Define the policy of the agent."""
        global action
        state_idx = self.all_states.index(str(tuple(state)))

        if mode == 'train':
            if random.uniform(0, 1) > self.exploration_rate:
                q_values = self.q_table[state_idx]
                action = np.argmax(q_values)
            else:
                sampled_actions = str(tuple(self.env.action_space.sample().tolist()))
                action = self.all_actions.index(sampled_actions)

        elif mode == 'test':
            action = np.argmax(self.q_table[state_idx])

        return action

    def polynomial_decay(self, episode, max_episodes, initial_rate, final_rate, power):
        return max(final_rate, initial_rate - (initial_rate - final_rate) * (episode / max_episodes) ** power)

    def linear_decay(self, episode, max_episodes, initial_rate, final_rate):
        return max(final_rate, initial_rate - (initial_rate - final_rate) * (episode / max_episodes))

    def exponential_decay(self, episode, max_episodes, initial_rate, final_rate):
        decay_rate = 0.9999
        return max(final_rate, initial_rate * np.exp(-decay_rate * episode/ max_episodes))

    def train(self, alpha):
        """Train the agent."""
        actual_rewards = []
        predicted_rewards = []
        rewards_per_episode = []
        last_episode = {}
        visited_state_counts = {}
        q_value_history = []
        reward_history = []
        td_errors = []

        # Initialize CSV logging
        csv_file_path = os.path.join(self.results_subdirectory, 'approx-training_log.csv')
        csv_file = open(csv_file_path, mode='w', newline='')
        writer = csv.writer(csv_file)
        # Write headers
        writer.writerow(['Episode', 'Step', 'State', 'Action', 'Reward', 'Next_State', 'Terminated'])

        for episode in tqdm(range(self.max_episodes)):
            # print(f"+-------- Episode: {episode} -----------+")
            state = self.env.reset()
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
            step = 0
            episode_td_errors = []

            while not terminated:
                action = self._policy('train', c_state)
                converted_state = str(tuple(c_state))
                state_idx = self.all_states.index(converted_state)  # Define state_idx here

                list_action = list(eval(self.all_actions[action]))
                c_list_action = [i * 50 for i in list_action] # for 0, 1, 2,

                action_alpha_list = [*c_list_action, alpha]

                # Execute the action and observe the next state and reward
                next_state, reward, terminated, _, info = self.env.step(action_alpha_list)

                # Update the Q-table using the observed reward and the maximum future value
                old_value = self.q_table[self.all_states.index(converted_state), action]
                next_max = np.max(self.q_table[self.all_states.index(str(tuple(next_state)))])
                new_value = (1 - self.learning_rate) * old_value + self.learning_rate * (
                            reward + self.discount_factor * next_max)
                self.q_table[self.all_states.index(converted_state), action] = new_value

                # Calculate TD error
                td_error = abs(reward + self.discount_factor * next_max - old_value)
                episode_td_errors.append(td_error)

                # Store predicted reward (Q-value) for the taken action
                predicted_reward = self.q_table[state_idx, action]
                e_predicted_rewards.append(predicted_reward)

                # Increment the state-action visit count
                self.state_action_visits[state_idx, action] += 1
                self.state_visits[state_idx] += 1

                # Log the experience to CSV
                writer.writerow([episode, step, converted_state, action, reward, str(tuple(next_state)), terminated])
                step += 1
                c_state = next_state
                # Update other accumulators...
                week_reward = int(reward)
                total_reward += week_reward
                e_return.append(week_reward)
                q_value_history.append(np.mean(self.q_table))
                reward_history.append(reward)
                e_allowed.append(info['allowed'])
                e_infected_students.append(info['infected'])
                e_community_risk.append(info['community_risk'])
                if converted_state not in visited_state_counts:
                    visited_state_counts[converted_state] = 1
                else:
                    visited_state_counts[converted_state] += 1



            avg_episode_return = sum(e_return) / len(e_return)
            rewards_per_episode.append(avg_episode_return)
            avg_td_error = np.mean(episode_td_errors)  # Average TD error for this episode
            td_errors.append(avg_td_error)

            # If enough episodes have been run, check for convergence
            if episode >= self.moving_average_window - 1:
                window_rewards = rewards_per_episode[max(0, episode - self.moving_average_window + 1):episode + 1]
                moving_avg = np.mean(window_rewards)
                std_dev = np.std(window_rewards)

                # Store the current moving average for comparison in the next episode
                self.prev_moving_avg = moving_avg

                # Log the moving average and standard deviation along with the episode number
                average_return = total_reward / len(e_return)
                wandb.log({
                    'Moving Average': moving_avg,
                    'Standard Deviation': std_dev,
                    'average_return': average_return,
                    'Exploration Rate': self.exploration_rate,
                    'Learning Rate': self.learning_rate,
                    'Q-value Mean': np.mean(q_value_history[-100:]),
                    'Reward Mean': np.mean(reward_history[-100:]),
                    'TD Error Mean': np.mean(td_errors[-100:])
                })

            predicted_rewards.append(e_predicted_rewards)
            actual_rewards.append(e_return)

            # Exploration Strategies
            # self.exploration_rate = self.polynomial_decay(episode, self.max_episodes, 1.0, self.min_exploration_rate, 2)
            # Quals version
            self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate - (
                    1.0 - self.min_exploration_rate) / self.max_episodes)
            # if episode % 100 == 0:
            # #     decay = (1 - episode / self.max_episodes) ** 2
            # #     self.learning_rate = max(self.min_learning_rate, self.learning_rate * decay)
            #     self.learning_rate = self.exponential_decay(episode, self.max_episodes, self.learning_rate, self.min_learning_rate)

                # self.learning_rate = self.polynomial_decay(episode, self.max_episodes, 1.0, self.min_learning_rate, 2)

            # decay = (1 - episode / self.max_episodes) ** 2
            # self.learning_rate = max(self.min_learning_rate, self.learning_rate * decay)


            # 3.
            # if episode % 100 == 0:
            # Decay rate can be adjusted for different decay speeds
            # decay_rate = 0.001
            # self.learning_rate = max(self.min_learning_rate,
            #                          1.0 * np.exp(-decay_rate * episode))

            # self.learning_rate = max(self.min_learning_rate, self.learning_rate -
            #                          (0.01 - self.min_learning_rate) * (episode / self.max_episodes))

            # self.learning_rate = max(self.min_learning_rate,
            #                              self.learning_rate - (self.learning_rate - self.min_learning_rate) * (
            #                                          episode / self.max_episodes) ** 0.2)
                #
                # # Decay rate can be adjusted for different decay speeds
                # decay_rate = 0.9995
                # self.exploration_rate = max(self.min_exploration_rate,
                #                             self.min_exploration_rate + (1.0 - self.min_exploration_rate) * np.exp(
                #                                 -decay_rate * episode))

                # # Polynomial decay with power `a`
                # a = 2
                # self.exploration_rate = max(self.min_exploration_rate,
                #                             self.min_exploration_rate + (1.0 - self.min_exploration_rate) * (
                #                                         1 - (episode / self.max_episodes) ** a))

            #     self.learning_rate = max(self.min_learning_rate, self.learning_rate * 0.9)
            #     self.exploration_rate = max(self.min_exploration_rate, min(1.0, 1.0 - math.log10((episode + 1) / (self.max_episodes / 10))))


        print("Training complete.")
        self.save_q_table()
        visualize_q_table(self.q_table, self.results_subdirectory, self.max_episodes)

        csv_file.close()
        states = list(visited_state_counts.keys())
        visit_counts = list(visited_state_counts.values())
        states_visited_path = states_visited_viz(states, visit_counts,alpha, self.results_subdirectory)
        wandb.log({"States Visited": [wandb.Image(states_visited_path)]})

        avg_rewards = [sum(lst) / len(lst) for lst in actual_rewards]
        # Pass actual and predicted rewards to visualizer
        explained_variance_path = visualize_explained_variance(actual_rewards, predicted_rewards, self.results_subdirectory, self.max_episodes)
        wandb.log({"Explained Variance": [wandb.Image(explained_variance_path)]})


        # file_path_variance = visualize_variance_in_rewards(avg_rewards, self.results_subdirectory, self.max_episodes)
        # wandb.log({"Variance in Rewards": [wandb.Image(file_path_variance)]})

        # Inside the train method, after training the agent:
        all_states_path = visualize_all_states(self.q_table, self.all_states, self.states, self.run_name, self.max_episodes, alpha,
                            self.results_subdirectory)
        wandb.log({"All_States_Visualization": [wandb.Image(all_states_path)]})

        # file_path_heatmap = visualize_variance_in_rewards_heatmap(rewards_per_episode, self.results_subdirectory, bin_size=50) # 25 for 2500 episodes, 10 for 1000 episodes
        # wandb.log({"Variance in Rewards Heatmap": [wandb.Image(file_path_heatmap)]})

        # print("infected: ", last_episode['infected'], "allowed: ", last_episode['allowed'], "community_risk: ", last_episode['community_risk'])
        file_path_infected_vs_community_risk = visualize_infected_vs_community_risk_table(last_episode, alpha, self.results_subdirectory)
        # wandb.log({"Infected vs Community Risk": [wandb.Image(file_path_infected_vs_community_risk)]})

        return self.q_table

    def test(self, episodes, alpha, baseline_policy=None):
        """Test the trained agent with extended evaluation metrics."""

        total_class_capacity_utilized = 0
        last_action = None
        policy_changes = 0
        total_reward = 0
        rewards = []
        infected_dict = {}
        allowed_dict = {}
        rewards_dict = {}
        community_risk_dict = {}
        eval_dir = 'evaluation'
        if not os.path.exists(eval_dir):
            os.makedirs(eval_dir)

        eval_file_path = os.path.join(eval_dir, f'eval_policies_data_aaai_multi.csv')
        # Check if the file exists already. If not, create it and add the header
        if not os.path.isfile(eval_file_path):
            with open(eval_file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                # Write the header to the CSV file
                writer.writerow(['Alpha', 'Episode', 'Step', 'Infections', 'Allowed', 'Reward', 'CommunityRisk'])

        for episode in tqdm(range(episodes)):
            state = self.env.reset()
            c_state = state[0]
            terminated = False
            episode_reward = 0
            episode_infections = 0
            infected = []
            allowed = []
            community_risk = []
            eps_rewards = []

            while not terminated:
                converted_state = str(tuple(c_state))
                state_idx = self.all_states.index(converted_state)

                # Select an action based on the Q-table or baseline policy
                if baseline_policy:
                    action = baseline_policy(c_state)
                else:
                    action = np.argmax(self.q_table[state_idx])

                print("action", action)
                list_action = list(eval(self.all_actions[action]))
                print("list action", list_action)
                c_list_action = [i * 50 for i in list_action]  # for 0, 1, 2,
                # c_list_action = [i * 25 if i < 3 else 100 for i in list_action]

                action_alpha_list = [*c_list_action, alpha]
                # Execute the action and observe the next state and reward
                next_state, reward, terminated, _, info = self.env.step(action_alpha_list)
                print(info)
                eps_rewards.append(reward)
                infected.append(info['infected'])
                allowed.append(info['allowed'])
                community_risk.append(info['community_risk'])
                episode_infections += sum(info['infected'])

                # Update policy stability metrics
                if last_action is not None and last_action != action:
                    policy_changes += 1
                last_action = action

                # Update class utilization metrics
                total_class_capacity_utilized += sum(info['allowed'])

                # Update the state to the next state
                c_state = next_state

            infected_dict[episode] = infected
            allowed_dict[episode] = allowed
            rewards_dict[episode] = eps_rewards
            community_risk_dict[episode] = community_risk




        print("infected: ", infected_dict, "allowed: ", allowed_dict, "rewards: ", rewards_dict, "community_risk: ", community_risk_dict)
        for episode in infected_dict:
            plt.figure(figsize=(15, 5))

            # Flatten the list of lists for infections and allowed students
            infections = [inf[0] for inf in infected_dict[episode]] if episode in infected_dict else []
            allowed_students = [alw[0] for alw in allowed_dict[episode]] if episode in allowed_dict else []
            rewards = rewards_dict[episode] if episode in rewards_dict else []
            community_risk = community_risk_dict[episode] if episode in community_risk_dict else []

            # Convert range to numpy array for element-wise operations
            steps = np.arange(len(infections))

            # Define bar width and offset
            bar_width = 0.4
            offset = bar_width / 4

            # Bar plot for infections
            plt.bar(steps - offset, infections, width=bar_width, label='Infections', color='#bc5090', align='center')

            # Bar plot for allowed students
            plt.bar(steps + offset, allowed_students, width=bar_width, label='Allowed Students', color='#003f5c',
                    alpha=0.5, align='edge')

            # Line plot for rewards
            plt.plot(steps, rewards, label='Rewards', color='#ffa600', linestyle='-', marker='o')

            plt.xlabel('Step')
            plt.ylabel('Count')
            plt.title(f'Evaluation of agent {self.run_name} Policy for {episode} episodes')
            plt.legend()

            plt.tight_layout()

            # Save the figure
            fig_path = os.path.join(eval_dir, f'{self.run_name}_metrics.png')
            plt.savefig(fig_path)
            print(f"Figure saved to {fig_path}")

            plt.close()  # Close the figure to free up memory

        # # Calculate additional metrics if needed
        # # For example, average infections, average rewards, etc.
        # average_infections = sum(sum(inf) for inf in infected_dict.values()) / episodes
        # average_rewards = sum(sum(rew) for rew in rewards_dict.values()) / episodes
        #
        # summary_data = {
        #         'Average Infections': average_infections,
        #         'Average Rewards': average_rewards,
        #         'Policy Stability': (episodes - policy_changes) / episodes
        #     }
        # summary_table = pd.DataFrame(summary_data, index=[0])
        # print(summary_table)

        with open(eval_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)

            # Iterate over each episode and step to append the data
            for episode in tqdm(range(episodes)):
                for step in range(len(infected_dict[episode])):
                    writer.writerow([
                        alpha,
                        episode,
                        step,
                        infected_dict[episode][step],
                        allowed_dict[episode][step],
                        rewards_dict[episode][step],
                        community_risk_dict[episode][step]

                    ])

        print(f"Data for alpha {alpha} appended to {eval_file_path}")

        return infected_dict, allowed_dict, rewards_dict, community_risk_dict

    def test_baseline_random(self, episodes, alpha, baseline_policy=None):
        """Test the trained agent with extended evaluation metrics."""

        total_class_capacity_utilized = 0
        last_action = None
        policy_changes = 0
        total_reward = 0
        rewards = []
        infected_dict = {}
        allowed_dict = {}
        rewards_dict = {}
        community_risk_dict = {}
        eval_dir = 'evaluation'
        r_alpha = alpha * 100
        if not os.path.exists(eval_dir):
            os.makedirs(eval_dir)
        eval_file_path = os.path.join(eval_dir, f'eval_policies_data_aaai_multi.csv')
        # Check if the file exists already. If not, create it and add the header
        if not os.path.isfile(eval_file_path):
            with open(eval_file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                # Write the header to the CSV file
                writer.writerow(['Alpha', 'Episode', 'Step', 'Infections', 'Allowed', 'Reward', 'CommunityRisk'])

        for episode in tqdm(range(episodes)):
            state = self.env.reset()
            c_state = state[0]
            terminated = False
            episode_reward = 0
            episode_infections = 0
            infected = []
            allowed = []
            eps_rewards = []
            community_risk = []

            while not terminated:
                converted_state = str(tuple(c_state))
                state_idx = self.all_states.index(converted_state)

                # Select a random action
                sampled_actions = str(tuple(self.env.action_space.sample().tolist()))
                # print("sampled action", sampled_actions, self.exploration_rate)

                action = self.all_actions.index(sampled_actions)
                list_action = list(eval(self.all_actions[action]))
                c_list_action = [i * 50 for i in list_action]  # for 0, 1, 2,

                # c_list_action = [i * 25 if i < 3 else 100 for i in list_action]

                action_alpha_list = [*c_list_action, alpha]
                # Execute the action and observe the next state and reward
                next_state, reward, terminated, _, info = self.env.step(action_alpha_list)
                print(info)
                eps_rewards.append(reward)
                infected.append(info['infected'])
                allowed.append(info['allowed'])
                episode_infections += sum(info['infected'])
                community_risk.append(info['community_risk'])

                # Update policy stability metrics
                if last_action is not None and last_action != action:
                    policy_changes += 1
                last_action = action

                # Update class utilization metrics
                total_class_capacity_utilized += sum(info['allowed'])

                # Update the state to the next state
                c_state = next_state

            infected_dict[episode] = infected
            allowed_dict[episode] = allowed
            rewards_dict[episode] = eps_rewards
            community_risk_dict[episode] = community_risk

        print("infected: ", infected_dict, "allowed: ", allowed_dict, "rewards: ", rewards_dict, "community_risk: ", community_risk_dict)
        for episode in infected_dict:
            plt.figure(figsize=(15, 5))

            # Flatten the list of lists for infections and allowed students
            infections = [inf[0] for inf in infected_dict[episode]] if episode in infected_dict else []
            allowed_students = [alw[0] for alw in allowed_dict[episode]] if episode in allowed_dict else []
            rewards = rewards_dict[episode] if episode in rewards_dict else []

            # Convert range to numpy array for element-wise operations
            steps = np.arange(len(infections))

            # Define bar width and offset
            bar_width = 0.4
            offset = bar_width / 4

            # Bar plot for infections
            plt.bar(steps - offset, infections, width=bar_width, label='Infections', color='#bc5090', align='center')

            # Bar plot for allowed students
            plt.bar(steps + offset, allowed_students, width=bar_width, label='Allowed Students', color='#003f5c',
                    alpha=0.5, align='edge')

            # Line plot for rewards
            plt.plot(steps, rewards, label='Rewards', color='#ffa600', linestyle='-', marker='o')

            plt.xlabel('Step')
            plt.ylabel('Count')
            plt.title(f'Evaluation of Random Agent Policy for {episode + 1} episode(s)')
            plt.legend()

            plt.tight_layout()

            # Save the figure
            fig_path = os.path.join(eval_dir, f'episode_random_agent_metrics.png')
            plt.savefig(fig_path)
            print(f"Figure saved to {fig_path}")

            plt.close()  # Close the figure to free up memory

        with open(eval_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)

            # Iterate over each episode and step to append the data
            for episode in tqdm(range(episodes)):
                for step in range(len(infected_dict[episode])):
                    writer.writerow([
                        0.0,
                        episode,
                        step,
                        infected_dict[episode][step],
                        allowed_dict[episode][step],
                        rewards_dict[episode][step],
                        community_risk_dict[episode][step]
                    ])

        print(f"Data for alpha {0.0} appended to {eval_file_path}")