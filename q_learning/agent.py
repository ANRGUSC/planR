import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
        agent_type = "q_learning"
        self.results_subdirectory = os.path.join(self.results_directory, agent_type, run_name, timestamp)
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

    def _policy(self, mode, state):
        """Define the policy of the agent."""
        global action
        if mode == 'train':
            if random.uniform(0, 1) > self.exploration_rate:
                # print("Non random action selected", self.exploration_rate)
                dstate = str(tuple(state))
                action = np.argmax(self.q_table[self.all_states.index(dstate)])

            else:
                sampled_actions = str(tuple(self.env.action_space.sample().tolist()))
                # print("sampled action", sampled_actions, self.exploration_rate)

                action = self.all_actions.index(sampled_actions)
                # print("Action chosen", action)

        elif mode == 'test':
            dstate = str(tuple(state))
            action = np.argmax(self.q_table[self.all_states.index(dstate)])

        return action

    def train(self, alpha):
        """Train the agent."""
        print("Training with alpha: ", alpha)
        # reset Q table
        # rows = np.prod(self.env.observation_space.nvec)
        # columns = np.prod(self.env.action_space.nvec)
        # self.q_table = np.zeros((rows, columns))
        actual_rewards = []
        predicted_rewards = []
        rewards = []
        rewards_per_episode = []
        last_episode = {}
        # Initialize visited state counts dictionary
        visited_state_counts = {}

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

            while not terminated:
                # Select an action using the current state and the policy
                print("current state", c_state)
                action = self._policy('train', c_state)
                converted_state = str(tuple(c_state))
                print("converted state", converted_state)
                state_idx = self.all_states.index(converted_state)  # Define state_idx here

                list_action = list(eval(self.all_actions[action]))
                c_list_action = [i * 50 for i in list_action] # for 0, 1, 2,
                # c_list_action = [i * 25 if i < 3 else 100 for i in list_action]

                action_alpha_list = [*c_list_action, alpha]

                # Execute the action and observe the next state and reward
                next_state, reward, terminated, _, info = self.env.step(action_alpha_list)

                # Update the Q-table using the observed reward and the maximum future value
                old_value = self.q_table[self.all_states.index(converted_state), action]
                next_max = np.max(self.q_table[self.all_states.index(str(tuple(next_state)))])
                new_value = (1 - self.learning_rate) * old_value + self.learning_rate * (
                            reward + self.discount_factor * next_max)
                self.q_table[self.all_states.index(converted_state), action] = new_value

                # Store predicted reward (Q-value) for the taken action
                predicted_reward = self.q_table[state_idx, action]
                e_predicted_rewards.append(predicted_reward)

                # Increment the state-action visit count
                self.state_action_visits[state_idx, action] += 1

                # Update the state to the next state
                print("next state", next_state)
                c_state = next_state

                # Update other accumulators...
                week_reward = int(reward)
                total_reward += week_reward
                e_return.append(week_reward)
                e_allowed.append(info['allowed'])
                e_infected_students.append(info['infected'])
                e_community_risk.append(info['community_risk'])
                if converted_state not in visited_state_counts:
                    visited_state_counts[converted_state] = 1
                else:
                    visited_state_counts[converted_state] += 1
                print(info)

                # Log state, action, and Q-values.
                logging.info(f"State: {state}, Action: {action}, Q-values: {self.q_table[state_idx, :]}")

            avg_episode_return = sum(e_return) / len(e_return)
            rewards_per_episode.append(avg_episode_return)
            if episode % self.agent_config['agent']['checkpoint_interval'] == 0:
                checkpoint_path = os.path.join(self.results_subdirectory, f"qtable-{episode}-qtable.npy")
                np.save(checkpoint_path, self.q_table)

                # Call the visualizers functions here
                visualize_q_table(self.q_table, self.results_subdirectory, episode)
                # Example usage:
            # If enough episodes have been run, check for convergence
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

            logging.info(f"Episode: {episode}, Length: {len(e_return)}, Cumulative Reward: {sum(e_return)}, "
                         f"Exploration Rate: {self.exploration_rate}")

            # Render the environment at the end of each episode
            if episode % self.agent_config['agent']['checkpoint_interval'] == 0:
                self.env.render()
            predicted_rewards.append(e_predicted_rewards)
            actual_rewards.append(e_return)
            self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate - (
                        1.0 - self.min_exploration_rate) / self.max_episodes) # use this for approximate sir model including the learning rate decay
            decay = (1 - episode / self.max_episodes) ** 2
            self.learning_rate = max(self.min_learning_rate, self.learning_rate * decay)

            # decay = self.learning_rate_decay ** (episode / self.max_episodes)
            # self.learning_rate = max(self.min_learning_rate, self.learning_rate * decay)

            #self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * (self.exploration_decay_rate ** episode))

        print("Training complete.")
        print("all states", self.all_states)
        print("states", self.states)
        self.save_q_table()
        states = list(visited_state_counts.keys())
        visit_counts = list(visited_state_counts.values())
        states_visited_path = states_visited_viz(states, visit_counts,alpha, self.results_subdirectory)
        wandb.log({"States Visited": [wandb.Image(states_visited_path)]})

        avg_rewards = [sum(lst) / len(lst) for lst in actual_rewards]
        # Pass actual and predicted rewards to visualizer
        explained_variance_path = visualize_explained_variance(actual_rewards, predicted_rewards, self.results_subdirectory, self.max_episodes)
        wandb.log({"Explained Variance": [wandb.Image(explained_variance_path)]})


        file_path_variance = visualize_variance_in_rewards(avg_rewards, self.results_subdirectory, self.max_episodes)
        wandb.log({"Variance in Rewards": [wandb.Image(file_path_variance)]})

        # Inside the train method, after training the agent:
        all_states_path = visualize_all_states(self.q_table, self.all_states, self.states, self.run_name, self.max_episodes, alpha,
                            self.results_subdirectory)
        wandb.log({"All_States_Visualization": [wandb.Image(all_states_path)]})

        file_path_heatmap = visualize_variance_in_rewards_heatmap(rewards_per_episode, self.results_subdirectory, bin_size=10) # 25 for 2500 episodes, 10 for 1000 episodes
        wandb.log({"Variance in Rewards Heatmap": [wandb.Image(file_path_heatmap)]})

        print("infected: ", last_episode['infected'], "allowed: ", last_episode['allowed'], "community_risk: ", last_episode['community_risk'])
        file_path_infected_vs_community_risk = visualize_infected_vs_community_risk_table(last_episode, alpha, self.results_subdirectory)
        wandb.log({"Infected vs Community Risk": [wandb.Image(file_path_infected_vs_community_risk)]})

        return self.q_table

    def test_lyapunov(self, episodes, alpha, baseline_policy=None):
        """Test the trained agent with extended evaluation metrics.This function was used for 699 coursework.
        """

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
        # Define a threshold for how much the number of infections can vary to be considered at equilibrium
        delta = 5  # This is an example value and should be set according to the specific context
        if not os.path.exists(eval_dir):
            os.makedirs(eval_dir)

        eval_file_path = os.path.join(eval_dir, f'eval_policies_data_699.csv')
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

            # At the end of each episode, analyze the infection data to identify potential equilibrium points
            equilibrium_points = self.analyze_equilibrium(infected_dict, delta)

            # Plot the results for the episode including equilibrium points
            self.plot_results(episode, infected_dict, allowed_dict, rewards_dict, community_risk_dict, equilibrium_points, eval_dir)

        print("infected: ", infected_dict, "allowed: ", allowed_dict, "rewards: ", rewards_dict, "community_risk: ", community_risk_dict)

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

    def analyze_equilibrium(self, infected_dict, delta):
        for episode in infected_dict:
            plt.figure(figsize=(15, 5))

            # Flatten the list of lists for infections and allowed students
            infected = [inf[0] for inf in infected_dict[episode]] if episode in infected_dict else []
            equilibrium_points = []
            print(f"Analyzing equilibrium for {infected}")
            # Assuming each infected_list[step] is a number representing the count of infected individuals at that step
            for step in range(1, len(infected)):
                if abs(infected[step] - infected[step - 1]) <= delta:
                    equilibrium_points.append(infected[step])
                else:
                    equilibrium_points.append(None)  # Indicate no equilibrium at this step
            print(f"Equilibrium points: {equilibrium_points}")
            return equilibrium_points

    def plot_results(self, episode, infected_dict, allowed_dict, rewards_dict, community_risk_dict, equilibrium_points, eval_dir):
        infection = []
        comm_risk = []
        allowed = []
        for episode in infected_dict:
            plt.figure(figsize=(15, 5))

            # Flatten the list of lists for infections and allowed students
            infections = [inf[0] for inf in infected_dict[episode]] if episode in infected_dict else []
            infection = infections
            allowed_students = [alw[0] for alw in allowed_dict[episode]] if episode in allowed_dict else []
            allowed = allowed_students
            rewards = rewards_dict[episode] if episode in rewards_dict else []
            community_risk = community_risk_dict[episode] if episode in community_risk_dict else []
            comm_risk = community_risk
            equilibrium_steps = [step for step, eq in enumerate(equilibrium_points) if eq is not None]
            equilibrium_infected = [infections[step] for step in equilibrium_steps]
            plt.scatter(equilibrium_steps, equilibrium_infected, color='green', label='Equilibrium', zorder=5)

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
            fig_path = os.path.join(eval_dir, f'{self.run_name}_metrics_699.png')
            plt.savefig(fig_path)
            print(f"Figure saved to {fig_path}")
              # Close the figure to free up memory
        # Identify equilibrium points where the rate of change is close to zero
        infected_rate = [0] + self.calculate_rate_of_change(infection)
        # Define a threshold for what you consider 'close to zero'
        # Define a threshold for the rate of change to be considered in equilibrium
        equilibrium_threshold = 4
        plt.figure(figsize=(10, 6))
        # Color code for actions
        # Color code for actions
        action_colors = {0: 'green', 50: 'orange', 100: 'purple'}

        # Plot all states with colors based on actions
        for i, (cr, rate, act) in enumerate(zip(comm_risk, infected_rate, allowed)):
            plt.scatter(cr, rate, c=action_colors[act])

        # Create a scatter plot for all points
        # plt.scatter(comm_risk, infected_rate, c='blue', label='All States')

        # Identify and label equilibrium points (where the rate of change is close to zero)
        equilibrium_points = [i for i, roc in enumerate(infected_rate) if abs(roc) < equilibrium_threshold]
        for eq in equilibrium_points:
            plt.annotate('Eq', (comm_risk[eq], infected_rate[eq]), textcoords="offset points", xytext=(0, 10),
                         ha='center', fontsize=9)

        # Draw arrows between consecutive points to show transitions
        for i in range(1, len(comm_risk)):
            plt.annotate('', xy=(comm_risk[i], infected_rate[i]),
                         xytext=(comm_risk[i - 1], infected_rate[i - 1]),
                         arrowprops=dict(arrowstyle="->", color='gray'))

        # Add labels to each point to indicate the order of the steps
        for i, (cr, roc) in enumerate(zip(comm_risk, infected_rate)):
            plt.text(cr, roc, f'{i}', fontsize=9, ha='right')

        plt.xlabel('Community Risk')
        plt.ylabel('Rate of Change of Infected Individuals')
        plt.title('Scatter Plot with Transitions and Equilibrium Points')
        plt.legend()
        plt.grid(True)
        # Create a list of patches for the legend
        legend_patches = [mpatches.Patch(color=color, label=f'Action {action}') for action, color in
                          action_colors.items()]

        # Add the legend to the plot
        plt.legend(handles=legend_patches)
        new_fig_path = os.path.join(eval_dir, f'{self.run_name}_metrics_phase.png')
        plt.savefig(new_fig_path)
        print(f"Figure saved to {new_fig_path}")
        plt.close()

        # # Highlight equilibrium points
        # for eq_index in equilibrium_indices:
        #     plt.plot(comm_risk[eq_index], infected_rate[eq_index], 'ro', markersize=10,
        #              label='Equilibrium' if eq_index == equilibrium_indices[0] else "")
        #
        # # Annotate the points with the time step for clarity
        # for i, (cr, roc) in enumerate(zip(comm_risk, infected_rate)):
        #     plt.annotate(f'{i}', (cr, roc), textcoords="offset points", xytext=(0, 10), ha='center')
        #
        # plt.xlabel('Community Risk')
        # plt.ylabel('Rate of Change of Infected Individuals')
        # plt.title('Temporal Sequence with Equilibrium Points')
        # plt.grid(True)
        # plt.legend()
        # new_fig_path = os.path.join(eval_dir, f'{self.run_name}_metrics_phase.png')
        # plt.savefig(new_fig_path)
        # print(f"Figure saved to {new_fig_path}")
        # plt.close()

        # plt.figure(figsize=(10, 6))
        # infected_rate = [0] + self.calculate_rate_of_change(infection)
        # # Creating a scatter plot or line plot depending on your data structure
        # print(len(infected_rate), len(comm_risk))
        # # plt.scatter(comm_risk, infected_rate, c='blue', label='Infection Rate of Change')
        # plt.plot(comm_risk, infected_rate, '-o', c='blue', label='Infection Rate of Change')
        # # Annotate the points with the time step for clarity
        # for i, (cr, roc) in enumerate(zip(comm_risk, infected_rate)):
        #     plt.annotate(f'Time {i}', (cr, roc), textcoords="offset points", xytext=(0, 10), ha='center')
        #
        # plt.xlabel('Community Risk')
        # plt.ylabel('Rate of Change of Infected Individuals')
        # plt.title('Phase Plot of the System')
        # plt.legend()
        # new_fig_path = os.path.join(eval_dir, f'{self.run_name}_metrics_phase.png')
        # plt.savefig(new_fig_path)
        # print(f"Figure saved to {new_fig_path}")
        # plt.close()


    def calculate_rate_of_change(self,infected_list):
        # Initialize an empty list to store the rate of change
        rate_of_change = []

        # Iterate over the infected list to calculate the rate of change
        for i in range(1, len(infected_list)):
            change = infected_list[i] - infected_list[i - 1]
            rate_of_change.append(change)

        return rate_of_change



    # def test(self, episodes, alpha, baseline_policy=None):
    #     """Test the trained agent with extended evaluation metrics.This function was used for the quals and AI4ED paper.
    #     """
    #
    #     total_class_capacity_utilized = 0
    #     last_action = None
    #     policy_changes = 0
    #     total_reward = 0
    #     rewards = []
    #     infected_dict = {}
    #     allowed_dict = {}
    #     rewards_dict = {}
    #     community_risk_dict = {}
    #     eval_dir = 'evaluation'
    #     if not os.path.exists(eval_dir):
    #         os.makedirs(eval_dir)
    #
    #     eval_file_path = os.path.join(eval_dir, f'eval_policies_data_aaai_multi.csv')
    #     # Check if the file exists already. If not, create it and add the header
    #     if not os.path.isfile(eval_file_path):
    #         with open(eval_file_path, mode='w', newline='') as file:
    #             writer = csv.writer(file)
    #             # Write the header to the CSV file
    #             writer.writerow(['Alpha', 'Episode', 'Step', 'Infections', 'Allowed', 'Reward', 'CommunityRisk'])
    #
    #     for episode in tqdm(range(episodes)):
    #         state = self.env.reset()
    #         c_state = state[0]
    #         terminated = False
    #         episode_reward = 0
    #         episode_infections = 0
    #         infected = []
    #         allowed = []
    #         community_risk = []
    #         eps_rewards = []
    #
    #         while not terminated:
    #             converted_state = str(tuple(c_state))
    #             state_idx = self.all_states.index(converted_state)
    #
    #             # Select an action based on the Q-table or baseline policy
    #             if baseline_policy:
    #                 action = baseline_policy(c_state)
    #             else:
    #                 action = np.argmax(self.q_table[state_idx])
    #
    #             print("action", action)
    #             list_action = list(eval(self.all_actions[action]))
    #             print("list action", list_action)
    #             c_list_action = [i * 50 for i in list_action]  # for 0, 1, 2,
    #             # c_list_action = [i * 25 if i < 3 else 100 for i in list_action]
    #
    #             action_alpha_list = [*c_list_action, alpha]
    #             # Execute the action and observe the next state and reward
    #             next_state, reward, terminated, _, info = self.env.step(action_alpha_list)
    #             print(info)
    #             eps_rewards.append(reward)
    #             infected.append(info['infected'])
    #             allowed.append(info['allowed'])
    #             community_risk.append(info['community_risk'])
    #             episode_infections += sum(info['infected'])
    #
    #             # Update policy stability metrics
    #             if last_action is not None and last_action != action:
    #                 policy_changes += 1
    #             last_action = action
    #
    #             # Update class utilization metrics
    #             total_class_capacity_utilized += sum(info['allowed'])
    #
    #             # Update the state to the next state
    #             c_state = next_state
    #
    #         infected_dict[episode] = infected
    #         allowed_dict[episode] = allowed
    #         rewards_dict[episode] = eps_rewards
    #         community_risk_dict[episode] = community_risk
    #
    #
    #
    #
    #     print("infected: ", infected_dict, "allowed: ", allowed_dict, "rewards: ", rewards_dict, "community_risk: ", community_risk_dict)
    #     for episode in infected_dict:
    #         plt.figure(figsize=(15, 5))
    #
    #         # Flatten the list of lists for infections and allowed students
    #         infections = [inf[0] for inf in infected_dict[episode]] if episode in infected_dict else []
    #         allowed_students = [alw[0] for alw in allowed_dict[episode]] if episode in allowed_dict else []
    #         rewards = rewards_dict[episode] if episode in rewards_dict else []
    #         community_risk = community_risk_dict[episode] if episode in community_risk_dict else []
    #
    #         # Convert range to numpy array for element-wise operations
    #         steps = np.arange(len(infections))
    #
    #         # Define bar width and offset
    #         bar_width = 0.4
    #         offset = bar_width / 4
    #
    #         # Bar plot for infections
    #         plt.bar(steps - offset, infections, width=bar_width, label='Infections', color='#bc5090', align='center')
    #
    #         # Bar plot for allowed students
    #         plt.bar(steps + offset, allowed_students, width=bar_width, label='Allowed Students', color='#003f5c',
    #                 alpha=0.5, align='edge')
    #
    #         # Line plot for rewards
    #         plt.plot(steps, rewards, label='Rewards', color='#ffa600', linestyle='-', marker='o')
    #
    #         plt.xlabel('Step')
    #         plt.ylabel('Count')
    #         plt.title(f'Evaluation of agent {self.run_name} Policy for {episode} episodes')
    #         plt.legend()
    #
    #         plt.tight_layout()
    #
    #         # Save the figure
    #         fig_path = os.path.join(eval_dir, f'{self.run_name}_metrics.png')
    #         plt.savefig(fig_path)
    #         print(f"Figure saved to {fig_path}")
    #
    #         plt.close()  # Close the figure to free up memory
    #
    #     # # Calculate additional metrics if needed
    #     # # For example, average infections, average rewards, etc.
    #     # average_infections = sum(sum(inf) for inf in infected_dict.values()) / episodes
    #     # average_rewards = sum(sum(rew) for rew in rewards_dict.values()) / episodes
    #     #
    #     # summary_data = {
    #     #         'Average Infections': average_infections,
    #     #         'Average Rewards': average_rewards,
    #     #         'Policy Stability': (episodes - policy_changes) / episodes
    #     #     }
    #     # summary_table = pd.DataFrame(summary_data, index=[0])
    #     # print(summary_table)
    #
    #     with open(eval_file_path, mode='a', newline='') as file:
    #         writer = csv.writer(file)
    #
    #         # Iterate over each episode and step to append the data
    #         for episode in tqdm(range(episodes)):
    #             for step in range(len(infected_dict[episode])):
    #                 writer.writerow([
    #                     alpha,
    #                     episode,
    #                     step,
    #                     infected_dict[episode][step],
    #                     allowed_dict[episode][step],
    #                     rewards_dict[episode][step],
    #                     community_risk_dict[episode][step]
    #
    #                 ])
    #
    #     print(f"Data for alpha {alpha} appended to {eval_file_path}")
    #
    #     return infected_dict, allowed_dict, rewards_dict, community_risk_dict

    def is_stable(self, infected_counts, threshold=5, window=10):
        # Check if the number of infections stabilizes
        if len(infected_counts) < window:
            return False
        return max(infected_counts[-window:]) - min(infected_counts[-window:]) <= threshold
    # def test(self, episodes, alpha, baseline_policy=None):
    #     """Test the trained agent with extended evaluation metrics.This function was used for the quals and AI4ED paper.
    #     """
    #
    #     total_class_capacity_utilized = 0
    #     last_action = None
    #     policy_changes = 0
    #     infected_dict = {}
    #     allowed_dict = {}
    #     rewards_dict = {}
    #     community_risk_dict = {}
    #     eval_dir = 'evaluation'
    #     if not os.path.exists(eval_dir):
    #         os.makedirs(eval_dir)
    #
    #     # Define the path for the CSV file
    #     data_file_path = os.path.join(eval_dir, 'test_simulation_data.csv')
    #     # eval_file_path = os.path.join(eval_dir, f'eval_policies_data_aaai_multi.csv')
    #     # Check if the file exists already. If not, create it and add the header
    #     # if not os.path.isfile(eval_file_path):
    #     #     with open(eval_file_path, mode='w', newline='') as file:
    #     #         writer = csv.writer(file)
    #     #         # Write the header to the CSV file
    #     #         writer.writerow(['Alpha', 'Episode', 'Step', 'Infections', 'Allowed', 'Reward', 'CommunityRisk'])
    #
    #     infected_distribution = []
    #
    #     for episode in tqdm(range(episodes)):
    #         state = self.env.reset()
    #         c_state = state[0]
    #         terminated = False
    #         episode_infections = 0
    #         infected = []
    #         allowed = []
    #         community_risk = []
    #         eps_rewards = []
    #
    #         while not terminated:
    #             converted_state = str(tuple(c_state))
    #             state_idx = self.all_states.index(converted_state)
    #
    #             # Select an action based on the Q-table or baseline policy
    #             if baseline_policy:
    #                 action = baseline_policy(c_state)
    #             else:
    #                 action = np.argmax(self.q_table[state_idx])
    #
    #
    #             list_action = list(eval(self.all_actions[action]))
    #
    #             c_list_action = [i * 50 for i in list_action]  # for 0, 1, 2,
    #             action_alpha_list = [*c_list_action, alpha]
    #             # Execute the action and observe the next state and reward
    #             next_state, reward, terminated, _, info = self.env.step(action_alpha_list)
    #             print(info)
    #             eps_rewards.append(reward)
    #             infected.append(info['infected'])
    #             allowed.append(info['allowed'])
    #             community_risk.append(info['community_risk'])
    #             episode_infections += sum(info['infected'])
    #             # Update policy stability metrics
    #             if last_action is not None and last_action != action:
    #                 policy_changes += 1
    #             last_action = action
    #
    #             # Update class utilization metrics
    #             total_class_capacity_utilized += sum(info['allowed'])
    #
    #             # Update the state to the next state
    #             c_state = next_state
    #
    #         infected_dict[episode] = infected
    #         allowed_dict[episode] = allowed
    #         rewards_dict[episode] = eps_rewards
    #         community_risk_dict[episode] = community_risk
    #         infected_distribution.append(int(episode_infections/len(infected)))
    #     # Analyzing the infected distribution
    #     plt.figure()
    #     plt.hist(infected_distribution, bins=30)
    #     plt.title('Distribution of Means of Infected Counts')
    #     plt.xlabel('Mean of Infected Count')
    #     plt.ylabel('Frequency')
    #     fig_path = os.path.join(eval_dir, f'{self.run_name}_infected_distribution_test_lyapunovfxn.png')
    #     plt.savefig(fig_path)
    #     plt.close()
    #
    #     # Write the data to the CSV file
    #     with open(data_file_path, mode='w', newline='') as file:
    #         writer = csv.writer(file)
    #         # Write the header to the CSV file
    #         writer.writerow(['Episode', 'Step', 'Infected', 'Allowed', 'Reward', 'CommunityRisk'])
    #
    #         # Write the data
    #         for episode in infected_dict:
    #             for step in range(len(infected_dict[episode])):
    #                 writer.writerow([
    #                     episode,
    #                     step,
    #                     infected_dict[episode][step][0],  # Assuming single value in list
    #                     allowed_dict[episode][step][0],  # Assuming single value in list
    #                     rewards_dict[episode][step],
    #                     community_risk_dict[episode][step]
    #                 ])
    #
    #     return data_file_path

    def test(self, episodes, alpha, baseline_policy=None):
        """Test the trained agent with extended evaluation metrics.This function was used for the quals and AI4ED paper.
        """
        infected_dict = {}
        allowed_dict = {}
        rewards_dict = {}
        community_risk_dict = {}
        eval_dir = 'evaluation'
        if not os.path.exists(eval_dir):
            os.makedirs(eval_dir)

        # Define the path for the CSV file
        data_file_path = os.path.join(eval_dir, 'report_data_test_random_low_high_perturbed.csv')

        infected_distribution = []

        for episode in tqdm(range(episodes)):
            state = self.env.reset()
            c_state = state[0]
            terminated = False
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


                list_action = list(eval(self.all_actions[action]))

                c_list_action = [i * 50 for i in list_action]  # for 0, 1, 2,
                action_alpha_list = [*c_list_action, alpha]
                # Execute the action and observe the next state and reward
                next_state, reward, terminated, _, info = self.env.step(action_alpha_list)
                print(info)
                eps_rewards.append(reward)
                infected.append(info['infected'])
                allowed.append(info['allowed'])
                community_risk.append(info['community_risk'])
                episode_infections += sum(info['infected'])
                # Update the state to the next state
                c_state = next_state

            infected_dict[episode] = infected
            allowed_dict[episode] = allowed
            rewards_dict[episode] = eps_rewards
            community_risk_dict[episode] = community_risk
            infected_distribution.append(int(episode_infections/len(infected)))

        # Write the data to the CSV file
        with open(data_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write the header to the CSV file
            writer.writerow(['Episode', 'Step', 'Infected', 'Allowed', 'Reward', 'CommunityRisk'])

            # Write the data
            for episode in infected_dict:
                for step in range(len(infected_dict[episode])):
                    writer.writerow([
                        episode,
                        step,
                        infected_dict[episode][step][0],  # Assuming single value in list
                        allowed_dict[episode][step][0],  # Assuming single value in list
                        rewards_dict[episode][step],
                        community_risk_dict[episode][step]
                    ])

        # df = pd.read_csv(data_file_path)
        #
        # # Calculating the average number of infected individuals per episode
        # average_infected_per_episode = df.groupby('Episode')['Infected'].mean()
        #
        # # We will use a rolling window to smooth out the average number of infected individuals over episodes.
        # # This helps in identifying episodes where the average number of infections is relatively constant,
        # # which may indicate an asymptotically stable equilibrium.
        #
        # # Choose a window size for smoothing. The appropriate window size may depend on the total number of episodes.
        # window_size = 5  # Example window size, can be adjusted based on data characteristics.
        #
        # # Calculate the rolling mean of the average infected individuals per episode.
        # rolling_mean_infected = average_infected_per_episode.rolling(window=window_size).mean()
        #
        # # Calculate the rolling standard deviation as a measure of stability
        # rolling_std_infected = average_infected_per_episode.rolling(window=window_size).std()
        #
        # # Identify potential equilibrium points where the rolling standard deviation is below a certain threshold,
        # # indicating low variability and potential stability.
        # low_std_threshold = 1  # Can be adjusted based on the desired level of stability.
        # potential_stable_equilibrium = rolling_std_infected[rolling_std_infected < low_std_threshold]
        #
        # # Plotting the rolling mean and standard deviation
        # plt.figure(figsize=(14, 7))
        #
        # # Rolling mean
        # plt.plot(rolling_mean_infected, label='Rolling Mean of Infected', color='blue')
        #
        # # Highlight potential equilibrium points
        # plt.scatter(potential_stable_equilibrium.index, rolling_mean_infected[potential_stable_equilibrium.index],
        #             color='red', label='Potential Stable Equilibrium', zorder=5)
        #
        # plt.fill_between(rolling_std_infected.index, rolling_mean_infected - rolling_std_infected,
        #                  rolling_mean_infected + rolling_std_infected, color='grey', alpha=0.5, label='Rolling STD')
        #
        # plt.title('Rolling Mean and STD of Average Infected Individuals per Episode')
        # plt.xlabel('Episode')
        # plt.ylabel('Average Number of Infected Individuals')
        # plt.legend()
        # plt.grid(True)
        #
        # # Displaying potential stable equilibrium episodes and their average infection numbers
        # stable_equilibrium_with_values = rolling_mean_infected[potential_stable_equilibrium.index]
        # print(stable_equilibrium_with_values)
        #
        # fig_path = os.path.join(eval_dir, f'{self.run_name}_time_series.png')
        # plt.savefig(fig_path)
        # print(f"Figure saved to {fig_path}")


        # # We will first need to calculate the average 'Infected' and 'Allowed' per episode.
        # # Let's group the data by 'Episode' and calculate the mean for 'Infected' and 'Allowed'.
        #
        # average_infected_allowed_per_episode = df.groupby('Episode')[['Infected', 'Allowed']].mean().reset_index()
        #
        # # Now we can create the phase space plot using these averages.
        # plt.figure(figsize=(10, 6))
        # plt.scatter(average_infected_allowed_per_episode['Infected'], average_infected_allowed_per_episode['Allowed'],
        #             c='blue', alpha=0.5)
        # plt.xlabel('Average Number of Infected Individuals per Episode')
        # plt.ylabel('Average Number of Allowed Students per Episode')
        # plt.title('Phase Space Plot: Average Infected vs Allowed per Episode')
        # plt.grid(True)
        # phase_fig_path = os.path.join(eval_dir, f'{self.run_name}_infall.png')
        # plt.savefig(phase_fig_path)
        # print(f"Figure saved to {phase_fig_path}")

        return data_file_path
    def test_stl(self, episodes, alpha, baseline_policy=None):
        """STL test the trained agent.
        """

        total_class_capacity_utilized = 0
        last_action = None
        policy_changes = 0
        infected_dict = {}
        allowed_dict = {}
        rewards_dict = {}
        community_risk_dict = {}
        eval_dir = 'evaluation'
        if not os.path.exists(eval_dir):
            os.makedirs(eval_dir)

        eval_file_path = os.path.join(eval_dir, f'eval_stl.csv')
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

                # print("action", action)
                list_action = list(eval(self.all_actions[action]))
                # print("list action", list_action)
                c_list_action = [i * 50 for i in list_action]  # for 0, 1, 2,
                # c_list_action = [i * 25 if i < 3 else 100 for i in list_action]

                action_alpha_list = [*c_list_action, alpha]
                # Execute the action and observe the next state and reward
                next_state, reward, terminated, _, info = self.env.step(action_alpha_list)
                # print(info)
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

        # print("infected: ", infected_dict, "allowed: ", allowed_dict, "rewards: ", rewards_dict, "community_risk: ", community_risk_dict)


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

        # print(f"Data for alpha {alpha} appended to {eval_file_path}")
        stl_evals = self.evaluate_stl_specifications(infected_dict, community_risk_dict, 25, 0.5, 15, 80, 5)
        quantitative_semantics = self.compute_quantitative_semantics(infected_dict, community_risk_dict, 25, 0.5, 15, 80, 5)
        print("stl evals", stl_evals)
        print("quantitative semantics", quantitative_semantics)

        # Graphical Representation with I_high and I_safe included
        plt.figure(figsize=(12, 8))

        # Parameters
        I_high = 80  # High threshold for oscillation count
        I_safe = 15  # Safe level for the number of infected individuals
        I_threshold = 25  # Threshold for the number of infected individuals

        # Plotting the number of infected individuals for each episode
        for episode in infected_dict:
            plt.plot(infected_dict[episode], label=f'Episode {episode} - Infected')

        # Marking the thresholds
        plt.axhline(y=I_threshold, color='orange', linestyle='--', label='I_threshold (25)')
        plt.axhline(y=I_high, color='r', linestyle='--', label='I_high (80)')
        plt.axhline(y=I_safe, color='green', linestyle='--', label='I_safe (15)')

        plt.xlabel('Time Steps')
        plt.ylabel('Number of Infected Individuals')
        plt.title('Infection Trajectories with Safety and Oscillation Thresholds')
        plt.legend()
        plt.grid(True)
        # Save the figure
        fig_path = os.path.join(eval_dir, f'{self.run_name}_stl_trajectories.png')
        plt.savefig(fig_path)
        print(f"Figure saved to {fig_path}")

        plt.close()  # Close the figure to free up memory

        return infected_dict, allowed_dict, rewards_dict, community_risk_dict, stl_evals

    # Function to compute the quantitative semantics of each trajectory
    def compute_quantitative_semantics(self,infected_dict, community_risk_dict, I_threshold, C_threshold, I_safe, I_high,
                                       tau):
        quantitative_results = {"rho_phi_1": [], "rho_phi_2": [], "rho_phi_3": []}

        for episode in infected_dict:
            infected_episode = [inf[0] for inf in infected_dict[episode]] if episode in infected_dict else []
            community_risk_episode = community_risk_dict[episode]

            # Compute rho for ϕ1 - Safety Specification
            rho_phi_1 = min([I_threshold - i for i in infected_episode])
            quantitative_results["rho_phi_1"].append(rho_phi_1)

            # Compute rho for ϕ2 - Eventual Risk Reduction Specification
            rho_phi_2_values = []
            for t, risk in enumerate(community_risk_episode):
                if risk > C_threshold:
                    # We only consider times t_prime within the time window tau
                    rho_phi_2_values.append(min([I_safe - infected_episode[t_prime] for t_prime in
                                                 range(t, min(t + tau, len(infected_episode)))]))
            rho_phi_2 = max(rho_phi_2_values) if rho_phi_2_values else float(
                'inf')  # If no risk exceeds threshold, ϕ2 is trivially satisfied
            quantitative_results["rho_phi_2"].append(rho_phi_2)

            # Compute rho for ϕ3 - Stabilization Specification
            oscillation_count = self.count_oscillations_above_threshold(infected_episode, I_high)
            rho_phi_3 = 1 - oscillation_count
            quantitative_results["rho_phi_3"].append(rho_phi_3)

        return quantitative_results
    def evaluate_stl_specifications(self, infected_dict, community_risk_dict, I_threshold, C_threshold, I_safe, I_high, tau):
        results = {"phi_1": [], "phi_2": [], "phi_3": []}
        for episode in infected_dict:
            infected_episode = [inf[0] for inf in infected_dict[episode]] if episode in infected_dict else []
            community_risk = community_risk_dict[episode]
            # print("community risk", community_risk)

            # Evaluate ϕ1
            phi_1_satisfied = all(i < I_threshold for i in infected_episode)
            results["phi_1"].append(phi_1_satisfied)

            # Evaluate ϕ2
            phi_2_satisfied = True
            for t in range(len(community_risk)):
                if community_risk[t] > C_threshold:
                    phi_2_satisfied &= any(
                        infected_episode[t_prime] < I_safe for t_prime in range(t, min(t + tau, len(infected_episode))))
            results["phi_2"].append(phi_2_satisfied)

            # Evaluate ϕ3
            oscillations = self.count_oscillations_above_threshold(infected_episode, I_high)
            phi_3_satisfied = (oscillations <= 1)
            results["phi_3"].append(phi_3_satisfied)
        # print(results)
        return results

    # Implement a helper function to count oscillations
    def count_oscillations_above_threshold(self,infected, I_high):
        """
            Counts the number of oscillations above a given threshold in the infected data.

            Parameters:
            infected (list): List of infected counts over time.
            I_high (int): The threshold above which oscillations are counted.

            Returns:
            int: Number of oscillations above the threshold.
            """

        count = 0
        above_threshold = False

        for i in range(1, len(infected)):
            if infected[i - 1] <= I_high and infected[i] > I_high:
                # Rising above the threshold
                above_threshold = True
            elif infected[i - 1] > I_high and infected[i] <= I_high and above_threshold:
                # Falling below the threshold after being above
                count += 1
                above_threshold = False

        return count



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