def train(self, alpha):
        """Train the agent."""
        # self.train_collector.collect(n_step=self.batch_size*self.training_num)
        prev_moving_avg = float("-inf")
        peak_steps = 0
        rewards_per_episode = []
        self.policy.train()
        # self.policy.eval()
        actual_rewards = []
        predicted_rewards = []
        visited_state_counts = {}
        buf = ReplayBuffer(size=20)
        for episode in tqdm(range(int(self.max_episodes))):  # total step
        # for episode in tqdm(range(2)):  # total step

            e_return = []
            e_allowed = []
            e_infected_students = []
            e_return = []
            e_community_risk = []
            e_predicted_rewards = []
            total_reward = 0
            done = False
            state = self.env.reset()

            obs = state[0].reshape(1, -1)
            while not done:
                batch = Batch(obs=obs, act=None, rew=None, done=None, obs_next=None, info=None, policy=None)
                val = self.policy(batch)
                action = val.act[0]
                e_predicted_rewards.append(val.logits[0][action-1].item())
                state, reward, done, _, info = self.env.step(action)
                next_obs = np.array(state).reshape(1, -1)
                buf.add(Batch(obs=obs, act=action, rew=reward, done=done, obs_next=next_obs, terminated = done, truncated=0, info=info, policy=None))
                obs = next_obs
                discrete_state = str(tuple(i//10 for i in state))
                if discrete_state not in visited_state_counts:
                    visited_state_counts[discrete_state] = 1
                else:
                    visited_state_counts[discrete_state] += 1

                week_reward = float(reward)
                total_reward += week_reward
                e_return.append(week_reward)
                e_allowed.append(info['allowed'])
                e_infected_students.append(info['infected'])
                e_community_risk.append(info['community_risk'])

                # Example usage:
            # If enough episodes have been run, check for convergence
            self.policy.update(0, buf, batch_size=self.batch_size, repeat=1)
            avg_episode_return = sum(e_return) / len(e_return)
            rewards_per_episode.append(avg_episode_return)
            if episode >= self.moving_average_window - 1:
                window_rewards = rewards_per_episode[max(0, episode - self.moving_average_window + 1):episode + 1]
                moving_avg = np.mean(window_rewards)
                std_dev = np.std(window_rewards)

                # Store the current moving average for comparison in the next episode


                # Log the moving average and standard deviation along with the episode number
                wandb.log({
                    'Moving Average': moving_avg,
                    'Standard Deviation': std_dev,
                    'average_return': total_reward/len(e_return),
                    'step': episode  # Ensure the x-axis is labeled correctly as 'Episodes'
                })
            predicted_rewards.append(e_predicted_rewards)
            actual_rewards.append(e_return)

        visit_counts = list(visited_state_counts.values())
        states = list(visited_state_counts.keys())
        states_visited_path = states_visited_viz(states, visit_counts,alpha, self.results_subdirectory)
        wandb.log({"States Visited": [wandb.Image(states_visited_path)]})#IMP

        avg_rewards = [sum(lst) / len(lst) for lst in actual_rewards]
        # Pass actual and predicted rewards to visualizer
        print(actual_rewards)
        print(predicted_rewards)
        explained_variance_path = visualize_explained_variance(actual_rewards, predicted_rewards, self.results_subdirectory, self.max_episodes)
        wandb.log({"Explained Variance": [wandb.Image(explained_variance_path)]})#IMP
