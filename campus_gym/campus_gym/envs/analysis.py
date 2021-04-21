import json
import matplotlib.pyplot as plt
import numpy as np
import random
reward_file = open('deep_rewards.json')
test_reward_file = open('deep_testing_rewards.json')
allowed_file = open('deep_allowed.json')
infected_file = open('deep_infected.json')
allowed = json.load(allowed_file)
infected = json.load(infected_file)


###################### START REWARDS #############################################
rewards = json.load(reward_file)
test_rewards = json.load(test_reward_file)


# Average rewards per episode
rewards_title_avg_per_episode = 'Average rewards per episode'
rewards_episode_axis = list(rewards.keys())
rewards_average_rewards_axis = []

for i in (list(rewards.values())):
    avg_reward = int(sum(i)/len(i))
    rewards_average_rewards_axis.append(avg_reward)

# Average Rewards
x = rewards_episode_axis
y = rewards_average_rewards_axis


axis_index = [5, 10, 20, 40, 100, 200, 400, 500, 700, 1000, 1300, 1500, 1800, 2000, 2500, 2999]
#axis_index = [1, 2, 4, 8, 10, 20, 40, 50, 70, 100, 130, 150, 180, 300, 250, 999]


new_x = []
new_y = []

for i in axis_index:
    new_x.append(x[i])
    new_y.append(y[i])



plt.plot(new_x,new_y, label='average training rewards', marker='o')
plt.title("Average of Expected Reward")
plt.tick_params(axis='x', rotation=90)
plt.xlabel("Selected Episodes")
plt.ylabel("Average Expected Reward")
plt.show()

# Test Average rewards per episode
test_rewards_title_avg_per_episode = 'Average rewards per episode'
test_rewards_episode_axis = list(test_rewards.keys())
test_rewards_list = list(rewards.values())
test_reward_avg = []
for i in (list(test_rewards.values())):
    avg_reward = int(sum(i) / len(i))
    test_reward_avg.append(avg_reward)

test_x = test_rewards_episode_axis
test_y = test_reward_avg
plt.plot(test_x,test_y, marker= 'o')
plt.title("Average expected rewards after training for 100 episodes")
plt.ylim(0,60)
plt.xlabel("Episodes")
plt.ylabel("Average Expected Reward")
plt.show()

# Week by Week Average Rewards
week_by_week_rewards = list(map(int, [int(sum(l))/len(l) for l in zip(*(list(rewards.values())))]))
x_axis = []
for i in range(len(week_by_week_rewards)):
    x_axis.append(i)

plt.plot(x_axis, week_by_week_rewards, marker= 'o')
plt.ylim(0, 60)
plt.title('Training phase')
plt.xlabel('Week')
plt.ylabel('Expected Reward')
plt.show()

###################### END REWARDS #############################################
#Week by Week Infected and Allowed per course per episode

allowed_list = list(allowed.values())
infected_list = list(infected.values())
course_allowed_episodes = []
course_infected_episodes = []
for i in allowed_list:
    course_allowed_episodes.append(list(map(int, [int(sum(l))/len(l) for l in zip(*(i))])))
for j in infected_list:
    course_infected_episodes.append(list(map(int, [int(sum(m))/len(m) for m in zip(*(j))])))

course_allowed_episodes_array = np.array(course_allowed_episodes)
course_infected_episodes_array = np.array(course_infected_episodes)
# allowed_total_courses = course_allowed_episodes_array.shape[1]

course_infected_episodes_array_transpose = np.transpose(course_infected_episodes_array)
course_allowed_episodes_array_transpose = np.transpose(course_allowed_episodes_array)

allowed_plots_to_show = []
infected_plots_to_show = []

for row in course_allowed_episodes_array_transpose:
    allowed_plots_to_show.append(list(row))

for row in course_infected_episodes_array_transpose:
    infected_plots_to_show.append(list(row))

try:
    allowed_infected = zip(allowed_plots_to_show, infected_plots_to_show)
    allowed_infected_plot_pairs = []
    for a, b in allowed_infected:
        allowed_infected_plot_pairs.append([a,b])

    for index, course in enumerate(allowed_infected_plot_pairs):
        plt.scatter(course[0], course[1])
        plt.title("course: " + str(index))
        plt.xlabel("Allowed students")
        plt.ylabel("Infected student")
        plt.show()

except:
    print("Size of allowed and infected need to match")

