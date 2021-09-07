import matplotlib.pyplot as plt
import seaborn as sns
import json

training_rewards = json.load(open('results/E-greedy/episode_rewards.json'))
training_actions = json.load(open('results/E-greedy/episode_actions.json'))
training_allowed = json.load(open('results/E-greedy/episode_allowed.json'))
training_infected = json.load(open('results/E-greedy/episode_infected.json'))

# Extract allowed and infected students per course.
# TODO: Make this part of the code to be dynamic. i.e not assuming number of courses as implemented below.


def extract_course_details(allowed_students_or_infected_students_dict):
    course_1 = {}
    course_2 = {}
    course_3 = {}
    for key, value in allowed_students_or_infected_students_dict.items():
        course_1_l = []
        course_2_l = []
        course_3_l = []

        for i in value:
            course_1_l.append(i[0])
            course_2_l.append(i[1])
            course_3_l.append(i[2])

        course_1[key] = course_1_l
        course_2[key] = course_2_l
        course_3[key] = course_3_l

    return course_1, course_2, course_3


def get_avg(training_rewards_dict):
    training_rewards_list = []
    for key, value in training_rewards_dict.items():
        expected_avg_reward = int(sum(value)/len(value))
        training_rewards_list.append(expected_avg_reward)

    return training_rewards_list

course_1_infected, course_2_infected, course_3_infected = extract_course_details(training_infected)
course_1_allowed, course_2_allowed, course_3_allowed = extract_course_details(training_allowed)
course_1_actions, course_2_actions, course_3_actions = extract_course_details(training_actions)

# Plot and save Rewards
average_rewards = get_avg(training_rewards)
sns.pointplot(list(range(0, len(average_rewards))), average_rewards )
plt.savefig('results/E-greedy/ci_rewards.png')

# Plot and save course1 infected vs allowed

