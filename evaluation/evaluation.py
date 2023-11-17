# Following the provided data, the script to calculate and visualize the mean reward for different alpha values is as follows:

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway

# Sample data from the CSV file
csv_file_path = 'evaluation_policies_data.csv'

# Convert the string data to a DataFrame
df = pd.read_csv(csv_file_path)

# Data Analysis
# Step 1: Group data by Alpha value and calculate mean for each alpha
mean_data = df.groupby('Alpha').mean().reset_index()

# Step 2: Plot the mean reward for each alpha
sns.barplot(x='Alpha', y='Reward', data=mean_data)
plt.title('Mean Reward for Different Alpha Values and Random Policy (alpha=0.0)')
plt.xlabel('Alpha')
plt.ylabel('Mean Reward')
plt.tight_layout()

# Save the figure
plt.savefig('mean_reward.png')
# Calculating variance for each alpha value
variance_data = df.groupby('Alpha')['Reward'].var().reset_index()

# Creating box plots to show the variance in rewards for each alpha value
plt.figure(figsize=(10, 6))
sns.boxplot(x='Alpha', y='Reward', data=df)
plt.title('Variance in Rewards for Different Alpha Values')
plt.xlabel('Alpha')
plt.ylabel('Reward Variance')
plt.savefig('variance_in_rewards.png')

plt.close()  # Close the figure to free up memory

# Grouping the data by alpha value
grouped_data = df.groupby('Alpha')['Reward']

# ANOVA test
anova_result = f_oneway(*[group for name, group in grouped_data])

# Display the ANOVA test result
print(anova_result.pvalue)
