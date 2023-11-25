import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sample data from the CSV file
csv_file_path = 'eval_policies_data_aaai.csv'
df = pd.read_csv(csv_file_path)
# Convert 'Infections' and 'Allowed' columns to numeric after removing square brackets
df['Infections'] = df['Infections'].str.strip('[]').astype(int)
df['Allowed'] = df['Allowed'].str.strip('[]').astype(int)

# Calculate the mean of 'Allowed' and 'Infections' for each alpha value, including Alpha in the dataframe
mean_values = df.groupby('Alpha', as_index=False)[['Infections', 'Allowed','CommunityRisk']].mean()

# Creating a scatter plot of the mean values with different colors for each alpha value
plt.figure(figsize=(10, 6))
for alpha in mean_values['Alpha'].unique():
    subset = mean_values[mean_values['Alpha'] == alpha]
    plt.scatter(subset['Infections'], subset['Allowed'], label=f'Alpha {alpha}', alpha=0.7, s=50)  # s is the size of dots

plt.xlabel('Mean Infections')
plt.ylabel('Mean Allowed')
plt.title('Mean Allowed vs. Mean Infections for Different Alpha Values')
plt.legend(loc='lower right')
plt.grid(True)

plt.savefig('tradeoff_multi.png')

# Step 2: Plot the mean reward for each alpha
mean_data = df.groupby('Alpha')['Reward'].mean().reset_index()
sns.barplot(x='Alpha', y='Reward', data=mean_data)
plt.title('Mean Reward for Different Alpha Values and Random Policy (alpha=0.0)')
plt.xlabel('Alpha')
plt.ylabel('Mean Reward')
plt.tight_layout()
plt.savefig('meanreward_multi.png')

# Calculating variance for each alpha value
variance_data = df.groupby('Alpha')['Reward'].var().reset_index()

# Creating box plots to show the variance in rewards for each alpha value
plt.figure(figsize=(10, 6))
sns.boxplot(x='Alpha', y='Reward', data=df)
plt.title('Variance in Rewards for Different Alpha Values')
plt.xlabel('Alpha')
plt.ylabel('Reward Variance')
plt.savefig('variance_multi.png')

plt.close()  # Close the figure to free up memory
