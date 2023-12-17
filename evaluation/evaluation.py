# Following the provided data, the script to calculate and visualize the mean reward for different alpha values is as follows:

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from prettytable import PrettyTable

# Sample data from the CSV file
csv_file_path = 'eval_policies_data_aaai.csv'
df = pd.read_csv(csv_file_path)
# Convert 'Infections' and 'Allowed' columns to numeric after removing square brackets
df['Infections'] = df['Infections'].str.strip('[]').astype(int)
df['Allowed'] = df['Allowed'].str.strip('[]').astype(int)

# # Filter out the alpha value of 0.0
# df_filtered = df[df['Alpha'] != 0.0]

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

# # Add a new column for grouping
# df['Group'] = df['Alpha'].apply(lambda x: f'Policy Agent (Alpha {x})' if x != 0.0 else 'Random Agent')
#
# # Conducting Tukey's HSD Test for pairwise comparison between each alpha and the random agent
# tukey_result = pairwise_tukeyhsd(endog=df['Reward'], groups=df['Group'], alpha=0.05)
#
# # Display the results
# print(tukey_result)
# # Pretty table for results
# table = PrettyTable()
# table.field_names = ["Group 1", "Group 2", "Mean Difference", "p-adj", "Lower CI", "Upper CI", "Reject"]
# for group1, group2, meandiff, p_adj, lower, upper, reject in zip(tukey_result.groupsunique, tukey_result.groupsunique, tukey_result.meandiffs, tukey_result.pvalues, tukey_result.confint[:,0], tukey_result.confint[:,1], tukey_result.reject):
#     table.add_row([group1, group2, meandiff, p_adj, lower, upper, reject])
# print(table)


# # step 3: ANOVA test
# # compare with alpha = 0.0 ony
# grouped_data = df[df['Alpha'] != 0.0].groupby('Alpha')['Reward']
# anova_result = f_oneway(*[group for name, group in grouped_data])
# print(f"ANOVA Test p-value: {anova_result.pvalue}")
#
# # Conducting Tukey's HSD test if ANOVA is significant
# if anova_result.pvalue < 0.05:
#     print("\nConducting Tukey's HSD Test for post hoc analysis...")
#     tukey_result = pairwise_tukeyhsd(endog=df['Reward'], groups=df['Alpha'], alpha=0.05)
#     print(tukey_result)
#     # pretty table
#     table = PrettyTable()
#     table.field_names = ["alpha", "mean", "std", "min", "max"]
#     for i in range(len(tukey_result.groupsunique)):
#         table.add_row([tukey_result.groupsunique[i], tukey_result.meandiffs[i], tukey_result.std_pairs[i],
#                        tukey_result.confint[i][0], tukey_result.confint[i][1]])
#     print(table)

# step 4: 3D plot
# do a plot without dots

# Plotting the progression of infections over time for each alpha value
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
for alpha in df['Alpha'].unique():
    subset = df[df['Alpha'] == alpha]
    ax.scatter(subset['CommunityRisk'], subset['Allowed'],subset['Infections'] , label=f'Alpha {alpha}', alpha=0.7)

ax.set_zlabel('Infected')
ax.set_ylabel('Allowed')
ax.set_xlabel('Community Risk')
ax.set_title('Allowed vs. Infections for Different Alpha Values')
ax.legend(loc='best', bbox_to_anchor=(1, 0.5), title="Alpha Values")

plt.savefig('CR-3d_multi.png')



# # step 5: 2D plot
# # Plotting the progression of infections over time for each alpha value
# plt.figure(figsize=(10, 6))
# for alpha in df_filtered['Alpha'].unique():
#     subset = df_filtered[df_filtered['Alpha'] == alpha]
#     plt.scatter(subset['Infections'], subset['Allowed'], label=f'Alpha {alpha}', alpha=0.7)
#
# plt.xlabel('Infections')
# plt.ylabel('Allowed')
# plt.title('Allowed vs. Infections for Different Alpha Values')
# plt.legend()
# plt.grid(True)
# plt.savefig('2d.png')



plt.close()  # Close the figure to free up memory



# Plotting the progression of infections over time for each alpha value
# plt.figure(figsize=(10, 6))
# for alpha in df_filtered['Alpha'].unique():
#     subset = df_filtered[df_filtered['Alpha'] == alpha]
#     plt.scatter(subset['Infections'], subset['Allowed'], label=f'Alpha {alpha}', alpha=0.7)
#
# plt.xlabel('Infections')
# plt.ylabel('Allowed')
# plt.title('Allowed vs. Infections for Different Alpha Values')
# plt.legend()
# plt.grid(True)
# plt.show()
# # Step 2: Plot the mean reward for each alpha
# sns.barplot(x='Alpha', y='Reward', data=mean_data)
# plt.title('Mean Reward for Different Alpha Values and Random Policy (alpha=0.0)')
# plt.xlabel('Alpha')
# plt.ylabel('Mean Reward')
# plt.tight_layout()
#
# # Save the figure
# plt.savefig('final_mean_reward.png')
# # Calculating variance for each alpha value
# variance_data = df.groupby('Alpha')['Reward'].var().reset_index()
#
# # Creating box plots to show the variance in rewards for each alpha value
# plt.figure(figsize=(10, 6))
# sns.boxplot(x='Alpha', y='Reward', data=df)
# plt.title('Variance in Rewards for Different Alpha Values')
# plt.xlabel('Alpha')
# plt.ylabel('Reward Variance')
# plt.savefig('final_variance_in_rewards.png')
#
# plt.close()  # Close the figure to free up memory
#
# # Grouping the data by alpha value
# grouped_data = df.groupby('Alpha')['Reward']
#
# # ANOVA test
# anova_result = f_oneway(*[group for name, group in grouped_data])
#
# # Display the ANOVA test result
# # Display the ANOVA test result
# print(f"ANOVA Test p-value: {anova_result.pvalue}")
# # Conducting Tukey's HSD test if ANOVA is significant
# if anova_result.pvalue < 0.05:
#     print("\nConducting Tukey's HSD Test for post hoc analysis...")
#     tukey_result = pairwise_tukeyhsd(endog=df['Reward'], groups=df['Alpha'], alpha=0.05)
#     print(tukey_result)