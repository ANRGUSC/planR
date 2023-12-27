import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from scipy.stats import ks_2samp


def chi_square_test_for_independence(observed_matrix):
    chi2, p, dof, expected = chi2_contingency(observed_matrix)
    return chi2, p


def kolmogorov_smirnov_test(distribution1, distribution2):
    statistic, p_value = ks_2samp(distribution1, distribution2)
    return statistic, p_value


data = pd.read_csv('report_data_half_half_52.csv')
data['NextInfected'] = data.groupby('Episode')['Infected'].shift(-1)
data = data.dropna()
transition_matrix = pd.crosstab(data['Infected'], data['NextInfected'], normalize='index')
# Ensure the matrix is square by adding missing states (if any)
all_states = set(data['Infected']).union(set(data['NextInfected']))
transition_matrix = transition_matrix.reindex(index=all_states, columns=all_states, fill_value=0)
plt.figure(figsize=(10, 8))
sns.heatmap(transition_matrix, annot=False, cmap='coolwarm')
plt.title('Transition Probability Matrix-Same as Training Environment ')
plt.xlabel('Next State')
plt.ylabel('Current State')
# save the figure
plt.savefig('transition_matrix_half_half_52.png')

# # Calculate the transition matrix with actions (Allowed)
# n_transition_matrix = pd.crosstab(index=[data['Infected'], data['Allowed']],
#                                 columns=data['NextInfected'],
#                                 normalize='index')

# Filter to show only probabilities greater than 0
non_zero_transitions = transition_matrix[transition_matrix > 0].stack().reset_index()
# Adjust the column names based on the actual structure of non_zero_transitions
if len(non_zero_transitions.columns) == 4:
    non_zero_transitions.columns = ['Current State', 'Next State', 'Action', 'Probability']
elif len(non_zero_transitions.columns) == 3:
    non_zero_transitions.columns = ['Current State', 'Next State', 'Probability']

# Eigenvalue decomposition to find the steady-state distribution
eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
# Find the index of the eigenvalue that is closest to 1
closest_to_one = np.isclose(eigenvalues, 1)
if not np.any(closest_to_one):
    raise ValueError("No eigenvalue close to 1 found.")

stationary_distribution_idx = np.where(closest_to_one)[0][0]
stationary_distribution = np.abs(np.real(eigenvectors[:, stationary_distribution_idx]))
stationary_distribution /= np.sum(stationary_distribution)

print("Sum of steady-state probabilities:", np.sum(stationary_distribution))
print("Range of steady-state probabilities:", np.min(stationary_distribution), np.max(stationary_distribution))

# Normalize the stationary distribution if not already
if not np.isclose(np.sum(stationary_distribution), 1):
    stationary_distribution /= np.sum(stationary_distribution)

# Ensure all probabilities are non-negative
stationary_distribution = np.clip(stationary_distribution, 0, 1)

# Save the stationary distribution to a DataFrame and CSV
stationary_distribution_df = pd.DataFrame(stationary_distribution, index=all_states).reset_index()
stationary_distribution_df.columns = ['State', 'Probability']
stationary_distribution_df.to_csv('stationary_distribution_half_half_52.csv', index=False)

# Visualization of the steady-state distribution
plt.figure(figsize=(10, 6))
plt.bar(stationary_distribution_df['State'], stationary_distribution_df['Probability'], color='skyblue')
plt.xlabel('State')
plt.ylabel('Probability')
plt.title('Steady-State Distribution: Same as Training Environment')
plt.xticks(rotation=90)  # Rotate the x-axis labels for better readability if necessary
plt.tight_layout()  # Adjust layout to prevent clipping of labels
plt.savefig('steady_state_distribution_half_half_52.png')
