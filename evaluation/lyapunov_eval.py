import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import numpy as np


# Load the data
data = pd.read_csv('report_data.csv')
test_data = pd.read_csv('report_data.csv')

# Preprocess the data (if necessary)
# For example, using StandardScaler for normalization
scaler = StandardScaler()
features = ['Infected','CommunityRisk']
data[features] = scaler.fit_transform(data[features])
test_data[features] = scaler.fit_transform(test_data[features])

eval_dir = 'evaluation'
if not os.path.exists(eval_dir):
    os.makedirs(eval_dir)

# Define a custom dataset
class EpisodicSimulationDataset(Dataset):
    def __init__(self, dataframe, episode_length):
        self.episode_length = episode_length
        # Only select 'Infected' and 'CommunityRisk' columns for reshaping
        self.features = dataframe[['Infected', 'CommunityRisk']].values

        # Reshape data into episodes based only on the selected features
        # The shape now should be (-1, episode_length, 2) as there are two features
        self.episodes = self.features.reshape(-1, episode_length, 2)

    def __len__(self):
        # Total number of transitions is the number of episodes times the (episode_length - 1)
        return (self.episodes.shape[0]) * (self.episode_length - 1)

    def __getitem__(self, idx):
        # Calculate episode index and step index within episode
        episode_idx = idx // (self.episode_length - 1)
        step_idx = idx % (self.episode_length - 1)

        current_state = torch.tensor(self.episodes[episode_idx, step_idx, :], dtype=torch.float32)
        next_state = torch.tensor(self.episodes[episode_idx, step_idx + 1, :], dtype=torch.float32)
        return current_state, next_state


class LyapunovFunctionNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LyapunovFunctionNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def lyapunov_loss(lyapunov_values, next_lyapunov_values):
    # Lyapunov condition: V(x) > 0
    lyapunov_positive = torch.mean(F.relu(-lyapunov_values))

    # Lyapunov derivative condition: dV/dt < 0
    lyapunov_derivative = torch.mean(F.relu(next_lyapunov_values - lyapunov_values))

    # Combine both conditions into a single loss
    loss = lyapunov_positive + lyapunov_derivative
    return loss



def train_lyapunov_network(input_size, hidden_size, output_size, num_epochs):
    # Create the dataset and DataLoader
    dataset = EpisodicSimulationDataset(data, 15)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    test_dataset = EpisodicSimulationDataset(data, 15)


    # Split your dataset into training and validation sets
    train_size = int(0.8 * len(dataloader.dataset))  # 80% for training
    val_size = len(dataloader.dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataloader.dataset, [train_size, val_size])

    # DataLoaders for training and validation sets
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Initialize network and optimizer
    lyapunov_network = LyapunovFunctionNetwork(input_size, hidden_size, output_size)
    optimizer = torch.optim.Adam(lyapunov_network.parameters())
    train_losses = []
    val_losses = []

    # Training loop
    for epoch in range(num_epochs):
        for current_state, next_state in train_loader:
            # print("Current State: ", current_state.shape)
            optimizer.zero_grad()

            # Compute Lyapunov function values
            lyapunov_value = lyapunov_network(current_state)
            next_lyapunov_value = lyapunov_network(next_state)

            # Compute loss and backpropagate
            loss = lyapunov_loss(lyapunov_value, next_lyapunov_value)
            train_losses.append(loss.item())
            loss.backward()
            optimizer.step()

        # Print epoch loss or add any other logging mechanism
        print(f'Epoch {epoch} Loss: {loss.item()}')

    # Save the trained network
    torch.save(lyapunov_network.state_dict(), 'lyapunov_network.pth')

    # # Validate the trained network
    # lyapunov_network.eval()  # Set the network to evaluation mode
    # with torch.no_grad():  # Disable gradient computation for validation
    #     val_losses = []
    #     for current_state, next_state in val_loader:
    #         lyapunov_value = lyapunov_network(current_state)
    #         next_lyapunov_value = lyapunov_network(next_state)
    #
    #         val_loss = lyapunov_loss(lyapunov_value, next_lyapunov_value)
    #         val_losses.append(val_loss.item())
    #         val_losses.append(val_loss.item())
    #
    #     avg_val_loss = sum(val_losses) / len(val_losses)
    #     print(f'Average Validation Loss: {avg_val_loss}')
    # # Plotting the training and validation losses
    # plt.figure(figsize=(10, 5))
    # plt.plot(train_losses, label='Training Loss')
    # plt.plot(val_losses, label='Validation Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Training and Validation Loss')
    # plt.legend()
    # fig_path = os.path.join(eval_dir, 'training_validation_loss.png')
    # plt.savefig(fig_path)
    # print(f"Figure saved to {fig_path}")
    # # plt.close()
    #
    # lyapunov_values = []
    # derivatives = []
    #
    # lyapunov_network.eval()
    # with torch.no_grad():
    #     for current_state, next_state in test_loader:
    #         current_value = lyapunov_network(current_state).item()
    #         next_value = lyapunov_network(next_state).item()
    #
    #         lyapunov_values.append(current_value)
    #         derivatives.append((next_value - current_value))  # Simple discrete derivative
    #
    # # Plotting Lyapunov function values
    # plt.figure(figsize=(10, 5))
    # plt.plot(lyapunov_values, label='Lyapunov Function Value')
    # plt.xlabel('Sample')
    # plt.ylabel('Value')
    # plt.title('Lyapunov Function Values Over Different States')
    # plt.legend()
    # lyapunov_fig_path = os.path.join(eval_dir, 'lyapunov_values.png')
    # plt.savefig(lyapunov_fig_path)
    # print(f"Figure saved to {lyapunov_fig_path}")
    #
    # # Plotting derivatives
    # plt.figure(figsize=(10, 5))
    # plt.plot(derivatives, label='Derivative of Lyapunov Function')
    # plt.xlabel('Sample')
    # plt.ylabel('Derivative Value')
    # plt.title('Derivatives of Lyapunov Function Over Different States')
    # plt.legend()
    # derivatives_fig_path = os.path.join(eval_dir, 'derivatives.png')
    # plt.savefig(derivatives_fig_path)
    # print(f"Figure saved to {derivatives_fig_path}")
    #
    # # Calculate Lyapunov function values
    # # lyapunov_values = []
    # # Inverse transform to get actual values
    # actual_values = scaler.inverse_transform(test_data[features])
    #
    # with torch.no_grad():
    #     for current_state, _ in test_loader:
    #         # Calculate Lyapunov function value
    #         lyapunov_value = lyapunov_network(current_state).item()
    #         lyapunov_values.append(lyapunov_value)
    #
    # # Convert lists to NumPy arrays if they aren't already
    # actual_values = np.array(actual_values)
    # lyapunov_values = np.array(lyapunov_values)
    #
    # # Calculate the minimum and maximum values for normalization
    # actual_min, actual_max = actual_values.min(), actual_values.max()
    # lyapunov_min, lyapunov_max = lyapunov_values.min(), lyapunov_values.max()
    #
    # # Normalize the actual values and Lyapunov values for better comparison
    # actual_values_normalized = (actual_values - actual_min) / (actual_max - actual_min)
    # lyapunov_values_normalized = (lyapunov_values - lyapunov_min) / (lyapunov_max - lyapunov_min)
    # # Calculate tolerance based on the normalized scale
    # tolerance = np.std(np.diff(lyapunov_values_normalized)) * 3  # Example: 3-sigma rule of thumb
    # print("actual_values_normalized: ", actual_values_normalized)
    #
    # fig, ax1 = plt.subplots()
    #
    # color = 'tab:red'
    # ax1.set_xlabel('Time')
    # ax1.set_ylabel('Actual Infected Count (normalized)', color=color)
    # ax1.plot(actual_values_normalized, color=color)
    # ax1.tick_params(axis='y', labelcolor=color)
    #
    # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    #
    # color = 'tab:blue'
    # ax2.set_ylabel('Lyapunov Function Values (normalized)', color=color)
    # ax2.plot(lyapunov_values_normalized, color=color)
    # ax2.tick_params(axis='y', labelcolor=color)
    # fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # plt.title('Comparison of Normalized Actual Infected Count and Lyapunov Function Values')
    # plt.legend()
    # comparison_fig_path = os.path.join(eval_dir, 'comparison.png')
    # plt.savefig(comparison_fig_path)
    # print(f"Figure saved to {comparison_fig_path}")
    #
    #
    # plt.close()

    # # Execute the functions
    # certify_message = certify_stability(lyapunov_values, derivatives, tolerance)
    # validate_message = validate_lyapunov_function(lyapunov_values, tolerance)
    # print(certify_message)
    # print(validate_message)
    # # Example usage:
    # # initial_state is the state you want to test
    # # perturbation is a vector (of the same size as the state) that represents the initial perturbation
    # initial_state = [0.5, 0.1]  # Replace with the actual initial state of your system
    # perturbation = [0.01, 0.01]  # Replace with the actual perturbation you want to apply
    # test_response_to_perturbations(lyapunov_network, initial_state, perturbation, steps=10)


# Certify Stability
def certify_stability(lyapunov_values, derivatives, tolerance):
    # Check if all Lyapunov values are positive
    if np.any(lyapunov_values <= 0):
        return False, "Lyapunov function is not positive definite."

    # Check if the derivative is negative or zero (within a tolerance)
    if np.any(derivatives > tolerance):
        return False, "Lyapunov derivative is not non-positive."

    return True, "System is stable according to the Lyapunov function."


# Validate Lyapunov Function
def validate_lyapunov_function(lyapunov_values, tolerance):
    # Check for any increases in Lyapunov function value
    if np.any(np.diff(lyapunov_values) > tolerance):
        return False, "Lyapunov function does not consistently decrease."

    return True, "Lyapunov function validated."


def test_response_to_perturbations(network, initial_state, perturbation, steps=10):
    network.eval()

    # Convert the initial state to a tensor and evaluate it
    state_tensor = torch.tensor(initial_state, dtype=torch.float32)
    original_value = network(state_tensor).item()

    # Apply perturbation and evaluate
    perturbed_state = state_tensor + torch.tensor(perturbation, dtype=torch.float32)
    perturbed_value = network(perturbed_state).item()

    # Initialize a variable to track the current value
    current_value = perturbed_value

    # Check if the initial perturbation made the value increase
    if current_value > original_value:
        print("Perturbation increased the Lyapunov value, suggesting instability.")
        return False

    # Iterate for the given number of steps
    for step in range(steps):
        # Apply a random perturbation at each step
        random_perturbation = torch.randn_like(state_tensor) * 0.01  # Small random perturbation
        perturbed_state += random_perturbation

        # Evaluate the perturbed state
        perturbed_value = network(perturbed_state).item()

        # Check if the Lyapunov function value is decreasing
        if perturbed_value > current_value:
            print(f"Perturbation at step {step} increased the Lyapunov value.")
            return False  # If value increases, system may not be stable

        # Update the current value
        current_value = perturbed_value

    print("System is likely stable: Lyapunov values did not increase after perturbations.")
    return True  # If the loop completes, the system is likely stable under small perturbations


# Call the function with appropriate parameters
input_size = 2  # Number of features (Infected, Allowed, CommunityRisk)
hidden_size = 10  # Example size, adjust as necessary
output_size = 1  # Outputting a single Lyapunov function value
num_epochs = 10  # Example number of epochs, adjust as necessary

train_lyapunov_network(input_size, hidden_size, output_size, num_epochs)

