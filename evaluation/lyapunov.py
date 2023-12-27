import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
class SimulationDataset(Dataset):
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


def certify_stability(model, test_loader, perturbation_loader):
    model.eval()  # Set the model to evaluation mode

    def evaluate_lyapunov_conditions(data_loader):
        V_values = []
        V_next_values = []

        with torch.no_grad():
            for current_state, next_state in data_loader:
                V_x = model(current_state).squeeze()
                V_x_next = model(next_state).squeeze()

                V_values.extend(V_x.tolist())
                V_next_values.extend(V_x_next.tolist())

        return V_values, V_next_values

    # Evaluate on test data
    V_test, V_next_test = evaluate_lyapunov_conditions(test_loader)

    # Evaluate on perturbation data
    V_perturb, V_next_perturb = evaluate_lyapunov_conditions(perturbation_loader)

    # Visualization
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(V_test, V_next_test, alpha=0.7)
    plt.plot([0, max(V_test)], [0, max(V_test)], 'r--')  # Line y = x for reference
    plt.xlabel('V(x)')
    plt.ylabel('V(x\')')
    plt.title('Test Data - Lyapunov Function')

    plt.subplot(1, 2, 2)
    plt.scatter(V_perturb, V_next_perturb, alpha=0.7)
    plt.plot([0, max(V_perturb)], [0, max(V_perturb)], 'r--')
    plt.xlabel('V(x)')
    plt.ylabel('V(x\')')
    plt.title('Perturbation Data - Lyapunov Function')

    plt.tight_layout()
    plt.savefig('lyapunov_function.png')

    # Certify stability
    stable_test = all(v_next < v for v, v_next in zip(V_test, V_next_test))
    stable_perturb = all(v_next < v for v, v_next in zip(V_perturb, V_next_perturb))

    if stable_test and stable_perturb:
        print("The system is stable under test and perturbation conditions.")
    else:
        print("The system may not be stable.")
def lyapunov_loss(lyapunov_values, next_lyapunov_values):
    # Lyapunov condition: V(x) > 0
    lyapunov_positive = torch.mean(F.relu(-lyapunov_values))

    # Lyapunov derivative condition: dV/dt < 0
    lyapunov_derivative = torch.mean(F.relu(next_lyapunov_values - lyapunov_values))

    # Combine both conditions into a single loss
    loss = lyapunov_positive + lyapunov_derivative
    return loss


def train_model(model, data_loader, epochs):
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(epochs):
        epoch_loss = 0.0
        for current_state, next_state in data_loader:
            optimizer.zero_grad()

            V_x = model(current_state)
            V_x_next = model(next_state)

            loss = lyapunov_loss(V_x, V_x_next)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()


        print(f'Epoch {epoch + 1}, Average Loss: {epoch_loss / len(data_loader)}')

    # Save the trained model
    torch.save(model.state_dict(), 'lyapunov_model.pth')


def test_model(model, data_loader):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0

    with torch.no_grad():
        for current_state, next_state in data_loader:
            V_x = model(current_state)
            V_x_next = model(next_state)

            loss = lyapunov_loss(V_x, V_x_next)
            total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    print(f"Average test loss: {avg_loss}")

class LyapunovFunction(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LyapunovFunction, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



in_features = 2  # 'Infected' and 'Allowed'
hidden_size = 5
output_size = 1  # Lyapunov function has only one output
lyapunov_function = LyapunovFunction(in_features, hidden_size, output_size)
data = pd.read_csv('report_data.csv')

dataset = SimulationDataset(data, episode_length=15)

data_loader = DataLoader(dataset, batch_size=30, shuffle=True)

# Split your dataset into training and validation sets
train_size = int(0.8 * len(data_loader.dataset))  # 80% for training
val_size = len(data_loader.dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(data_loader.dataset, [train_size, val_size])

# Training the model
# train_model(lyapunov_function, train_dataset, epochs=10)

# Load the trained model weights
state_dict = torch.load('lyapunov_model.pth')
lyapunov_function.load_state_dict(state_dict)
# Validate the model
test_model(lyapunov_function, val_dataset)


# Load your test and perturbation data
test_data = pd.read_csv('report_data_test_random_low_high.csv')
test_dataset = SimulationDataset(test_data, episode_length=15)
test_loader = DataLoader(test_dataset, batch_size=30, shuffle=True)

perturbation_data = pd.read_csv('report_data_test_random_low_high_perturbed.csv')
perturbation_dataset = SimulationDataset(perturbation_data, episode_length=15)
perturbation_loader = DataLoader(perturbation_dataset, batch_size=30, shuffle=True)

# Certify stability
certify_stability(lyapunov_function, test_loader, perturbation_loader)