import numpy as np
import matplotlib.pyplot as plt

# Model parameters
alpha_m = 0.02  # Transmission risk within the classroom
beta = 0.01  # Community risk scaling factor
c_risk_mean = 0.03  # Mean community risk of infection
c_risk_std = 0.01  # Standard deviation of community risk of infection

# Function to compute R0 with varying community risk
def compute_R0(N, c_risk):
    return N * (alpha_m + beta * c_risk)

# Generate data
N_range = np.linspace(0, 100, 1000)
R0_values_mean = [compute_R0(N, c_risk_mean) for N in N_range]
R0_values_upper = [compute_R0(N, c_risk_mean + c_risk_std) for N in N_range]
R0_values_lower = [compute_R0(N, c_risk_mean - c_risk_std) for N in N_range]

# Find the threshold points
threshold_index_mean = np.where(np.array(R0_values_mean) >= 1)[0][0]
threshold_N_mean = N_range[threshold_index_mean]
threshold_index_upper = np.where(np.array(R0_values_upper) >= 1)[0][0]
threshold_N_upper = N_range[threshold_index_upper]
threshold_index_lower = np.where(np.array(R0_values_lower) >= 1)[0][0]
threshold_N_lower = N_range[threshold_index_lower]

# Create the plot
plt.figure(figsize=(12, 8))

# Plot R0
plt.plot(N_range, R0_values_mean, 'b-', label='$R_0$ (mean community risk)')
plt.plot(N_range, R0_values_upper, 'b--', label='$R_0$ (mean + std community risk)')
plt.plot(N_range, R0_values_lower, 'b-.', label='$R_0$ (mean - std community risk)')

# Fill areas
plt.fill_between(N_range[:threshold_index_mean], 0, R0_values_mean[:threshold_index_mean], color='g', alpha=0.3, label='DFE Stable ($R_0 < 1$)')
plt.fill_between(N_range[threshold_index_mean:], 1, R0_values_mean[threshold_index_mean:], color='r', alpha=0.3, label='EE Stable ($R_0 \geq 1$)')

# Add threshold line
plt.axhline(y=1, color='k', linestyle='--', label='$R_0 = 1$')

# Customize the plot
plt.xlabel('Number of Students ($N_i$)')
plt.ylabel('$R_0$')
plt.title(f'Threshold Behavior of $R_0$ (α_m = {alpha_m}, β = {beta}, community risk = {c_risk_mean} ± {c_risk_std})')
plt.legend()
plt.grid(True)
plt.ylim(0, 5)

# Add text annotations
plt.text(5, 0.5, 'DFE Stable\n($R_0 < 1$)', fontsize=10, ha='left', va='center')
plt.text(80, 3, 'EE Stable\n($R_0 \geq 1$)', fontsize=10, ha='left', va='center')

plt.tight_layout()
plt.savefig("threshold_behavior.png")
plt.show()

print(f"Threshold number of students (mean): {threshold_N_mean:.2f}")
print(f"Threshold number of students (mean + std): {threshold_N_upper:.2f}")
print(f"Threshold number of students (mean - std): {threshold_N_lower:.2f}")
