import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from scipy.stats import truncnorm
import seaborn as sns

np.random.seed(10)
#
# def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
#     return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)
#
# def get_infected_students_apprx_si(previous_infected, allowed, community_risk, const_1, const_2, total_students=100):
#     susceptible = max(0, total_students - previous_infected)
#     new_infected_inside = int((const_1 * previous_infected) * (susceptible / total_students) * susceptible * allowed)
#     new_infected_outside = int(const_2 * (community_risk * allowed) * susceptible)
#     recovered = max(int(1.0 * previous_infected), 0)
#     total_infected = new_infected_inside + new_infected_outside
#     infected = min(previous_infected + int(total_infected) - recovered, allowed)
#     return infected
#
# def get_infected_v01(previous_infected, allowed, community_risk, const_1, const_2, total_students=100):
#     infected = int(((const_1 * previous_infected) * allowed) + ((const_2 * community_risk) * allowed ** 2))
#     return infected
#
# # SIR Model Equations
# def sir_model(y, t, N, beta, gamma):
#     S, I, R = y
#     dSdt = -beta * S * I / N
#     dIdt = beta * S * I / N - gamma * I
#     dRdt = gamma * I
#     return dSdt, dIdt, dRdt
#
# def si_model(y, t, N, beta):
#     S, I = y
#     dSdt = -beta * S * I / N
#     dIdt = beta * S * I / N
#     return dSdt, dIdt
#
# # Parameters
# N = 100
# beta = 0.2  # Transmission rate
# gamma = 0.9  # Recovery rate
# initial_infected = 5
# initial_recovered = 0
# initial_susceptible = N - initial_infected - initial_recovered
# initial_conditions = initial_susceptible, initial_infected, initial_recovered
# t = np.linspace(0, 100, 100)  # Time steps
#
# # Solve SIR model
# solution = odeint(sir_model, initial_conditions, t, args=(N, beta, gamma))
# solution = np.array(solution)
#
# # Solve SI model
# solution_si = odeint(si_model, [initial_susceptible, initial_infected], t, args=(N, beta))
#
# # Approximate SI Model - setup similar to previous
# time_steps = 100
# infected = [initial_infected]
# susceptible = [initial_susceptible]
# # Generate samples using normal distribution and clip values
# community_risk_samples = np.random.normal(loc=0.5, scale=0.2, size=time_steps)
# community_risk_samples = np.clip(community_risk_samples, 0, 1)  # Ensure values are within 0 and 1
#
# allowed_samples = np.random.normal(loc=50, scale=15, size=time_steps)
# allowed_samples = np.clip(allowed_samples, 0, 100)  # Ensure values are within 0 and 100
#
# # allowed_samples = np.full(time_steps, 100)
#
# for i in range(1, time_steps):
#     allowed_i = max(1, int(allowed_samples[i]))
#     community_risk_i = max(0, min(community_risk_samples[i], 1))
#     new_infections = get_infected_students_apprx_si(infected[-1], allowed=100, const_1=0.00002, const_2=0.009, community_risk=community_risk_i, total_students=N)
#     new_infected = min(new_infections, allowed_i)
#     new_susceptible = N - new_infected
#     infected.append(new_infected)
#     susceptible.append(new_susceptible)
#
# # Plot results using subplots
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
#
# # SI Model Plot - update with colors
# # ax1.plot(t, solution_si[:, 0], label='S - SI', color="#2B5597")  # Susceptible - blue
# # ax1.plot(t, solution_si[:, 1], label='I - SI', color="#F26178")  # Infected - red
# # ax1.set_title('SI Model')
# # ax1.set_xlabel('Time')
# # ax1.set_ylabel('Number of Individuals')
# # ax1.legend()
# #
# # # # SIR Model Plot - update with colors
# # # ax1.plot(t, solution[:, 0], label='S - SIR', color="#2B5597")  # Susceptible - blue
# # # ax1.plot(t, solution[:, 1], label='I - SIR', color="#F26178")  # Infected - red
# # # ax1.plot(t, solution[:, 2], label='R - SIR', color="#908C13")  # Recovered - olive
# # # ax1.set_title('SIR Model')
# # # ax1.set_xlabel('Time')
# # # ax1.set_ylabel('Number of Individuals')
# # # ax1.legend()
# #
# # Approximate SI Model Plot - update with colors and additional details
# ax2.plot(range(time_steps), susceptible, label='S - Approximate SI', color="#2B5597")  # Susceptible - blue
# ax2.plot(range(time_steps), infected, label='I - Approximate SI', color="#F26178")  # Infected - red
# ax2.plot(range(time_steps), allowed_samples[:time_steps], label='Allowed', linestyle=':', color='grey')  # Allowed - grey
# ax2.set_title('Approximate SI Model')
# ax2.set_xlabel('Time')
# ax2.set_ylabel('Number of Individuals')
# ax2.legend()
#
# plt.suptitle('Comparison of SIR and Approximate SI Models')
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.savefig('sir_vs_si.png')
# plt.show()
# # Hyperparameter Sweep
# allowed_values = np.linspace(0, 100, 3)
# community_risk_values = np.linspace(0.1, 0.9, 10)
# infected_results = np.zeros((len(allowed_values), len(community_risk_values), 100))
#
# for i, allowed in enumerate(allowed_values):
#     for j, community_risk in enumerate(community_risk_values):
#         infected = [5]  # initial infected
#         for time_step in range(1, 100):
#             infected.append(get_infected_students_apprx_si(infected[-1], allowed=allowed, community_risk=community_risk, const_1=0.00001, const_2=0.009,total_students=N))
#         infected_results[i, j, :] = infected
#
# # 3D Surface Plot
# fig = plt.figure(figsize=(14, 10))
# ax = fig.add_subplot(111, projection='3d')
# X, Y = np.meshgrid(allowed_values, community_risk_values)
# Z = infected_results[:, :, -1]  # Get the number of infected at the last time step
#
# # Plot surface
# surf = ax.plot_surface(X, Y, Z.T, cmap='plasma')
# ax.set_xlabel('Allowed')
# ax.set_ylabel('Community Risk')
# ax.set_zlabel('Infected')
# ax.set_title('Impact of Allowed and Community Risk on Infection Spread in Approximate SI Model')
# fig.colorbar(surf, shrink=0.5, aspect=5)
# plt.savefig('3d_surface_plot.png')
#
# # Hyperparameter Sweep for 'allowed' and 'community_risk'
# allowed_values = np.linspace(0, 100, 10)
# community_risk_values = np.linspace(0., 1., 10)
# time_steps = 100
#
# average_infected_results_allowed = []
# average_infected_results_risk = []
#
#
# # Sweep over 'allowed' values
# for allowed in allowed_values:
#     infected_counts = []
#     for _ in range(time_steps):
#         previous_infected = initial_infected
#         for _ in range(100):  # Simulate over time steps
#             previous_infected = get_infected_students_apprx_si(previous_infected, allowed, 0.5, 0.00001, 0.009, N)
#             infected_counts.append(previous_infected)
#     average_infected_results_allowed.append(np.mean(infected_counts))
#
# # Sweep over 'community_risk' values
# for community_risk in community_risk_values:
#     infected_counts = []
#     for _ in range(time_steps):
#         previous_infected = initial_infected
#         for _ in range(100):  # Simulate over time steps
#             previous_infected = get_infected_students_apprx_si(previous_infected, 100, community_risk, 0.00001, 0.009, N)
#             infected_counts.append(previous_infected)
#     average_infected_results_risk.append(np.mean(infected_counts))
#
# # Plotting results for 'allowed'
# plt.figure(figsize=(14, 7))
# plt.subplot(1, 2, 1)
# plt.plot(allowed_values, average_infected_results_allowed, marker='o')
# plt.title('Average Infected vs Allowed')
# plt.xlabel('Allowed')
# plt.ylabel('Average Infected')
#
# # Plotting results for 'community_risk'
# plt.subplot(1, 2, 2)
# plt.plot(community_risk_values, average_infected_results_risk, marker='o')
# plt.title('Average Infected vs Community Risk')
# plt.xlabel('Community Risk')
# plt.ylabel('Average Infected')
#
# plt.tight_layout()
# plt.savefig('sensitivity_analysis-allowed-comm.png')
# plt.show()
#
# # Parameters
# N = 100
# initial_infected = 5
# const_1_values = np.linspace(0.00001, 0.1, 1000)
# const_2_values = np.linspace(0.001, 0.1, 1000)
# time_steps = 100
# allowed_range = (0, 100)  # Allowed can be any value between 0 and 100
# community_risk_range = (0, 1)  # Community risk can be any value between 0 and 1
#
#
# # Sweeping const_1 with random 'allowed' and 'community_risk'
# average_infected_results_const_1 = []
# std_infected_results_const_1 = []
# num_simulations = 100  # Number of simulations for each const_1 value
#
# for const_1 in const_1_values:
#     infected_counts_const_1 = []
#
#     for _ in range(num_simulations):
#         previous_infected = initial_infected
#
#         for _ in range(time_steps):
#             allowed = np.random.uniform(*allowed_range)
#             community_risk = np.random.uniform(*community_risk_range)
#             previous_infected = get_infected_students_apprx_si(previous_infected, allowed, 0.2, const_1, 0.009, N)
#
#         infected_counts_const_1.append(previous_infected)
#
#     average_infected_const_1 = np.mean(infected_counts_const_1)
#     std_infected_const_1 = np.std(infected_counts_const_1)
#     average_infected_results_const_1.append(average_infected_const_1)
#     std_infected_results_const_1.append(std_infected_const_1)
#
# # Sweeping const_2 with random 'allowed' and 'community_risk'
# average_infected_results_const_2 = []
# std_infected_results_const_2 = []
# num_simulations = 100  # Number of simulations for each const_2 value
#
# for const_2 in const_2_values:
#     infected_counts_const_2 = []
#
#     for _ in range(num_simulations):
#         previous_infected = initial_infected
#
#         for _ in range(time_steps):
#             allowed = np.random.uniform(*allowed_range)
#             community_risk = np.random.uniform(*community_risk_range)
#             previous_infected = get_infected_students_apprx_si(previous_infected, allowed, 0.2, 0.00001, const_2, N)
#
#         infected_counts_const_2.append(previous_infected)
#
#     average_infected_const_2 = np.mean(infected_counts_const_2)
#     std_infected_const_2 = np.std(infected_counts_const_2)
#     average_infected_results_const_2.append(average_infected_const_2)
#     std_infected_results_const_2.append(std_infected_const_2)
#
# # Plotting results for const_1
# plt.figure(figsize=(14, 7))
# plt.subplot(1, 2, 1)
# plt.plot(const_1_values, average_infected_results_const_1)
# plt.fill_between(const_1_values,
#                  np.array(average_infected_results_const_1) - 1.96 * np.array(std_infected_results_const_1),
#                  np.array(average_infected_results_const_1) + 1.96 * np.array(std_infected_results_const_1),
#                  alpha=0.2)
# plt.title('Average Infected vs const_1')
# plt.xlabel('const_1')
# plt.ylabel('Average Infected')
#
# # Plotting results for const_2
# plt.subplot(1, 2, 2)
# plt.plot(const_2_values, average_infected_results_const_2)
# plt.fill_between(const_2_values,
#                  np.array(average_infected_results_const_2) - 1.96 * np.array(std_infected_results_const_2),
#                  np.array(average_infected_results_const_2) + 1.96 * np.array(std_infected_results_const_2),
#                  alpha=0.2)
# plt.title('Average Infected vs const_2')
# plt.xlabel('const_2')
# plt.ylabel('Average Infected')
#
# plt.tight_layout()
# plt.savefig('sensitivity_analysis-const1-const2.png')


def simulate_epidemic_peak(initial_infected, allowed, community_risk, const_1, const_2, total_students=100):
    global current_infected
    previous_infected = initial_infected
    # recovered = 0.1 * previous_infected
    for step in range(100):  # Simulate over some time steps
        susceptible = max(0, total_students - previous_infected)
        new_infected_inside = int(
            const_1 * previous_infected * allowed * (susceptible/ total_students) * susceptible)
        new_infected_outside = int(const_2 * community_risk * allowed * susceptible)
        recovered = max(int(1.0 * previous_infected), 0)
        total_infected = min(new_infected_inside + new_infected_outside, allowed)
        current_infected = min(previous_infected + int(total_infected) - recovered, allowed)
    return current_infected


# Simulation parameters
community_risk_values = np.linspace(0, 1, 10)
allowed_values = np.linspace(0, 100, 3)
results = np.zeros((len(community_risk_values), len(allowed_values)))
const_1 = 0.1  # reduce this to a smaller value
const_2 = 0.1 # reduce this value to be very small 0.01, 0.02
# Sweep over 'allowed' and 'community_risk'
for i, community_risk in enumerate(community_risk_values):
    for j, allowed in enumerate(allowed_values):
        peak_infected = simulate_epidemic_peak(0, allowed, community_risk, const_1, const_2, 100)
        results[i, j] = peak_infected
        # print(f"Community Risk: {i}, Allowed: {j}, Peak Infected: {peak_infected}")

# Round off the results to the nearest integer
results = np.round(results).astype(int)
# print(results)

# Plotting and saving the heatmap
plt.figure(figsize=(12, 10))
ax = sns.heatmap(results[::-1],
                 xticklabels=np.round(allowed_values).astype(int),
                 yticklabels=np.round(community_risk_values[::-1], 2),
                 annot=True, fmt="d", cmap="plasma",
                 annot_kws={"size": 14})  # Increase font size of the annotations
ax.set_title('Heatmap of Peak Infections', fontsize=20)
ax.set_xlabel('Allowed', fontsize=16)
ax.set_ylabel('Community Risk', fontsize=16)
ax.tick_params(axis='x', labelsize=14)  # Increase font size of x-axis tick labels
ax.tick_params(axis='y', labelsize=14)  # Increase font size of y-axis tick labels
plt.tight_layout()  # Ensure the layout fits without cutting off parts of the plot
plt.savefig('heatmap_peak_infections.png')
plt.show()
plt.close()