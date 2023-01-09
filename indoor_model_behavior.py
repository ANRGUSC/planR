import numpy as np
import matplotlib.pyplot as plt
from campus_digital_twin import campus_state

p_initial = np.arange(0.0, 1.0, 0.01)
room_capacity = [25, 50, 75, 100]
p_infection_to_plot = []
for j in room_capacity:
    p_infection = []
    for i in p_initial:
        p_infection.append(round(campus_state.calculate_indoor_infection_prob(j, i), 4))
    p_infection_to_plot.append(p_infection)

plt.xlabel("p_initial")
plt.ylabel("p_infection")
plt.title("Indoor infection model behavior  D0=10")

for i, v in enumerate(p_infection_to_plot):
    action = (i + 1) * 25
    plt.plot(p_initial, v, label='%s %% allowed' %action)

plt.legend()
plt.show()
