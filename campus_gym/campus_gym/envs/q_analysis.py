"""
Colormaps options for visualizing the q_table
https://matplotlib.org/stable/tutorials/colors/colormaps.html
"""
from matplotlib import pyplot as plt
import numpy as np
state_action_array = np.load('qtables/Test1-9999-qtable.npy')
print(state_action_array)
plt.imshow(state_action_array, cmap='tab20b')
plt.show()


