"""
Colormaps options for visualizing the q_table
https://matplotlib.org/stable/tutorials/colors/colormaps.html
"""
from matplotlib import pyplot as plt
import numpy as np
img_array = np.load('qtables/10-qtable.npy')
print(img_array.shape)
plt.imshow(img_array, cmap='tab20b')
plt.show()
