import numpy as np
import matplotlib.pyplot as plt
q_table = np.load('200-qtable.npy')


plt.imshow(q_table, interpolation='nearest', aspect='0.05')

plt.title('Q-table')
plt.colorbar()
plt.show()
print(q_table)