import numpy as np
import glob

import matplotlib.pyplot as plt

data = np.loadtxt("/home/ws/exp_data/exp1_data_1.txt", delimiter=",")

x = np.arange(0, 0.05 * len(data), 0.05)

plt.figure(figsize=(20, 5))
plt.plot(data[:, 0], data[:, 3], label='Drone X Position')
plt.xlabel('Time [s]')
plt.ylabel('Z Position [m]')
plt.hlines(1.5, 0, data[-1, 0], colors='r', linestyles='dashed')
plt.grid()
plt.savefig('/home/ws/exp_data/exp1_data_1.png')
plt.show()