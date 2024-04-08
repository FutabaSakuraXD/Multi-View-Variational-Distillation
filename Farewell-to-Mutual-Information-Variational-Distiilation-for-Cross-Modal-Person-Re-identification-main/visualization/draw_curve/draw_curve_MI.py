import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import spline

signal =10
targets = ["Specific MI", "Shared MI"]
mutual_information = []
for target in targets:
    MI = []
    with open("./MI", 'r') as f:
        count = 0
        for line in f:
            count += 1
            if line.find("Epoch") != 0: continue
            if count % signal != 0: continue
            if target == "Specific MI":
                MI.append(-2 * 64 * float(line[line.find(target) + 20: line.find(target) + 26]))
            elif target == "Shared MI":
                MI.append(-2 * 64 * float(line[line.find(target) + 18: line.find(target) + 24]))
    MI = np.asarray(MI)
    mutual_information.append(MI)
#new_vsd_loss = np.linspace(vsd_loss.min(), vsd_loss.max(), vsd_loss.shape[0])
epoch = []
for i in range(1, mutual_information[0].shape[0] + 1):
    epoch.append(i*5)
epoch = np.asarray(epoch)
fig = plt.figure(figsize=(6, 6))
l1, = plt.plot(epoch, mutual_information[0], color='#FFB6C1', linewidth=2.0, linestyle='-.', label='2x64-D', marker=",")
l2, = plt.plot(epoch, mutual_information[1], color='#87CEFA', linewidth=2.0, linestyle='-.', label='2x128-D', marker=",")

plt.legend(loc = 'upper right')
plt.savefig("VCD")
plt.show()