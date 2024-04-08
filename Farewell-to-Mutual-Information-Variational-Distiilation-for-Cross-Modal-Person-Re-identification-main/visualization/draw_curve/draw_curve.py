import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import style
print(plt.style.available)
plt.style.use('fivethirtyeight')

font_legend = fm.FontProperties(fname='/usr/share/fonts/truetype/Times/TIMES.ttf', size=16)
font_label = fm.FontProperties(fname='/usr/share/fonts/truetype/Times/TIMES.ttf', size=16+4)
signal = 3
file_names = ["./dim=64", "./dim=128", "./dim=256", "./dim=384"]
batch_factors = [64, 128, 256, 384]
vsd_losses = []
for file_name in file_names:
    vsd_loss = []
    with open(file_name, 'r') as f:
        count = 0
        for line in f:
            count += 1
            if line.find("Epoch") != 0: continue
            if count % signal != 0: continue
            # print(line[line.find("VSD Loss")+17:line.find("VSD Loss")+23])
            #vsd_loss.append(-64 * float(line[line.find("VSD Loss")+17:line.find("VSD Loss")+23]))
            vsd_loss.append(-2 * 64 * float(line[line.find("VML Loss") + 17:line.find("VML Loss") + 23]))
    vsd_loss = np.asarray(vsd_loss)
    vsd_losses.append(vsd_loss)
#new_vsd_loss = np.linspace(vsd_loss.min(), vsd_loss.max(), vsd_loss.shape[0])
epoch = []
for i in range(1, vsd_losses[0].shape[0] + 1):
    epoch.append(i*5/2)
epoch = np.asarray(epoch)

fig = plt.figure(figsize=(8, 5), frameon=False, facecolor="#DCDCDC")


l1, = plt.plot(epoch, vsd_losses[0], color='#1F77B4', linewidth=2.0, label='2x64-D',  marker=",")
l2, = plt.plot(epoch, vsd_losses[1], color='#FF7F0E', linewidth=2.0, label='2x128-D', marker=",")
l3, = plt.plot(epoch, vsd_losses[2], color='#9467BD', linewidth=2.0, label='2x256-D', marker=",")
l3, = plt.plot(epoch, vsd_losses[3], color='#D62728', linewidth=2.0, label='2x384-D', marker=",")

plt.xlabel("Epoch", fontproperties=font_label)
plt.tight_layout()
plt.grid(False)
plt.legend(loc = 'upper right', prop=font_legend)
plt.savefig("VCD.svg", format="svg")
plt.show()