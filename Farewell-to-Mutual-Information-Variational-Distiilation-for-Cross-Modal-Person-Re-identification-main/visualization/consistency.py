
import scipy.io as scio
import numpy
path_RGB = "/home/txd/Variational Distillation/cur_Best_Ablation_Exp11_8/z_sh_RGB/cam1.mat" # VCD
path_IR = "/home/txd/Variational Distillation/cur_Best_Ablation_Exp11_8/z_sh_IR/cam1.mat" # VCD
#path_RGB = "/home/txd/Variational Distillation/Exps/FSE_6/specific_z/cam1.mat" # Conventional IB
#path_IR = "/home/txd/Variational Distillation/Exps/FSE_6/shared_z/cam1.mat" # Conventional IB

z_sh_RGB = data = scio.loadmat(path_RGB)
z_sh_IR = data = scio.loadmat(path_IR)
#for i in range(333): print("ID: " + str(i) + "   shape:" + str(z_sh_RGB["feature_test"][0][i].shape))

from time import time
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from matplotlib import style
print(plt.style.available)
plt.style.use('fivethirtyeight')

from sklearn import manifold

index = [309, 308, 305, 304, 294, 292, 278, 275, 218,
         200, 199, 198, 187, 183, 181, 180, 177, 176, 173,
          142, 134, 131, 113, 136, 163] # 114, 211

colors_IR = numpy.ndarray
feats_IR = numpy.ndarray
c_list, f_list = [], []
for i in range(len(index)):
    if i == 0:
        feats_IR = z_sh_IR["feature_test"][0][index[i]]
        f_list.append(feats_IR)
        colors_IR = numpy.ndarray((feats_IR.shape[0],))
        colors_IR[:] = i
        c_list.append(colors_IR)
    else:
        feat = z_sh_IR["feature_test"][0][index[i]]
        f_list.append(feat)
        color = numpy.ndarray((feat.shape[0],))
        color[:] = i
        c_list.append(color)
        feats_IR = numpy.concatenate([feats_IR, feat])
        colors_IR = numpy.concatenate([colors_IR, color])
colors_RGB = numpy.ndarray
feats_RGB = numpy.ndarray
for i in range(len(index)):
    if i == 0:
        feats_RGB = z_sh_RGB["feature_test"][0][index[i]]
        f_list.append(feats_RGB)
        colors_RGB = numpy.ndarray((feats_RGB.shape[0],))
        colors_RGB[:] = i
        c_list.append(colors_RGB)
    else:
        feat = z_sh_RGB["feature_test"][0][index[i]]
        f_list.append(feat)
        color = numpy.ndarray((feat.shape[0],))
        color[:] = i
        c_list.append(color)
        feats_RGB = numpy.concatenate([feats_RGB, feat])
        colors_RGB = numpy.concatenate([colors_RGB, color])
#feats = numpy.concatenate([feats_IR,feats_RGB])
#colors = numpy.concatenate([colors_IR,colors_RGB])
#print(feats.shape)
#print(colors.shape)
#feats[:][:] *= 10000
feats_IR[:][:] *= 4500
feats_RGB[:][:] *= 4500

'''t-SNE'''
t0 = time()
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
#Y = tsne.fit_transform(feats) # Y.__class__=ndarray, shape of which is (feats.shape[0], 2)
Y_IR = tsne.fit_transform(feats_IR)
Y_RGB = tsne.fit_transform(feats_RGB)


t1 = time()
print("t-SNE: %.2g sec" % (t1 - t0))
fig = plt.figure(figsize=(8, 8), frameon=False, facecolor="grey")
ax = fig.add_subplot(1, 1, 1)
res_IR =    plt.scatter(Y_IR[:, 0], Y_IR[:, 1],  c="#6983B2", cmap=plt.cm.Spectral, alpha=1, s=80)
res_RGB = plt.scatter(Y_RGB[:, 0], Y_RGB[:, 1],  c="#F4B183", cmap=plt.cm.Spectral, alpha=1, s=80)
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.grid(True)
plt.savefig("VCD_Fusion.svg", format="svg")
plt.show()




