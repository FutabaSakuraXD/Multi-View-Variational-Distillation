
import scipy.io as scio
import numpy
path_RGB = "/home/txd/Variational Distillation/cur_Best_Ablation_Exp11_8/observation/cam1.mat"
path_IR = "/home/txd/Variational Distillation/cur_Best_Ablation_Exp11_8/observation/cam1.mat"

z_sh_RGB = data = scio.loadmat(path_RGB)
z_sh_IR = data = scio.loadmat(path_IR)
for i in range(333): print("ID: " + str(i) + "   shape:" + str(z_sh_RGB["feature_test"][0][i].shape))

from time import time
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

from sklearn import manifold

index = [309, 308, 305, 304, 294, 292, 278, 275, 218, 211,
         200, 199, 198, 187, 183, 181, 180, 177, 176, 173,
          142, 134, 131, 114, 113, 136, 163]

colors = numpy.ndarray
feats = numpy.ndarray
c_list, f_list = [], []
for i in range(len(index)):
    if i == 0:
        feats = z_sh_RGB["feature_test"][0][index[i]] * 1 + z_sh_IR["feature_test"][0][index[i]] * 0
        f_list.append(feats)
        colors = numpy.ndarray((feats.shape[0],))
        colors[:] = i
        c_list.append(colors)
    else:
        feat = z_sh_RGB["feature_test"][0][index[i]] * 1 + z_sh_IR["feature_test"][0][index[i]] * 0
        f_list.append(feat)
        color = numpy.ndarray((feat.shape[0],))
        color[:] = i
        c_list.append(color)
        feats = numpy.concatenate([feats, feat])
        colors = numpy.concatenate([colors, color])
print(feats.shape)
print(colors.shape)
feats[:][:] *= 10000

'''t-SNE'''
t0 = time()
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
Y = tsne.fit_transform(feats) # Y.__class__=ndarray, shape of which is (feats.shape[0], 2)

t1 = time()
print("t-SNE: %.2g sec" % (t1 - t0))
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
res = plt.scatter(Y[:, 0], Y[:, 1], s=160, c=colors, cmap=plt.cm.Spectral, alpha=0.5)
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
#plt.grid(True)

plt.savefig("RGB")
plt.show()




