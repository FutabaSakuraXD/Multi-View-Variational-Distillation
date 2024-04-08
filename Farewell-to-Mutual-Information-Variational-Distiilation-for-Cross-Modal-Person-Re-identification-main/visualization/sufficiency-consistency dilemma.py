import os
import random

import scipy.io as scio
import numpy


from time import time
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

from sklearn import manifold

#path_RGB = "/home/txd/Variational Distillation/Extra_Final_Exp4++/z_SP_RGB/cam1.mat" # VSD
#path_IR = "/home/txd/Variational Distillation/Extra_Final_Exp4++/z_SP_IR/cam1.mat" # VSD
path_RGB = "../MM01/Exp3(num_instance=8,batch=128)/v_ms_representation_cam6.mat" # Conventional IB
path_IR  = "../MM01/Exp3(num_instance=8,batch=128)/i_ms_representation_cam6.mat" # Conventional IB

path_RGB = "../MM01/Exp3(num_instance=8,batch=128)/observationcam6.mat" # Conventional IB
path_IR  = "../MM01/Exp3(num_instance=8,batch=128)/observationcam6.mat" # Conventional IB

#path_RGB = "../MM01/Exp2(num_instance=8)/observationcam1.mat"
#path_IR  = "../MM01/Exp2(num_instance=8)/observationcam1.mat"

#path_RGB = "../MM01/Exp3(num_instance=8,batch=128)/observationcam1.mat"

z_sh_IR  = scio.loadmat(path_IR)
z_sh_RGB = scio.loadmat(path_RGB)#.update(z_sh_IR)

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

index = [305, 304, 298, 296, 295, 292, 282, 279, 278, 275,
         269, 247, 241, 239, 227, 224, 217]
index = [103, 272, 259, 103, 233, 179, 115, 172, 313, 236,
         214, 140, 282, 122, 249, 243, 96, 257, 82, 264,
         286, 12, 289, 171, 312, 132, 17, 133, 186, 290,
         250, 330, 292, 253, 315, 1, 156, 142, 141, 195]
#index = [i for i in range(0, 332)]
index = [random.randint(0,332) for i in range(20)]
random.shuffle(index)
# 只注重sufficiency使得同视图但不同标签的的数据分布集中, 三个集中区域代表三个视图
#index = [168, 11, 166, 112, 202, 250, 33, 170, 21, 280, 135, 221, 306, 327, 111, 233, 332, 157, 94]

# 只注重consistency, 使得判别性能骤降, 最终导致所有类别的样本混淆在一起
#index = [39, 214, 68, 50, 103, 104, 223, 85, 98, 43, 163, 84, 92, 287, 332, 280, 31]
# 当然大概率还需要展示一组ours作为对比
#index = [42, 304, 231, 207, 77, 250, 252, 242, 108, 205, 61, 116, 179, 192, 54, 18]
index = [27, 1, 41, 198, 78, 173, 297, 276, 129, 286, 173, 301, 288, 123, 332, 89, 108, 53]
random.shuffle(index)
print(index)

colors = numpy.ndarray

feats = numpy.ndarray
c_list, f_list = [], []
ID_count = 1
valid_idx = []
for i in range(0, len(index)):
    if i == 0:
        feats = z_sh_RGB["feature_test"][0][index[i]] * 0 + z_sh_IR["feature_test"][0][index[i]] * 1
        f_list.append(feats)
        colors = numpy.ndarray((feats.shape[0],))
        colors[:] = i
        c_list.append(colors)
        valid_idx.append(index[i])
    else:
        feat = z_sh_RGB["feature_test"][0][index[i]] * 0 + z_sh_IR["feature_test"][0][index[i]] * 1
        if feat.shape[-1]==0: continue
        f_list.append(feat)
        color = numpy.ndarray((feat.shape[0],))
        color[:] = i
        c_list.append(color)
        feats = numpy.concatenate([feats, feat])
        colors = numpy.concatenate([colors, color])
        ID_count += 1
        valid_idx.append(index[i])
    #if ID_count == 10: break
#print(colors)
print(feats.shape)
print(colors.shape)
print(max(colors))
print("utilized idx:" + str(valid_idx))
feats[:][:] *= 5000

'''t-SNE'''
t0 = time()
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
Y = tsne.fit_transform(feats) # Y.__class__=ndarray, shape of which is (feats.shape[0], 2)Y

t1 = time()
print("t-SNE: %.2g sec" % (t1 - t0))
fig = plt.figure(figsize=(8, 8), frameon=False)
ax = fig.add_subplot(1, 1, 1)
#res = plt.scatter(Y[:, 0], Y[:, 1], s=160, c=colors, cmap=plt.cm.Spectral, alpha=0.75, marker='o')

res = plt.scatter(Y[0:114, 0], Y[0:114, 1], s=160, c=colors[0:114], cmap=plt.cm.Spectral, alpha=0.75, marker='o')
res = plt.scatter(Y[114:228, 0], Y[114:228, 1], s=160, c=colors[114:228], cmap=plt.cm.Spectral, alpha=0.75, marker='P')
res = plt.scatter(Y[228:, 0], Y[228:, 1], s=160, c=colors[228:], cmap=plt.cm.Spectral, alpha=0.75, marker='D')

ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())

#plt.title(str(valid_idx), fontdict=font)
#®plt.yticks(range(10, 100, 10))
#plt.xticks(range(0, 20, 1))
plt.grid(True, linestyle='--', alpha=1)

plt.savefig("/home/txd/visualization/ours.svg", format="svg", bbox_inches='tight')
plt.show()




