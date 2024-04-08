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
#path_RGB = "../MM01/Exp6(384*192)/v_observationcam1.mat" # Conventional IB
#path_IR  = "../MM01/Exp6(384*192)/i_observationcam1.mat" # Conventional IB

#path_RGB = "../MM01/Exp3(num_instance=8,batch=128)/v_ms_representation_cam1.mat" # Conventional IB
#path_IR  = "../MM01/Exp3(num_instance=8,batch=128)/i_ms_representation_cam1.mat" # Conventional IB

path_RGB = "../MM01/Exp2(num_instance=8)/observationcam1.mat"
path_IR  = "../MM01/Exp2(num_instance=8)/observationcam1.mat"

path_RGB = "../MM01/Exp3(num_instance=8,batch=128)/observationcam1.mat"

z_sh_IR  = scio.loadmat(path_IR)
z_sh_RGB = scio.loadmat(path_RGB)#.update(z_sh_IR)

font = {'family' : 'Times New Roman',
         #'style': 'italic',
         'weight': 'bold',
        #'color':  'darkred',
        'size': 10,}

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

index = [305, 304, 298, 296, 295, 292, 282, 279, 278, 275,
         269, 247, 241, 239, 227, 224, 217]
index = [331, 330, 329, 325, 324, 320, 318, 315, 311, 306,
         305, 304, 298, 296, 295, 292, 282, 279, 278, 275,
         269, 247, 241, 239, 227, 224, 217]
#index = [i for i in range(0, 332)]
index = [random.randint(0,332) for i in range(30)]
random.shuffle(index)
print(index)
#index = [331, 330, 329, 325, 324, 320,]
#index = [247, 269, 324, 329, 304]
# 316 and 318 are fairly close
# 307, 306, 305, 297, 295, 294, 292, 282, 280, 278, 277, 271, 269, 248, 243, 241, 224, 218 maybe a choice
# [103, 272, 259, 103, 233, 179, 115, 172, 313, 236, 214, 140, 282, 122, 249, 243, 96, 257, 82, 264, 286, 12, 289, 171, 312, 132, 17, 133, 186, 290, 250, 330, 292, 253, 315, 1, 156, 142, 141, 195]
# either-or: 296 and 297
#index = [293, 276, 44, 278, 53, 91, 206, 252, 168, 113, 218], 304, 63, 239, 136, 89, 229, 213, 257, 83, 250, 228, 193, 231, 243, 136, 77, 316]
idx = [293, 276, 278, 53, 91, 206, 252, 168, 113, 218]#, 63, 136, 257, 193, 231, 243, 331, 330, 329, 324]
random.shuffle(idx)
index = [idx[i] for i in range(5)]
#index = [206, 218, 113, 168, 53, 293, 276, 278, 91, 252]
index = [252, 276, 53, 168, 113, 293, 278, 91, 206, 218]#, 63, 136, 257, 193, 231]#, 243, 331, 330, 329, 324]
#index = [53, 293, 252, 206, 276, 278, 91, 168, 113, 218]

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
feats[:][:] *= 3166

'''t-SNE'''
t0 = time()
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
Y = tsne.fit_transform(feats) # Y.__class__=ndarray, shape of which is (feats.shape[0], 2)Y

t1 = time()
print("t-SNE: %.2g sec" % (t1 - t0))
fig = plt.figure(figsize=(8, 8), frameon=False)
ax = fig.add_subplot(1, 1, 1)
res = plt.scatter(Y[:, 0], Y[:, 1], s=160, c=colors, cmap=plt.cm.Spectral, alpha=0.75)
#for i in range(feats.shape[0]):
#    plt.annotate(index[int(colors[i]-1)],xy=(Y[i][0], Y[i][1]),xytext=(Y[i][0]+0.1, Y[i][1]+0.1))
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())

#plt.title(str(valid_idx), fontdict=font)
#®plt.yticks(range(10, 100, 10))
#plt.xticks(range(0, 20, 1))
plt.grid(True, linestyle='--', alpha=1)

plt.savefig("/home/txd/visualization/only-consistency", format="svg", bbox_inches='tight')
plt.show()




