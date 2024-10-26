'''
plot inversion result
'''
import sys
sys.path.append("../..")

import os
import timeit
import matplotlib.pyplot as plt
import numpy as np
import mesher, gridder, utils
from vis import mpl
from gravmag import prism

# parameters
set = 'model04_complex'
test = 'T1'
# # 密度范围
rhomin, rhomax = 0, 1
boundary = [rhomin, rhomax]

# 创建文件夹
if not os.path.exists('picture'):
    os.mkdir('picture')
if not os.path.exists('picture/' + set + test):
    os.mkdir('picture/' + set + test)

# 读入模型剖分
with open('modeldata/' + set + test + '/mshape.txt') as f:
    nz, ny, nx = np.loadtxt(f)
nz, ny, nx = int(nz), int(ny), int(nx)


# 读入反演结果
with open('result/' + set + test + '/inversion_model.txt') as f:
    model_inv_all = np.loadtxt(f)

# 取出其中一个模型
model_inv = np.mean(model_inv_all, axis=0)
# model_inv = model_inv_all[3, :]
# 反演结果切片图
rhoInv3D = model_inv.reshape(nz, ny, nx)
plt.figure(figsize=(10, 2))
for i in range(nz):
    plt.suptitle('inversion model')
    ax = plt.subplot(2, 5, i+1)
    plt.imshow(rhoInv3D[i, :, :], vmin=0, vmax=1)
    plt.xticks(np.arange(0, nx, 10))
    plt.yticks(np.arange(0, ny, 10))
    ax.invert_yaxis()
    plt.colorbar()

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
plt.savefig('picture/' + set + test + '/invmodel.png', dpi=500, bbox_inches='tight')
plt.show()
