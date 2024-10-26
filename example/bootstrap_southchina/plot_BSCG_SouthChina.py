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
set = 'SouthChina'
test = 'T1'
# 密度范围
rhomin, rhomax = -1, 1
boundary = [rhomin, rhomax]
# profile
profile = 2

# 创建文件夹
if not os.path.exists('picture'):
    os.mkdir('picture')
if not os.path.exists('picture/' + set + test):
    os.mkdir('picture/' + set + test)

# 读入初始模型、carve的index、mzs;
with open('modeldata/' + set + test + '/initial_model_mesh.txt') as f:
    rho = np.loadtxt(f, usecols=[0])
with open('modeldata/' + set + test + '/maskindex.txt') as f:
    mask = np.loadtxt(f, usecols=[0])
with open('modeldata/' + set + test + '/mzs.txt') as f:
    mzs = np.loadtxt(f, usecols=[0])
# 读入模型剖分
with open('modeldata/' + set + test + '/mshape.txt') as f:
    nz, ny, nx = np.loadtxt(f)
nz, ny, nx = int(nz), int(ny), int(nx)
# 读入反演结果
with open('result/' + set + test + '/inversion_model.txt') as f:
    model_inv_all = np.loadtxt(f)
with open('result/' + set + test + '/data_misfit.txt') as f:
    data_misfit_all = np.loadtxt(f)
with open('result/' + set + test + '/model_misfit.txt') as f:
    model_misfit_all = np.loadtxt(f)
with open('result/' + set + test + '/regul_factor.txt') as f:
    regul_factor_all = np.loadtxt(f)


# 将密度值转换回规则网格rho_mesh
model_inv_all_mesh = np.zeros((model_inv_all.shape[0], nz*ny*nx))
for i in range(model_inv_all.shape[0]):
    model_inv_all_mesh[i, :] = utils.carve2rho(model_inv_all[i, :], rho, mask)
# 计算每一个网格点密度值的均值和方差
model_inv_mean = np.mean(model_inv_all_mesh, axis=0)
model_inv_std = np.std(model_inv_all_mesh, axis=0)
rho_std_min, rho_std_max = model_inv_std.min(), model_inv_std.max()
print(rho_std_min, rho_std_max)
# ---------------画出模型均值
# model_inv = model_inv_all[3, :]
# 反演结果切片图
rhoInv3D_mean = model_inv_mean.reshape(nz, ny, nx)
plt.figure(figsize=(20, 10))
for i in range(nz):
    plt.suptitle('inversion model - mean')
    ax = plt.subplot(np.ceil(nz/10.0), 10, i + 1)
    plt.title("{:.2f}m".format(mzs[i + 1]))
    plt.imshow(rhoInv3D_mean[i, :, :], vmin=rhomin, vmax=rhomax, cmap=plt.cm.jet)
    plt.xticks(np.arange(0, nx, 10))
    plt.yticks(np.arange(0, ny, 10))
    ax.invert_yaxis()
    plt.colorbar()

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
plt.savefig('picture/' + set + test + '/invmodel-mean.png', dpi=500, bbox_inches='tight')
#plt.show()

# 模型均值的剖面图
plt.figure()
plt.title('mean model profile: y(layer) = {}'.format(profile))
plt.imshow(rhoInv3D_mean[:, profile, :], vmin=rhomin, vmax=rhomax, cmap=plt.cm.jet)
plt.colorbar()
plt.xticks(np.arange(0, nx, 10))
plt.yticks(np.arange(0, nz, 5))
plt.xlabel("x (layer)")
plt.ylabel("z (layer)")
# plt.axis('equal')

plt.savefig('picture/' + set + test + '/invmodel-mean-profile.png', dpi=500, bbox_inches='tight')
#plt.show()

# ---------------画出模型方差
# model_inv = model_inv_all[3, :]
# 反演结果切片图
rhoInv3D_std = model_inv_std.reshape(nz, ny, nx)
plt.figure(figsize=(20, 10))
for i in range(nz):
    plt.suptitle('inversion model - std')
    ax = plt.subplot(np.ceil(nz/10.0), 10, i + 1)
    plt.title("{:.2f}m".format(mzs[i + 1]))
    plt.imshow(rhoInv3D_std[i, :, :], vmin=rho_std_min, vmax=rho_std_max, cmap=plt.cm.jet)
    plt.xticks(np.arange(0, nx, 10))
    plt.yticks(np.arange(0, ny, 10))
    ax.invert_yaxis()
    plt.colorbar()

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
plt.savefig('picture/' + set + test + '/invmodel-mean.png', dpi=500, bbox_inches='tight')
#plt.show()

# 模型方差的剖面图
plt.figure()
plt.title('std model profile: y(layer) = {}'.format(profile))
plt.imshow(rhoInv3D_std[:, profile, :], vmin=rho_std_min, vmax=rho_std_max, cmap=plt.cm.jet)
plt.colorbar()
plt.xticks(np.arange(0, nx, 10))
plt.yticks(np.arange(0, nz, 5))
plt.xlabel("x (layer)")
plt.ylabel("z (layer)")
# plt.axis('equal')

plt.savefig('picture/' + set + test + '/invmodel-mean-profile.png', dpi=500, bbox_inches='tight')
#plt.show()
plt.close()

# ------某一个点的密度的概率分布
plt.figure()
plt.hist(model_inv_all_mesh[:, 800], bins=50, range=[rhomin, rhomax])
# plt.scatter(model_inv_all_mesh[:, 0])

plt.xlabel("samples")
plt.ylabel("frequency")
plt.savefig('picture/' + set + test + '/hist-OnePoint.png', dpi=500, bbox_inches='tight')
plt.show()


# ----------------误差曲线
# 某一次采样的曲线情况
nsample = 3
plt.figure()
plt.suptitle('misfit of one sample')
plt.subplot(221)
plt.title("data_misfit")
plt.plot(data_misfit_all[nsample, :])

plt.subplot(222)
plt.title("model_misfit")
plt.plot(model_misfit_all[nsample, :])

plt.subplot(223)
plt.title("regul_factor")
plt.plot(regul_factor_all[nsample, :])

plt.xlabel("iteration")

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
plt.savefig('picture/' + set + test + '/misfit-OneSample.png', dpi=500, bbox_inches='tight')
plt.show()

# 所有采样的曲线情况
plt.figure()
plt.suptitle('misfit of all samples')
plt.subplot(121)
plt.title("data_misfit")
plt.plot(data_misfit_all[:, -1])

plt.subplot(122)
plt.title("model_misfit")
plt.plot(model_misfit_all[:, -1])

plt.xlabel("samples")

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
plt.savefig('picture/' + set + test + '/misfit-AllSamples.png', dpi=500, bbox_inches='tight')
plt.show()