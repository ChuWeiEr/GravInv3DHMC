# 03-台阶模型
# 模型空间4000m*3000m*1000m；网格个数40*30*10；间距100m
# 密度差1g/cm3
# 输出文件：正演重力数据
# 注：球棱柱正演的单位-经纬度；m；g/cm3

import sys
sys.path.append("..")
import os
import numpy as np
import mesher, utils
from gravmag import prism
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from vis import mpl, myv
# plt.rcParams['axes.linewidth'] = 1  # 图框宽度
# plt.rcParams['figure.dpi'] = 300  # plt.show显示分辨率
font = {'family': 'serif',
        'serif': 'Times New Roman',
        'weight': 'normal',
        'size': 15}
plt.rc('font', **font)
# 名字
set = 'model03_twodykes'
# 创建文件夹
if not os.path.exists('modeldata'):
    os.mkdir('modeldata')
if not os.path.exists('picture'):
    os.mkdir('picture')
if not os.path.exists('picture/OriginalModel'):
    os.mkdir('picture/OriginalModel')

# 剖面位置
profile = 15
# 取值范围
rhomin, rhomax = 0, 1
# 网格大小
dx, dy, dz = 100, 100, 100
# 小棱柱
nx, ny, nz = 30, 40, 10
# 研究区范围
xmin, xmax, ymin, ymax, zmin, zmax = 0, nx*dx, 0, ny*dy, 0, nz*dz
mesh_rho = mesher.PrismMesh(bounds=(xmin, xmax, ymin, ymax, zmin, zmax), spacing=(dz, dy, dx))
rho = np.zeros([nx * ny * nz])

# 密度按照棱柱排，先x再y再z
for iz in range(nz):
    for iy in range(ny):
        for ix in range(nx):
            # 左
            if iz + 8 <= iy <= iz + 10 and 14 <= ix <= 16 and 1 <= iz <= 3:
                index = nx*ny*iz + nx*iy + ix
                rho[index] = rhomax
            # 右
            if -1 * iz + 24 <= iy <= -1 * iz + 32 and 11 <= ix <= 19 and 2 <= iz <= 7:
                index = nx * ny * iz + nx * iy + ix
                rho[index] = rhomax

mesh_rho.addprop('density', rho)
# 保存模型数据
np.savetxt("modeldata/{}_rho.dat".format(set), rho, fmt='%.5f', delimiter=' ')

rho3D = rho.reshape(nz, ny, nx)

# 模型切片图
# ---------rho
plt.figure(figsize=(10, 6))
ax = plt.subplot()
plt.pcolor(rho3D[2, :, :].T, vmin=rhomin, vmax=rhomax, cmap=plt.cm.jet)
cb = plt.colorbar()
cb.set_label('$g/cm^3$')
plt.xticks(np.arange(0, ny + 1, 10), np.arange(0, (ny + 1) * dy, 10 * dy))
plt.yticks(np.arange(0, nx + 1, 10), np.arange(0, (nx + 1) * dx, 10 * dx))
plt.gca().set_aspect('equal', adjustable='box')
plt.xlim([0, ny])
plt.ylim([0, nx])
plt.xlabel("Easting(m)")
plt.ylabel("Northing(m)")

plt.subplots_adjust(left=None, bottom=None, right=0.9, top=None, wspace=0.5, hspace=0.5)
plt.savefig('picture/OriginalModel/{}-1layer.png'.format(set), dpi=500, bbox_inches='tight')
plt.savefig('picture/OriginalModel/{}-1layer.eps'.format(set), dpi=500, bbox_inches='tight')
#plt.show()

# 圈出模型的外围边框--剖面
# -------rho
plt.figure(figsize=(10, 2))
ax = plt.subplot()
plt.pcolor(rho3D[:, :, profile], vmin=rhomin, vmax=rhomax, cmap=plt.cm.jet)
cb = plt.colorbar()
cb.set_label('$g/cm^3$')
plt.xticks(np.arange(0, ny+1, 10), np.arange(0, (ny+1)*dy, 10*dy))
plt.yticks(np.arange(0, nz+1, 5), np.arange(0, (nz+1)*dz, 5*dz))
plt.xlim([0, ny])
plt.ylim([0, nz])
plt.xlabel("Easting(m)")
plt.ylabel("Depth(m)")
ax.invert_yaxis()

plt.savefig('picture/OriginalModel/{}-profile-rho.png'.format(set), dpi=500, bbox_inches='tight')
plt.savefig('picture/OriginalModel/{}-profile-rho.eps'.format(set), dpi=500, bbox_inches='tight')
#plt.show()

# # 模型3D作图
# myv.figure()
# plot = myv.prisms(mesh_rho, prop='density')
# axes = myv.axes(myv.outline(), fmt='%.0f')
# myv.savefig('picture/OriginalModel/{}_3Dmodel.png'.format(set))
# myv.show()

# 正演--40*30个观测点
shape = (nx, ny)
xp, yp, zp = utils.regular((xmin, xmax, ymin, ymax), shape, z=0)
gz_pre, _ = prism.gz(xp, yp, zp, mesh_rho)
# -------gz------给正演数据加上高斯噪声,当做观测数据
noise_gz = np.random.normal(loc=0, scale=0.02 * gz_pre.max(), size=gz_pre.shape[0])
gz_noise = gz_pre + noise_gz
# 保存数据
np.savetxt('modeldata/{}_gz_noise.txt'.format(set), np.c_[xp, yp, zp, gz_noise], fmt="%.5f", delimiter=" ")


# ----正演结果画图
# ----gz with noise
plt.figure(figsize=(10, 5))
plt.axis('scaled')
mpl.contourf(yp, xp, gz_noise, shape, 15)
plt.xlabel("Easting(m)")
plt.ylabel("Northing(m)")
cb = plt.colorbar()
cb.set_label('mGal')
plt.savefig('picture/OriginalModel/{}_gz_noise.png'.format(set), dpi=500, bbox_inches='tight')
plt.savefig('picture/OriginalModel/{}_gz_noise.eps'.format(set), dpi=500, bbox_inches='tight')
#plt.show()

