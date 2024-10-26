# z方向等比率变大模型

import sys
sys.path.append("../..")
import os
import numpy as np
import mesher, utils
from gravmag import prism
import matplotlib.pyplot as plt
from vis import mpl

# 名字
set = 'model_ratio'
# 创建文件夹
if not os.path.exists('modeldata'):
    os.mkdir('modeldata')
if not os.path.exists('picture'):
    os.mkdir('picture')

# 取值范围
rho_density = 0.4  # 密度大小
dx, dy, dz = 200, 200, 200
# 研究区范围
xmin, xmax, ymin, ymax, zmin, zmax = 0, 6000, 0, 6000, 0, 6000
mesh = mesher.PrismMesh(bounds=(xmin, xmax, ymin, ymax, zmin, zmax),
                            spacing=(dz, dy, dx), ratio=1.05)
# 小棱柱
nz, ny, nx = mesh.shape
print(mesh.shape)
# rho
rho = np.zeros([nx * ny * nz])
# 密度按照棱柱排，先x再y再z
for iz in range(nz):
    for iy in range(ny):
        for ix in range(nx):
            # ix是竖着的轴
            # ---左边竖
            if 10 <= iy <= 11 and 5 <= ix <= 25 and 2 <= iz <= 15:
                index = nx * ny * iz + nx * iy + ix
                rho[index] = rho_density
            # 横1
            if 12 <= iy <= 21 and 23 <= ix <= 25 and 3 <= iz <= 16:
                index = nx*ny*iz + nx*iy + ix
                rho[index] = rho_density
            # 横2
            if 12 <= iy <= 21 and 14 <= ix <= 16 and 5 <= iz <= 9:
                index = nx*ny*iz + nx*iy + ix
                rho[index] = rho_density
            # 横3
            if 12 <= iy <= 21 and 5 <= ix <= 7 and 3 <= iz <= 16:
                index = nx*ny*iz + nx*iy + ix
                rho[index] = rho_density


mesh.addprop('density', rho)
# 保存模型数据
# 计算模型点的坐标 -----排列顺序：先lon变，再lat变，最后r变
plons, plats, prs = mesh.get_xs(), mesh.get_ys(), mesh.get_zs()
grid_rs, grid_lats, grid_lons = np.meshgrid(prs[:-1],
                                            plats[:-1] + dy / 2, plons[:-1] + dx / 2,
                                            indexing='ij')
with open("modeldata/{}_rho.dat".format(set), "w") as f:
    np.savetxt(f, np.c_[grid_lons.ravel(), grid_lats.ravel(),
                        grid_rs.ravel(), rho], fmt="%.8f", delimiter=' ')

# 模型3D作图
#myv.figure()
#plot = myv.prisms(mesh, prop='density')
#axes = myv.axes(myv.outline(), fmt='%.0f')
#myv.savefig('picture/{}_3Dmodel.png'.format(set))
#myv.show()

# 模型切片图
# ---------rho
rho3D = rho.reshape(nz, ny, nx)
plt.figure(figsize=(12, 8))
for i in range(nz):
    plt.suptitle('model of density')
    ax = plt.subplot(4, 5, i + 1)
    plt.pcolor(rho3D[i, :, :].T, cmap=plt.cm.jet)
    plt.colorbar()
    plt.xticks(np.arange(0, ny + 1, 10), np.arange(0, (ny + 1) * dy, 10 * dy))
    plt.yticks(np.arange(0, nx + 1, 10), np.arange(0, (nx + 1) * dx, 10 * dx))
    plt.xlim([0, ny])
    plt.ylim([0, nx])
    plt.xlabel("Easting(m)")
    plt.ylabel("Northing(m)")

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
plt.savefig('picture/OriginalModel/{}-originmodel-rho.png'.format(set), dpi=500, bbox_inches='tight')
plt.show()


# 正演
shape = (nx, ny)
xp, yp, zp = utils.regular((xmin, xmax, ymin, ymax), shape, z=0)
gz_pre, _ = prism.gz(xp, yp, zp, mesh)
# 给正演数据加上高斯噪声,当做观测数据
noise = np.random.normal(loc=0, scale=0.02*gz_pre.max(), size=gz_pre.shape[0])
gz_noise = gz_pre + noise
# 保存数据
np.savetxt('modeldata/{}_gz_noise.txt'.format(set), np.c_[xp, yp, zp, gz_noise], fmt="%.2f", delimiter=" ")

# 模型3D作图
# myv.figure()
# plot = myv.prisms(mesh, prop='density')
# axes = myv.axes(myv.outline(), fmt='%.0f')
# myv.savefig('picture/OriginalModel/{}_3Dmodel.png'.format(set))
#myv.show()


# 正演结果画图
plt.figure(figsize=(10,5))
plt.title('gz with noise(mGal)')
plt.axis('scaled')
mpl.contourf(yp, xp, gz_noise, shape, 15)
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.colorbar()
plt.savefig('picture/OriginalModel/{}_gz_noise.png'.format(set), dpi=500, bbox_inches='tight')
#plt.show()

# 正演结果画图
plt.figure(figsize=(10,5))
plt.title('gz(mGal)')
plt.axis('scaled')
mpl.contourf(yp, xp, gz_pre, shape, 15)
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.colorbar()
plt.savefig('picture/OriginalModel/{}_gz_pre.png'.format(set), dpi=500, bbox_inches='tight')
#plt.show()


