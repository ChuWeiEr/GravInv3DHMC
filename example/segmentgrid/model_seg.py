# 01-单一立方体模型
# 模型空间3000m*2000m*1000m；网格个数x*y*z=30*20*10；
# 异常体是300m*400m*300m，网格个数8*4*3；密度差1g/cm3；
# 水平x:第11-18
# 水平y:第8-11
# 垂向z:第3、4、5层
# 输出文件：正演重力异常数据
# 注：棱柱正演的单位: m；g/cm3

import sys
sys.path.append("../..")

import numpy as np
import mesher, utils
from gravmag import prism
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from vis import mpl

# 名字
test = 'model_seg'
# 取值范围
rhomin, rhomax = 0, 1
# 小棱柱
nx, ny, nz = 20, 30, 10
# 网格大小
dx, dy, dz = 100, 100, [100, 200, 300]
# 研究区范围
xmin, xmax, ymin, ymax, zmin, zmax = 0, 2000, 0, 3000, 0, 2100
divisionsection = [zmin, 300, 900, zmax]  # 分段
mesh = mesher.PrismMeshSegment((xmin, xmax, ymin, ymax, zmin, zmax), 
                                (dz, dy, dx), divisionsection)
rho = np.zeros([nx*ny*nz])
# 密度按照棱柱排，先x再y再z
for iz in range(nz):
    for iy in range(ny):
        for ix in range(nx):
            if 7 <= ix <= 10 and 10 <= iy <= 17 and 2 <= iz <= 4:
                index = nx*ny*iz + nx*iy + ix
                rho[index] = rhomax

mesh.addprop('density', rho)
# 保存模型数据
np.savetxt("modeldata/{}_rho.dat".format(test), rho, fmt='%.5f', delimiter=' ')

# 读入frame
# ---平面图
frame = np.loadtxt("modeldata/{}_frame_plane_handle.dat".format(test))
# ----剖面图
frame_pf = np.loadtxt("modeldata/{}_frame_profile_handle.dat".format(test))

# 正演模型切片图
rho3D = rho.reshape(nz, ny, nx)
plt.figure(figsize=(10, 2))
for i in range(nz):
    plt.suptitle('origin model')
    ax = plt.subplot(2, 5, i + 1)
    plt.pcolor(rho3D[i, :, :].T, vmin=rhomin, vmax=rhomax, cmap=plt.cm.jet)
    if i in list([2, 3, 4]):
        plt.plot(frame[:, 0] + 1, frame[:, 1] + 1, 'white')
    cb = plt.colorbar()
    cb.set_label('$g/cm^3$')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim([0, ny])
    plt.ylim([0, nx])

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
plt.savefig('picture/OriginalModel/{}-originmodel.png'.format(test), dpi=500, bbox_inches='tight')
#plt.show()

# 圈出模型的外围边框--剖面
profile = 10
# 正演模型剖面图
plt.figure(figsize=(10, 3))
plt.title('profile: y(layer) = {}'.format(profile))
ax = plt.subplot()
plt.pcolor(rho3D[:, :, profile], vmin=rhomin, vmax=rhomax, cmap=plt.cm.jet)
plt.colorbar()
frame_pf = np.array(frame_pf)
plt.plot(frame_pf[:, 0] + 1, frame_pf[:, 1] + 1, 'white')
plt.xticks(np.arange(0, ny, 10))
plt.yticks(np.arange(0, nz, 5))
plt.xlabel("Easting(m)")
plt.ylabel("Depth(m)")
ax.invert_yaxis()

plt.savefig('picture/OriginalModel/{}-profile.png'.format(test), dpi=500, bbox_inches='tight')
#plt.show()


# for p in mesh:
#     print(p)

# 正演
shape = (nx, ny)
xp, yp, zp = utils.regular((xmin, xmax, ymin, ymax), shape, z=0)
gz_pre, _ = prism.gz(xp, yp, zp, mesh)
# 给正演数据加上高斯噪声,当做观测数据
noise = np.random.normal(loc=0, scale=0.02*gz_pre.max(), size=gz_pre.shape[0])
gz_noise = gz_pre + noise
# 保存数据
np.savetxt('modeldata/{}_gz_noise.txt'.format(test), np.c_[xp, yp, zp, gz_noise], fmt="%.2f", delimiter=" ")

# 模型3D作图
# myv.figure()
# plot = myv.prisms(mesh, prop='density')
# axes = myv.axes(myv.outline(), fmt='%.0f')
# myv.savefig('picture/OriginalModel/{}_3Dmodel.png'.format(test))
#myv.show()


# 正演结果画图
plt.figure(figsize=(10,5))
plt.title('gz with noise(mGal)')
plt.axis('scaled')
mpl.contourf(yp, xp, gz_noise, shape, 15)
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.colorbar()
plt.savefig('picture/OriginalModel/{}_gz_noise.png'.format(test), dpi=500, bbox_inches='tight')
#plt.show()

# 正演结果画图
plt.figure(figsize=(10,5))
plt.title('gz(mGal)')
plt.axis('scaled')
mpl.contourf(yp, xp, gz_pre, shape, 15)
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.colorbar()
plt.savefig('picture/OriginalModel/{}_gz_pre.png'.format(test), dpi=500, bbox_inches='tight')
#plt.show()
