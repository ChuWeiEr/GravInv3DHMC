# 01-单一立方体模型
# 模型空间3000m*2000m*1000m；网格个数x*y*z=30*20*10；
# 异常体是300m*400m*300m，网格个数8*4*3；密度差1g/cm3；
# 水平x:第11-18
# 水平y:第8-11
# 垂向z:第3、4、5层
# 输出文件：正演重力异常数据
# 注：棱柱正演的单位: m；g/cm3

import sys
sys.path.append("..")

import numpy as np
import mesher, utils
from gravmag import prism
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from vis import mpl, myv

# 名字
# 定义研究区名称 & 测试名称
set = 'model04_complex'
test = 'T3'
# 取值范围
rhomin, rhomax = 0, 1
# 网格大小
dx, dy, dz = 100, 100, 100
# 读入模型信息
mshape = np.loadtxt('modeldata/' + set + test + '/mshape.txt')
nz, ny, nx = int(mshape[0]), int(mshape[1]), int(mshape[2])
shape = [nx, ny]
# 研究区范围
xmin, xmax, ymin, ymax, zmin, zmax = 0, nx*dx, 0, ny*dy, 0, nz*dz
# 读入misfit
with open('result/' + set + test + '/misfit.txt') as f:
    data_misfit, model_misfit, regul_factor = np.loadtxt(f, usecols=[0, 1, 2], unpack=True)
print("final regul_factor:", regul_factor[-1])
# 读入模型数据
rhoinv = np.loadtxt('result/' + set + test + '/inversion_model.txt')
# 读入异常
with open('modeldata/{}_gz_noise.txt'.format(set)) as f:
    xobs, yobs, heights, dobs = np.loadtxt(f, usecols=[0, 1, 2, 3], unpack=True)
dpre = np.loadtxt('result/' + set + test + '/inversion_data.txt')
derror = dpre - dobs

# 读入frame
frame1 = np.loadtxt("modeldata/{}_frame1_handle.dat".format(set))
frame2 = np.loadtxt("modeldata/{}_frame2_handle.dat".format(set))
frame3 = np.loadtxt("modeldata/{}_frame3_handle.dat".format(set))
frame4 = np.loadtxt("modeldata/{}_frame4_handle.dat".format(set))

# 误差曲线
plt.figure()
plt.subplot(221)
plt.title("data_misfit")
plt.plot(data_misfit)
plt.subplot(222)
plt.title("model_misfit")
plt.plot(model_misfit)
plt.subplot(223)
plt.title("regul_factor")
plt.plot(regul_factor)

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
plt.savefig('picture/{}_{}_misfit.png'.format(set, test), dpi=500, bbox_inches='tight')
#plt.show()

# 正演模型切片图
rho3D = rhoinv.reshape(nz, ny, nx)
plt.figure(figsize=(10, 2))
for i in range(nz):
    plt.suptitle('origin model')
    ax = plt.subplot(2, 5, i + 1)
    plt.pcolor(rho3D[i, :, :].T, vmin=rhomin, vmax=rhomax, cmap=plt.cm.jet)
    if i in list([2, 3, 4, 5, 6]):
        plt.plot(frame1[:, 0] + 0.5, frame1[:, 1] + 0.5, 'white')
    if i in list([3, 4, 5]):
        plt.plot(frame2[:, 0] + 0.5, frame2[:, 1] + 0.5, 'white')
    if i in list([2, 3, 4]):
        plt.plot(frame3[:, 0] + 0.5, frame3[:, 1] + 0.5, 'white')
    if i in list([2, 3, 4, 5, 6]):
        plt.plot(frame4[:, 0] + 0.5, frame4[:, 1] + 0.5, 'white')
    cb = plt.colorbar()
    cb.set_label('$g/cm^3$')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim([0, ny])
    plt.ylim([0, nx])

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
plt.savefig('picture/{}_{}_plane.png'.format(set, test), dpi=500, bbox_inches='tight')
#plt.show()


# 异常结果画图
plt.figure(figsize=(10,5))
plt.title('gz predicted(mGal)')
plt.axis('scaled')
mpl.contourf(yobs, xobs, dpre, shape, 15)
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.colorbar()
plt.savefig('picture/{}_{}_gzpre.png'.format(set, test), dpi=500, bbox_inches='tight')
#plt.show()

# 结果画图
plt.figure(figsize=(10, 5))
plt.title('gz error(mGal)')
plt.axis('scaled')
mpl.contourf(yobs, xobs, derror, shape, 15)
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.colorbar()
plt.savefig('picture/{}_{}_gzerror.png'.format(set, test), dpi=500, bbox_inches='tight')
#plt.show()

# # 模型3D作图
# mesh = mesher.PrismMesh(bounds=(xmin, xmax, ymin, ymax, zmin, zmax), spacing=(dz, dy, dx))
# mesh.addprop('density', rhoinv)
# myv.figure()
# plot = myv.prisms(mesh, prop='density')
# axes = myv.axes(myv.outline(), fmt='%.0f')
# myv.savefig('picture/{}_{}_3Dmodel.png'.format(set, test))
# myv.show()