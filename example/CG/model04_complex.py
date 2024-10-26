# 04-组合模型-4个异常体
# 模型空间4000m*3000m*1000m；网格个数40*30*10；
# 水平方向40*30（间距100m），深度上10层（间距100m）；
# 1g/cm3；

import sys
sys.path.append("..")
import os
import numpy as np
import mesher, utils
from gravmag import prism
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from vis import mpl, myv

def main():
    # 名字
    set = 'model04_complex'
    # 创建文件夹
    if not os.path.exists('modeldata'):
        os.mkdir('modeldata')
    if not os.path.exists('picture'):
        os.mkdir('picture')
    if not os.path.exists('picture/OriginalModel'):
        os.mkdir('picture/OriginalModel')
    
    # 读入frame
    frame1 = np.loadtxt("modeldata/{}_frame1_handle.dat".format(set))
    frame2 = np.loadtxt("modeldata/{}_frame2_handle.dat".format(set))
    frame3 = np.loadtxt("modeldata/{}_frame3_handle.dat".format(set))
    frame4 = np.loadtxt("modeldata/{}_frame4_handle.dat".format(set))
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
                # --小正方形（4)
                if 24 <= iy <= 27 and 7 <= ix <= 10 and 2 <= iz <= 6:
                    index = nx*ny*iz + nx*iy + ix
                    rho[index] = rhomax
                # --大正方形（3）
                if 27 <= iy <= 31 and 15 <= ix <= 20 and 3 <= iz <= 5:
                    index = nx * ny * iz + nx * iy + ix
                    rho[index] = rhomax
                # --L形（2）
                if 10 <= iy <= 16 and 5 <= ix <= 7 and 2 <= iz <= 4:
                    index = nx*ny*iz + nx*iy + ix
                    rho[index] = rhomax
                if 14 <= iy <= 16 and 7 <= ix <= 15 and 2 <= iz <= 4:
                    index = nx*ny*iz + nx*iy + ix
                    rho[index] = rhomax
                # ---长方形（1）
                if 9 <= iy <= 19 and 21 <= ix <= 24 and 2 <= iz <= 6:
                    index = nx*ny*iz + nx*iy + ix
                    rho[index] = rhomax

    mesh_rho.addprop('density', rho)
    # 保存模型数据
    np.savetxt("modeldata/{}_rho.dat".format(set), rho, fmt='%.5f', delimiter=' ')
        

    # 正演
    shape = (nx, ny)
    xp, yp, zp = utils.regular((xmin, xmax, ymin, ymax), shape, z=0)
    gz_pre, _ = prism.gz(xp, yp, zp, mesh_rho, njobs=4)
    # 给正演数据加上高斯噪声,当做观测数据
    noise = np.random.normal(loc=0, scale=0.02*gz_pre.max(), size=gz_pre.shape[0])
    gz_noise = gz_pre + noise
    # 保存数据
    np.savetxt('modeldata/{}_gz_noise.txt'.format(set), np.c_[xp, yp, zp, gz_noise], fmt="%.5f", delimiter=" ")


    # # 模型3D作图
    # myv.figure()
    # plot = myv.prisms(mesh, prop='density')
    # axes = myv.axes(myv.outline(), fmt='%.0f')
    # myv.wall_bottom(axes.axes.bounds, opacity=0.2)
    # myv.wall_north(axes.axes.bounds)
    # myv.savefig('picture/{}_3Dmodel.png'.format(set))
    # myv.show()

    # 正演模型切片图
    rho3D = rho.reshape(nz, ny, nx)
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
    plt.savefig('picture/OriginalModel/{}_plane.png'.format(set), dpi=500, bbox_inches='tight')
    #plt.show()

  
    # 正演结果画图
    plt.figure(figsize=(10,5))
    plt.title('gz(mGal)')
    plt.axis('scaled')
    mpl.contourf(yp, xp, gz_noise, shape, 15)
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.colorbar()
    plt.savefig('picture/OriginalModel/{}_gzpre_noise.png'.format(set), dpi=500, bbox_inches='tight')
    #plt.show()

if __name__ == "__main__":

    main()

