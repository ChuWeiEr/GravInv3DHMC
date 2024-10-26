# 07-global模型
# 模型空间180°*360°*100km；网格间距3°*3°*10km;网格个数60*120*10；
# 异常体是，网格个数；密度差1g/cm3；
# 输出文件：正演重力异常数据
# 注：球棱柱正演的单位-经纬度；m；g/cm3

import sys
sys.path.append("../..")
import os
import numpy as np
import mesher, utils
from gravmag import tesseroidforward
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.pyplot import MultipleLocator
plt.switch_backend('agg')
from vis import mpl

def main():
    # 名字
    set = 'model_global'
    # 创建文件夹
    if not os.path.exists('modeldata'):
        os.mkdir('modeldata') 
    if not os.path.exists('picture'):
        os.mkdir('picture')
    if not os.path.exists('picture/OriginalModel'):
        os.mkdir('picture/OriginalModel')

    # 读入边框文件--平面
    frame_plane1 =  np.loadtxt("modeldata/{}_frame_plane1_handle.dat".format(set))
    frame_plane2 =  np.loadtxt("modeldata/{}_frame_plane2_handle.dat".format(set))
    frame_plane3 =  np.loadtxt("modeldata/{}_frame_plane3_handle.dat".format(set))
    frame_plane4 =  np.loadtxt("modeldata/{}_frame_plane4_handle.dat".format(set))
    frame_plane5 =  np.loadtxt("modeldata/{}_frame_plane5_handle.dat".format(set))
    # 剖面
    frame_profile1 =  np.loadtxt("modeldata/{}_frame_profile1_handle.dat".format(set))
    frame_profile2 =  np.loadtxt("modeldata/{}_frame_profile2_handle.dat".format(set))
    frame_profile3 =  np.loadtxt("modeldata/{}_frame_profile3_handle.dat".format(set))
    frame_profile4 =  np.loadtxt("modeldata/{}_frame_profile4_handle.dat".format(set))
    # 使用多个核
    ncpu = 4
    # 剖面图位置
    profile_h = 32
    profile_v = 65
    # 取值范围
    rhomin, rhomax = 0, 0.8
    # 网格个数
    nlon, nlat, nr = 120, 60, 10
    dlon, dlat, dr = 3, 3, -300000
    # 研究区范围
    west, east, south, north, top, bottom = -180, 180, -90, 90, 0, nr*dr
    # 创建模型
    mesh = mesher.TesseroidMesh((west, east, south, north, top, bottom), (dr, dlat, dlon))

    # 排列顺序：先lon变，再lat变，最后r变;iy:y轴；ix:x轴。
    rho = np.zeros([nr*nlat*nlon])
    for iz in range(nr):
        for iy in range(nlat):
            for ix in range(nlon):
                # 异常体1：大正方体15个*15个
                if 25 <= iy <= 40 and 25 <= ix <= 40 and 2 <= iz <= 6:
                    index = nlon*nlat*iz + nlon*iy + ix
                    rho[index] = 0.8
                # 异常体2：大正方体
                if 10 <= iy <= 20 and 60 <= ix <= 70 and 2 <= iz <= 6:
                    index = nlon * nlat * iz + nlon * iy + ix
                    rho[index] = 0.4
                # 异常体3：大长方体
                if 45 <= iy <= 50 and 60 <= ix <= 90 and 2 <= iz <= 5:
                    index = nlon * nlat * iz + nlon * iy + ix
                    rho[index] = 0.6
                # 异常体4：小正方体
                if 30 <= iy <= 35 and 70 <= ix <= 80 and 2 <= iz <= 4:
                    index = nlon * nlat * iz + nlon * iy + ix
                    rho[index] = 0.5
                # 异常体5：小正方体
                if 25 <= iy <= 30 and 90 <= ix <= 100 and 2 <= iz <= 4:
                    index = nlon * nlat * iz + nlon * iy + ix
                    rho[index] = 0.5
                    
    # rho
    mesh.addprop('density', rho)
    # 保存模型数据
    np.savetxt("modeldata/{}_rho.dat".format(set), rho, fmt='%.5f', delimiter=' ')

    # -------切片图
    rho3D = rho.reshape(nr, nlat, nlon)
    plt.figure(figsize=(10, 2))
    for i in range(nr):
        plt.suptitle('origin model')
        ax = plt.subplot(2, 5, i + 1)
        plt.pcolor(rho3D[i, :, :], vmin=rhomin, vmax=rhomax, cmap=plt.cm.jet)
        if i in list([2, 3, 4, 5, 6]):
            plt.plot(frame_plane1[:, 0], frame_plane1[:, 1], 'white')
            plt.plot(frame_plane2[:, 0], frame_plane2[:, 1], 'white')
        if i in list([2, 3, 4, 5]):  
            plt.plot(frame_plane3[:, 0], frame_plane3[:, 1], 'white')
        if i in list([2, 3, 4]): 
            plt.plot(frame_plane4[:, 0], frame_plane4[:, 1], 'white')
            plt.plot(frame_plane5[:, 0], frame_plane5[:, 1], 'white')
        plt.colorbar()
        plt.xticks(np.linspace(0, nlon, 5))
        plt.yticks(np.linspace(0, nlat, 5))
        plt.xlim([0, nlon])
        plt.ylim([0, nlat])
        plt.xlabel("Easting")
        plt.ylabel("Northing")

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.7, hspace=0.5)
    plt.savefig('picture/OriginalModel/{}-rho-plane.png'.format(set), dpi=500, bbox_inches='tight')
    # plt.show()

    # -------剖面图profile_h
    plt.figure(figsize=(10, 5))
    ax = plt.subplot()
    plt.title('density profile: lat(layer) = {}'.format(profile_h))
    plt.pcolor(rho3D[:, profile_h, :], vmin=rhomin, vmax=rhomax, cmap=plt.cm.jet)
    cb = plt.colorbar()
    cb.set_label('$kg/m^3$')
    plt.plot(frame_profile1[:, 0], frame_profile1[:, 1], 'white')
    plt.plot(frame_profile2[:, 0], frame_profile2[:, 1], 'white')
    plt.xticks(np.linspace(0, nlon, 5))
    plt.yticks(np.linspace(0, nr, 5))
    plt.xlim([0, nlon])
    plt.ylim([0, nr])
    plt.xlabel("Easting")
    plt.ylabel("Radius")
    ax.invert_yaxis()

    plt.savefig('picture/OriginalModel/{}-profile_h.png'.format(set), dpi=500, bbox_inches='tight')
    #plt.show()
    plt.close()

    # -------剖面图profile_v
    plt.figure(figsize=(10, 5))
    ax = plt.subplot()
    plt.title('density profile: lon(layer) = {}'.format(profile_v))
    plt.pcolor(rho3D[:, :, profile_v], vmin=rhomin, vmax=rhomax, cmap=plt.cm.jet)
    cb = plt.colorbar()
    cb.set_label('$kg/m^3$')
    plt.plot(frame_profile3[:, 0], frame_profile3[:, 1], 'white')
    plt.plot(frame_profile4[:, 0], frame_profile4[:, 1], 'white')
    plt.xticks(np.linspace(0, nlon, 5))
    plt.yticks(np.linspace(0, nr, 5))
    plt.xlim([0, nlat])
    plt.ylim([0, nr])
    plt.xlabel("Northing")
    plt.ylabel("Radius")
    ax.invert_yaxis()

    plt.savefig('picture/OriginalModel/{}-profile_v.png'.format(set), dpi=500, bbox_inches='tight')
    #plt.show()
    plt.close()

    
    # ---------正演
    # Create the computation grid
    area = (west, east, south, north)
    shape = (nlon+1, nlat+1)
    # 后面的先变,观测面高度5km
    lons, lats, heights = utils.regular(area, shape, z=5000)

    # 正演计算
    gz_pre = tesseroidforward.gz(lons, lats, heights, mesh, njobs=ncpu)
    # -------gz------给正演数据加上高斯噪声,当做观测数据
    noise_gz = np.random.normal(loc=0, scale=0.02 * gz_pre.max(), size=gz_pre.shape[0])
    gz_noise = gz_pre + noise_gz
    # 保存数据
    np.savetxt('modeldata/{}_gz_noise.txt'.format(set),
               np.c_[lons, lats, heights, gz_noise], fmt="%.5f", delimiter=" ")

    # --------异常图
    plt.figure()
    bm = mpl.basemap(area, 'robin')
    mpl.contourf(lons, lats, gz_noise, shape, 15, basemap=bm)
    bm.drawmeridians(np.linspace(west, east, 5), labels=[0, 0, 0, 1], linewidth=0.2)  # 经线
    bm.drawparallels(np.linspace(south, north, 5), labels=[1, 0, 0, 0], linewidth=0.2)  # 纬线
    #bm.drawcoastlines()
    cb = plt.colorbar()
    cb.set_label('mGal')
    plt.savefig('picture/OriginalModel/{}-gz.png'.format(set), dpi=500, bbox_inches='tight')
    # plt.show()
    
    
 
if __name__ == '__main__':
    main()
