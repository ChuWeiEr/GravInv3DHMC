# 构建初始模型或者先验模型。
# 需要完全按照反演模型的网格剖分
# 从而得到按照反演模型需要格式排列的初始/先验密度模型
import sys
sys.path.append("../..")
import os
from scipy.interpolate import griddata
import numpy as np
import numpy as np
import mesher, utils
from vis import mpl, myv
import matplotlib.pyplot as plt
plt.switch_backend('agg')


def main():
    # 创建文件夹
    if not os.path.exists('modeldata'):
        os.mkdir('modeldata')
    if not os.path.exists('picture'):
        os.mkdir('picture')
    # 名字
    set = 'SC'
    # 读入需要做插值的初始模型
    with open("data/SC_Rho3D.txt") as f:
        Crust1lons, Crust1lats, Crust1rs, Crust1rhos = np.loadtxt(f, usecols=[0, 1, 2, 3], unpack=True)
    # ---模型构建-----
    # 网格大小
    dlon, dlat, dr = 0.5, 0.5, [-1000, -2000, -5000]
    # 研究区范围
    west, east, south, north, top, bottom = 106.5, 118.5, 16, 28, 2000, -60000
    # 创建模型
    mrange = (west, east, south, north, top, bottom)
    mspacing = (dr, dlat, dlon)
    mdivisionsection = [top, -5000, -15000, bottom]
    mesh = mesher.TesseroidMeshSegment(mrange, mspacing, mdivisionsection)
    # 查看模型信息
    nr, nlat, nlon = mesh.shape
    # print("model shape(nz, ny, nx):", nr, nlat, nlon)
    mlons, mlats, mrs = mesh.get_xs(), mesh.get_ys(), mesh.get_zs()
    #print("model index(mlons, mlats, mrs):", mlons, mlats, mrs)
    # 模型剖分
    with open("modeldata/modelinfo_{}.txt".format(set), 'w') as f:
        np.savetxt(f, np.c_[nr, nlat, nlon], fmt='%d', delimiter=' ')
    # 深度方向网格剖分
    with open("modeldata/model_index_{}.txt".format(set), 'w') as f:
        f.write("----------mlons----------\n")
        np.savetxt(f, mlons, fmt='%.2f', delimiter=' ')
        f.write("----------mlats----------\n")
        np.savetxt(f, mlats, fmt='%.2f', delimiter=' ')
        f.write("----------mrs----------\n")
        np.savetxt(f, mrs, fmt='%.2f', delimiter=' ')
    # define mesh
    grid_rs, grid_lats, grid_lons = np.meshgrid(mrs[:-1], mlats[:-1], mlons[:-1], indexing='ij')
    grid_mesh = (grid_lons.ravel(), grid_lats.ravel(), grid_rs.ravel())

    # 定义需要插值的原始点
    points = np.zeros([Crust1lons.shape[0], 3])
    points[:, 0] = Crust1lons
    points[:, 1] = Crust1lats
    points[:, 2] = Crust1rs  # 与inversion的z轴坐标系保持一致:向下为负；单位为m
    values = Crust1rhos  # 单位g/cm3

    # interpolation:nearest,linear,cubic; fill_value=2.670
    rho_interp = griddata(points, values, grid_mesh, method='linear', fill_value=0.2)

    # ------------将三维插值结果输出
    with open("data/{}_ApriorModel.txt".format(set), 'w') as f:
        np.savetxt(f, np.c_[grid_lons.ravel(), grid_lats.ravel(), grid_rs.ravel(), rho_interp],
                   fmt="%f", delimiter=' ')

    #--------------density3D绘图
    # mesh.addprop('density', rho_interp)
    # myv.figure()
    # plot = myv.tesseroids(mesh, prop='density')
    # #axes = myv.axes(myv.outline(), fmt='%.0f')
    # myv.savefig('picture/3Dmodel_interpCrust.png')
    # myv.show()

if __name__ == "__main__":
    main()




