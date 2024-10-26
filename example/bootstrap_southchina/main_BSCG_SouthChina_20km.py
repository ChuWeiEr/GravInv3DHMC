'''
Example of gravity inversion using bootstrap method, model with topography

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
from inversion import reginv

# 计算时间
start_time = timeit.default_timer()
def main():

    # 定义研究区名称 & 测试名称
    set = 'SouthChina20km'
    test = 'T1'
    # 创建文件夹
    if not os.path.exists('modeldata'):
        os.mkdir('modeldata')
    if not os.path.exists('modeldata/' + set + test):
        os.mkdir('modeldata/' + set + test)
    if not os.path.exists('result'):
        os.mkdir('result')
    if not os.path.exists('result/' + set + test):
        os.mkdir('result/' + set + test)

    # 密度范围
    rhomin, rhomax = -1, 1
    boundary = [rhomin, rhomax]
    # 研究区大小1800*1800*5200m
    # 网格大小
    # dx, dy, dz = 5000, 5000, 1000
    dx, dy, dz = 100000, 100000, 2000
    # 研究区范围
    xmin, xmax, ymin, ymax, zmin, zmax = -750000, 950000, -820000, 880000, -3500, 50000
    # 创建模型
    mrange = (xmin, xmax, ymin, ymax, zmin, zmax)
    mspacing = (dz, dy, dx)
    # 读入观测数据
    with open('obsdata/scscs_proj_5000m_win_s7.txt') as f:
        xobs, yobs, heights, dobs = np.loadtxt(f, usecols=[0, 1, 2, 3], unpack=True)
    obsurface = (xobs, yobs, -1 * heights)
    # # ---读入地形数据，可以不规则，后面会做插值
    # with open('modeldata/gra_fa_baiyun_east_use_small.txt') as f:
    #     xtopo, ytopo, topography = np.loadtxt(f, usecols=[0, 1, 2], unpack=True)
    # ---模型
    # 实例化类
    inv3D = reginv.BootStrap(mrange, mspacing, obsurface, dobs, boundary,
                             samples=100, beta=0.01, maxk=10, mratio=1.5, njobs=4,
                             wavelet='1D', mtopo=(xobs, yobs, heights))
    # 查看模型剖分网格个数
    nz, ny, nx = inv3D.mshape
    print("model shape(nz, ny, nx):", nz, ny, nx)
    #  -----保存模型剖分信息
    # 模型剖分网格个数
    with open('modeldata/' + set + test + '/mshape.txt', 'w') as f:
        np.savetxt(f, inv3D.mshape, fmt='%d', delimiter=' ')
    # 模型深度方向剖分
    with open('modeldata/' + set + test + '/mzs.txt', 'w') as f:
        np.savetxt(f, inv3D.mzs, fmt='%.8f', delimiter=' ')
    # 保存carve掉的rho的index
    with open('modeldata/' + set + test + '/maskindex.txt', 'w') as f:
        np.savetxt(f, inv3D.mask, fmt='%d', delimiter=' ')
    # --保存规则排列的初始模型
    # 排列顺序：先x变，再y变，最后z变
    initial_model_mesh = np.zeros([nx * ny * nz])
    with open('modeldata/' + set + test + '/initial_model_mesh.txt', 'w') as f:
        np.savetxt(f, initial_model_mesh, fmt='%.5f', delimiter=' ')
    # 需要将carve的位置的密度值也carve掉，新的rho_carve(rho)作为初始模型去做反演，反演结束的结果是rho_carve。
    # 注：为了方便画图，在另一个画图脚本中将rho_carve变回规则的rho_mesh
    initial_model = utils.rho2carve(initial_model_mesh, inv3D.mask)

    # 使用CG解反演方程
    model_inv_all, data_misfit_all, model_misfit_all, regul_factor_all = inv3D.BSCG(initial_model)
    # 保存反演结果
    with open('result/' + set + test + '/inversion_model.txt', 'w') as f:
        np.savetxt(f, model_inv_all, fmt="%.5f\t", delimiter=' ')
    # 保存误差与正则化因子
    with open('result/' + set + test + '/data_misfit.txt', 'w') as f:
        np.savetxt(f, data_misfit_all, fmt="%.5f\t", delimiter=' ')
    with open('result/' + set + test + '/model_misfit.txt', 'w') as f:
        np.savetxt(f, model_misfit_all, fmt="%.5f\t", delimiter=' ')
    with open('result/' + set + test + '/regul_factor.txt', 'w') as f:
        np.savetxt(f, regul_factor_all, fmt="%.5f\t", delimiter=' ')

if __name__ == "__main__":

    main()

    # 所需时间
    end_time = timeit.default_timer()
    print("total time is: ", end_time - start_time)
