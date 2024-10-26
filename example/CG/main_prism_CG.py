'''
Example of gravity inversion using bootstrap method, model without topography
'''

import sys
sys.path.append("..")

import os
import timeit
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import mesher, utils
from vis import mpl
from gravmag import prism
from inversion import reginv

# 计算时间
start_time = timeit.default_timer()
def main():

    # 定义研究区名称 & 测试名称
    set = 'model03_twodykes'
    test = 'T8'
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
    rhomin, rhomax = 0, 1
    # 网格大小 & 个数
    dx, dy, dz = 100, 100, 100
    nx, ny, nz = 30, 40, 10
    # 研究区范围
    xmin, xmax, ymin, ymax, zmin, zmax = 0, nx*dx, 0, ny*dy, 0, nz*dz
    # 创建模型
    mrange = (xmin, xmax, ymin, ymax, zmin, zmax)
    mspacing = (dz, dy, dx)
    # 读入观测数据
    with open('modeldata/{}_gz_noise.txt'.format(set)) as f:
        xobs, yobs, heights, dobs = np.loadtxt(f, usecols=[0, 1, 2, 3], unpack=True)
    obsurface = (xobs, yobs, heights)
    # # ---读入地形数据，可以不规则，后面会做插值
    # with open('modeldata/gra_fa_baiyun_east_use_small.txt') as f:
    #     xtopo, ytopo, topography = np.loadtxt(f, usecols=[0, 1, 2], unpack=True)
    # ---模型
    # 实例化类
    inv3D = reginv.ConjugateGradient(dobs, mrange, mspacing, obsurface)
    # 查看模型剖分网格个数
    nz, ny, nx = inv3D.mshape
    print("model shape(nz, ny, nx):", nz, ny, nx)
    #  -----保存模型剖分信息
    # 模型剖分网格个数
    with open('modeldata/' + set + test + '/mshape.txt', 'w') as f:
        np.savetxt(f, inv3D.mshape, fmt='%d', delimiter=' ')
    # 初始模型
    initial_model = np.zeros([nx * ny * nz])
    apriorModel = np.zeros([nx * ny * nz])
    # 使用CG解反演方程
    model_inv, data_inv, data_misfit, model_misfit, regul_factor = inv3D.CG(initial_model, 
    apriorModel, boundary=[rhomin, rhomax], regularization='MS', beta=0.001, q=0.7, maxk=200)
    # 保存反演结果
    with open('result/' + set + test + '/inversion_model.txt', 'w') as f:
        np.savetxt(f, model_inv, fmt="%.5f", delimiter=' ')
    # 保存正则化因子
    with open('result/' + set + test + '/inversion_data.txt', 'w') as f:
        np.savetxt(f, data_inv, fmt="%.5f", delimiter=' ')
    
    with open('result/' + set + test + '/misfit.txt', 'w') as f:
        np.savetxt(f, np.c_[data_misfit, model_misfit, regul_factor], fmt="%.5f", delimiter=' ')


if __name__ == "__main__":

    main()

    # 所需时间
    end_time = timeit.default_timer()
    print("total time is: ", end_time - start_time)
