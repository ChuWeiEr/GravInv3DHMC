import sys
sys.path.append("../..")
import psutil
import time
import numpy as np
from inversion import hmc, potential
from mpi4py import MPI
import utils
from vis import mpl
import matplotlib.pyplot as plt
#import warnings
#warnings.filterwarnings('ignore')

def main(set, test, mspacing, rhomin, rhomax, Lrange, delta, Sigma, 
        RegulFactor, regularization, beta, nsamples):
    # mpi information
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    ncores = comm.Get_size()

    # -----create model------
    # size of a cell
    dlon, dlat, dr = mspacing[0], mspacing[1], mspacing[2]
    # boundary of model
    west, east, south, north, top, bottom = -180, 180, -90, 90, 0, -3000000
    # create model
    mrange = (west, east, south, north, top, bottom)
    mspacing = (dr, dlat, dlon)
    # -----load data------
    # ---load gravity data
    with open('modeldata/model_global_gz_noise.txt') as f:
        lons, lats, heights, dobs = np.loadtxt(f, usecols=[0, 1, 2, 3], unpack=True)
    obsurface = (lons, lats, heights)
    # 实例化类 
    model = potential.GravMagModule(dobs, mrange, mspacing, obsurface, 
                                coordinate="spherical", njobs=5, field="gravity")
    # 查看模型剖分信息
    nr, nlat, nlon = model.mshape
    print("model shape(nr, nlat, nlon):", nr, nlat, nlon)
    mlons, mlats, mrs = model.mxs, model.mys, model.mzs
    print("model index(mlons, mlats, mrs):", mlons, mlats, mrs)
    # 保存
    with open("modeldata/modelinfo_{}.txt".format(set), 'w') as f:
        np.savetxt(f, np.c_[model.mshape], fmt='%d', delimiter=' ')
    # 网格剖分
    with open("modeldata/mlons_index_{}.txt".format(set), 'w') as f:
        np.savetxt(f, mlons, fmt='%.2f', delimiter=' ')
    with open("modeldata/mlats_index_{}.txt".format(set), 'w') as f:
        np.savetxt(f, mlats, fmt='%.2f', delimiter=' ')
    with open("modeldata/mrs_index_{}.txt".format(set), 'w') as f:
        np.savetxt(f, mrs, fmt='%.2f', delimiter=' ')
    # 排列顺序：先lon变，再lat变，最后r变
    initial_model = np.ones([nr * nlat * nlon]) * 0.01
    # 先验模型
    aprior_model = np.ones([nr * nlat * nlon]) * 0.001

    # parameters（不需要修改）
    save_folder = "result/" + str(set) + str(test) + "_chain"
    # 每次都产生不同的随机数
    seed = 100
    # 统计模型，选择misfit最小的前几个模型
    num_best = 100
    # 模型的size
    nt = initial_model.shape[0]
    # 初始模型范围
    boundaries = np.ones((nt, 2))
    boundaries[:, 0] = rhomin
    boundaries[:, 1] = rhomax
    # plot sample tracks
    plotsamples = False # False True
    im1, im2 = 1, 2  # samples to plot
    # 前ndraws个模型处于预热过程，即采样未达到平稳状态,因此把这部分模型舍弃
    ndraws = 0
    # 是否选择自适应正则化(Adaptive; Fixed)，如果自适应设置衰减因子
    adaptiveRegul = "Fixed"
    RegulRate = 0.8
    # 密度约束方法(可选mandatory或者logarithmic) 
    constraint = "mandatory"
    log_factor = 1000

    if ncores == 1:
        hmc.HMCSample(model, nsamples, ndraws, delta, Lrange,
                      initial_model, aprior_model, boundaries, constraint, log_factor, dobs,
                      adaptiveRegul, RegulRate, RegulFactor, regularization, beta,
                      seed, Sigma, nbest=num_best, myrank=rank, save_folder=save_folder,
                      plotsamples=plotsamples, im=[im1, im2])

    else:
        hmc.HMCSample(model, nsamples, ndraws, delta, Lrange,
                      initial_model, aprior_model, boundaries, constraint, log_factor, dobs,
                      adaptiveRegul, RegulRate, RegulFactor, regularization, beta,
                      seed, Sigma, nbest=num_best, myrank=rank, save_folder=save_folder,
                      plotsamples=plotsamples, im=[im1, im2])

if __name__ == "__main__":
    # 查看电脑的总的内存信息
    info = psutil.virtual_memory()
    print('Total memory of this computer:%.4f GB' % (info.total / 1024 / 1024 / 1024))
    print('Current memory usage %.2f %%:' % (info.percent))
    print('Number of cpu:', psutil.cpu_count())
    print("Begain to calculate at ", time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()))
    # 读入文件
    parameters = []
    with open("SetPMTS.txt", "r") as f:
        for line in f:
            line = eval(line)
            parameters.append(line)

    # number of test
    attempt = int(sys.argv[1])
    set = parameters[attempt]["set"]
    test = parameters[attempt]["test"]
    rhomin, rhomax = parameters[attempt]["rhomin"], parameters[attempt]["rhomax"]
    mspacing =  parameters[attempt]["mspacing"]
    # dt & L
    Lrange = parameters[attempt]["Lrange"] # step size
    delta = parameters[attempt]["delta"] # step length
    Sigma = parameters[attempt]["Sigma"] # mass matrix
    # regularization
    RegulFactor = parameters[attempt]["RegulFactor"]
    regularization = parameters[attempt]["regularization"]
    beta = parameters[attempt]["beta"]
    # samples
    nsamples = parameters[attempt]["nsamples"]

    print(set +"\t" + test)
    print("Parameters info: \n"
          "boundaries=[%.2f, %.2f], dt=[%.5f], L=[%d, %d],\n"
          "regularization=%s, beta=%.4f, nsamples=%d"
          % (rhomin, rhomax, delta, Lrange[0], Lrange[1],regularization, beta, nsamples))
    # calculate
    start = time.time()
    main(set, test, mspacing, rhomin, rhomax, Lrange, delta, Sigma,
        RegulFactor, regularization, beta, nsamples)
    end = time.time()
    print("total time:", end - start)