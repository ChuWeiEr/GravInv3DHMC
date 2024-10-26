import sys
sys.path.append("../..")
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.pyplot import MultipleLocator
plt.switch_backend('agg')
from gravmag import prism
import mesher, utils
from vis import mpl, myv
import time

# 画出反演结果，包括数据、模型
# 读入参数文件
parameters = []
with open("SetPMTS.txt", "r") as f:
    for line in f:
        line = eval(line)
        parameters.append(line)

attempt = int(sys.argv[1])
chains = int(sys.argv[2])
set = parameters[attempt]["set"]
test = parameters[attempt]["test"]
rhomin, rhomax = parameters[attempt]["rhomin"], parameters[attempt]["rhomax"]
nsamples = parameters[attempt]["nsamples"]
print(set + '\t' + test)
print("Begain to calculate at ", time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()))

# 读入观测数据
with open('modeldata/{}_gz_noise.txt'.format(set), encoding='utf-8') as f:
    xobs, yobs, heights, dobs = np.loadtxt(f, usecols=[0, 1, 2, 3], unpack=True)
# 读入真实模型
rho_true = np.loadtxt("modeldata/{}_rho.dat".format(set))
# 小棱柱个数
with open("modeldata/modelinfo_{}.txt".format(set)) as f:
    nz, ny, nx = np.loadtxt(f, usecols=[0, 1, 2])
nz, ny, nx = int(nz), int(ny), int(nx)
# 读入frame
frame = np.loadtxt("modeldata/{}_frame_plane_handle.dat".format(set))
frame_pf = np.loadtxt("modeldata/{}_frame_profile_handle.dat".format(set))
# 统计最后last个模型用于画3D图，剖面图和平面图
last = 100
ndraws = nsamples - last
# chain 0, chains
for chain in range(0, chains):
    misfit = np.loadtxt("result/" + str(set) + str(test) + "_chain" + str(chain) + "/misfit.dat")
    # 所有模型
    #model_all = np.loadtxt("result/" + str(set) + str(test) + "_chain" + str(chain) + "/model.dat")
    with open("result/" + str(set) + str(test) + "_chain" + str(chain) + "/model.dat") as f:
        for i in range(ndraws):
            f.readline()
        model_all = np.loadtxt(f)
    
    # 误差曲线
    # 将误差开根号并归一化
    misfit_total = misfit[:, 0]
    misfit_data = misfit[:, 1]
    misfit_model = misfit[:, 2]
    misfit_total_norm = misfit[:, 3]
    misfit_data_norm = misfit[:, 4]
    misfit_model_norm = misfit[:, 5]
    
    # -------------误差曲线
    plt.figure()
    #plt.title("misfit")
    plt.plot(np.arange(0, nsamples), misfit_total_norm)
    plt.xlim([0, nsamples])
    plt.xlabel("iterations")
    plt.ylabel("Misfit")
    ## [left, bottom, width, height]四个参数(fractions of figure)可以非常灵活的调节子图中子图的位置
    plt.axes([0.3, 0.4, 0.5, 0.3])
    plt.title(r'Last %d iterations' % (last))
    plt.plot(np.arange(ndraws, nsamples), misfit_total_norm[ndraws:])
    plt.xlim([ndraws, nsamples])
    # plt.ylim([9, 9.5])
    ax = plt.gca()
    ax.ticklabel_format(style='plain', axis='y')  # 不使用科学记数法
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.7f')) # 坐标不再加上一个数

    plt.savefig('picture/' + str(set) + str(test) + '_chain{}-misfit_total.png'.format(chain), dpi=500,
                bbox_inches='tight')
    # plt.show()
    plt.close()

    # 统计最后last个模型用于画3D图，剖面图和平面图
    last = 100
    ndraws = nsamples - last
    # 误差曲线从nstart开始画
    nstart = 0
    # 剖面
    profile = 9
    # 网格大小
    dx, dy, dz = 100, 100, 100
    # 网格shape
    shape = (nx, ny)
    # 研究区范围
    xmin, xmax, ymin, ymax, zmin, zmax = 0, 2000, 0, 3000, 0, 1000

    # ***********************nbest
    # 计算每一个网格点密度值的均值和方差
    model_inv_mean = np.mean(model_all, axis=0)
    model_inv_std = np.std(model_all, axis=0)
    # 变为3D数组
    rho3D_std = model_inv_std.reshape(nz, ny, nx)
    rho3D_mean = model_inv_mean.reshape(nz, ny, nx)

    # mesh模块
    # ---------反演模型的均值
    mesh_inv_mean = mesher.PrismMesh(bounds=(xmin, xmax, ymin, ymax, zmin, zmax), spacing=(dz, dy, dx))
    mesh_inv_mean.addprop('density', model_inv_mean)
    # ---------反演模型的方差
    mesh_inv_std = mesher.PrismMesh(bounds=(xmin, xmax, ymin, ymax, zmin, zmax), spacing=(dz, dy, dx))
    mesh_inv_std.addprop('density', model_inv_std)

    # 根据均值和方差做正演计算
    dpre_mean, _ = prism.gz(xobs, yobs, heights, mesh_inv_mean)
    dpre_std, _ = prism.gz(xobs, yobs, heights, mesh_inv_std)

    # -----保存数据---
    # -----反演模型
    # 计算模型点的坐标
    xs, ys, zs = mesh_inv_mean.get_xs(), mesh_inv_mean.get_ys(), mesh_inv_mean.get_zs()
    grid_zs, grid_ys, grid_xs = np.meshgrid(zs[:-1], ys[:-1], xs[:-1], indexing='ij')
    header = "# x, y,z, model_inv_mean, model_inv_std"
    with open("result/" + str(set) + str(test) + "_chain{}/inversion_model.dat".format(chain), "w") as f:
        np.savetxt(f, np.c_[grid_xs.ravel(), grid_ys.ravel(), grid_zs.ravel(), model_inv_mean, model_inv_std], 
        fmt="%.8f", delimiter=' ')
    # -----异常
    d_error = dobs - dpre_mean
    header = "#xobs, yobs, heights, dpre_mean, dpre_std, d_error "
    with open("result/" + str(set) + str(test) + "_chain{}/inversion_anomaly.dat".format(chain), "w") as f:
        np.savetxt(f, np.c_[xobs, yobs, heights, dpre_mean, dpre_std, d_error], fmt="%.8f", delimiter=' ')

    
    # # 模型三维图均值
    # myv.figure()
    # plot = myv.prisms(mesh_inv_mean, prop='density')
    # axes = myv.axes(myv.outline(), fmt='%.0f')
    # myv.savefig('picture/' + str(set) + str(test) + '-chain{}-rhomean3D.png'.format(chain))
    # # myv.show()
    # # 模型三维图方差
    # myv.figure()
    # plot = myv.prisms(mesh_inv_std, prop='density')
    # axes = myv.axes(myv.outline(), fmt='%.0f')
    # myv.savefig('picture/' + str(set) + str(test) + '-chain{}-rhostd3D.png'.format(chain))
    # # myv.show()
    
    
    # 数据拟合均方根RMSD& 模型恢复均方根RMSM：
    RMSD = np.sqrt(np.linalg.norm(dobs - dpre_mean) ** 2 / len(dobs))
    RMSM = np.sqrt(np.linalg.norm(rho_true - model_inv_mean) ** 2 / model_all.shape[1])
    print("RMSD:", RMSD)
    print("RMSM:", RMSM)
    
    # ----------异常-mean
    plt.figure(figsize=(10, 3))
    plt.subplot(121)
    mpl.contourf(yobs, xobs, dpre_mean, shape, 15)
    plt.xlabel("Easting(m)")
    plt.ylabel("Northing(m)")
    cb = plt.colorbar()
    cb.set_label('mGal')

    plt.subplot(122)
    mpl.contourf(yobs, xobs, d_error, shape, 15)
    plt.xlabel("Easting(m)")
    plt.ylabel("Northing(m)")
    cb = plt.colorbar()
    cb.set_label('mGal')

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
    plt.savefig('picture/' + str(set) + str(test) + '-chain{}-Anomaly.png'.format(chain), dpi=500,
                bbox_inches='tight')
    # plt.show()


    # 模型切片图-mean
    plt.figure(figsize=(15, 3))
    for i in range(nz):
        # plt.suptitle('inversion density model (mean)')
        ax = plt.subplot(2, 5, i + 1)
        plt.pcolor(rho3D_mean[i, :, :].T, vmin=rhomin, vmax=rhomax, cmap=plt.cm.jet)
        plt.colorbar()
        if i in list([2, 3, 4]):
            plt.plot(frame[:, 0] + 1,frame[:, 1] + 1, 'white')
    
        plt.xticks(np.arange(0, ny + 1, 20), np.arange(0, (ny + 1) * dy, 20 * dy))
        plt.yticks(np.arange(0, nx + 1, 10), np.arange(0, (nx + 1) * dx, 10 * dx))
        plt.xlim([0, ny])
        plt.ylim([0, nx])
        plt.xlabel("Easting(m)")
        plt.ylabel("Northing(m)")

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=1, hspace=0.7)
    plt.savefig('picture/' + str(set) + str(test) + '-chain{}-rhomeanPlane.png'.format(chain), dpi=500,
                bbox_inches='tight')
    # plt.show()

    # 模型切片图-std
    plt.figure(figsize=(15, 3))
    for i in range(nz):
        # plt.suptitle('inversion density model (std)')
        ax = plt.subplot(2, 5, i + 1)
        plt.pcolor(rho3D_std[i, :, :].T, vmin=rho3D_mean.min(), vmax=rho3D_mean.max(), cmap=plt.cm.jet)
        plt.colorbar()
        if i in list([2, 3, 4]):
            plt.plot(frame[:, 0] + 1,frame[:, 1] + 1, 'white')
        plt.xticks(np.arange(0, ny + 1, 20), np.arange(0, (ny + 1) * dy, 20 * dy))
        plt.yticks(np.arange(0, nx + 1, 10), np.arange(0, (nx + 1) * dx, 10 * dx))
        plt.xlim([0, ny])
        plt.ylim([0, nx])
        plt.xlabel("Easting(m)")
        plt.ylabel("Northing(m)")

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=1, hspace=0.7)
    plt.savefig('picture/' + str(set) + str(test) + '-chain{}-rhostdPlane.png'.format(chain), dpi=500,
                bbox_inches='tight')
    # plt.show()

    # 模型剖面图
    # mean
    plt.figure(figsize=(10, 2))
    ax = plt.subplot()
    plt.pcolor(rho3D_mean[:, :, profile], vmin=rhomin, vmax=rhomax, cmap=plt.cm.jet)
    cb = plt.colorbar()
    cb.set_label('$g/cm^3$')

    frame_pf = np.array(frame_pf)
    plt.plot(frame_pf[:, 0] + 1, frame_pf[:, 1] + 1, 'white')

    plt.xticks(np.arange(0, ny + 1, 10), np.arange(0, (ny + 1) * dy, 10 * dy))
    plt.yticks(np.arange(0, nz + 1, 5), np.arange(0, (nz + 1) * dz, 5 * dz))
    plt.xlim([0, ny])
    plt.ylim([0, nz])
    plt.xlabel("Easting(m)")
    plt.ylabel("Depth(m)")
    ax.invert_yaxis()

    plt.savefig('picture/' + str(set) + str(test) + '-chain{}-rhomeanProfile.png'.format(chain), dpi=500,
                bbox_inches='tight')
    # plt.show()

    # std
    plt.figure(figsize=(10, 2))
    ax = plt.subplot()
    plt.pcolor(rho3D_std[:, :, profile], vmin=rho3D_mean.min(), vmax=rho3D_mean.max(), cmap=plt.cm.jet)
    cb = plt.colorbar()
    cb.set_label('$g/cm^3$')

    frame_pf = np.array(frame_pf)
    plt.plot(frame_pf[:, 0] + 1, frame_pf[:, 1] + 1, 'white')

    plt.xticks(np.arange(0, ny + 1, 10), np.arange(0, (ny + 1) * dy, 10 * dy))
    plt.yticks(np.arange(0, nz + 1, 5), np.arange(0, (nz + 1) * dz, 5 * dz))
    plt.xlim([0, ny])
    plt.ylim([0, nz])
    plt.xlabel("Easting(m)")
    plt.ylabel("Depth(m)")
    ax.invert_yaxis()
    plt.savefig('picture/' + str(set) + str(test) + '-chain{}-rhostdProfile.png'.format(chain), dpi=500,
                bbox_inches='tight')
    # plt.show()

    
    print("Successfully plot chain-%d by analysing last %d samples" % (chain, last))


