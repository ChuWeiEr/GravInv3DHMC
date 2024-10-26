import sys
sys.path.append("../..")
import os
import numpy as np
from gravmag import tesseroidforward
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.pyplot import MultipleLocator
plt.switch_backend('agg')
from vis import mpl, myv
import mesher, utils
import time
# plt.rcParams['axes.linewidth'] = 1  # 图框宽度
# plt.rcParams['figure.dpi'] = 300  # plt.show显示分辨率
font = {'family': 'serif',
        'serif': 'Times New Roman',
        'weight': 'normal',
        'size': 10}
plt.rc('font', **font)

def main(set, test, mspacing, rhomin, rhomax, nsamples, chains):
    # 联合反演
    # 创建文件夹
    if not os.path.exists('picture'):
        os.mkdir('picture')
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
    # ---观测数据
    with open('modeldata/{}_gz_noise.txt'.format(set)) as f:
        lons, lats, heights, dobs = np.loadtxt(f, usecols=[0, 1, 2, 3], unpack=True)
    # ---读入真实模型
    rho_true = np.loadtxt("modeldata/{}_rho.dat".format(set))
    # ---小棱柱个数
    with open("modeldata/modelinfo_{}.txt".format(set)) as f:
        nr, nlat, nlon = np.loadtxt(f, usecols=[0, 1, 2])
    nr, nlat, nlon = int(nr), int(nlat), int(nlon)
    # 网格大小
    dlon, dlat, dr = mspacing[0], mspacing[1], mspacing[2]
    # 研究区范围
    west, east, south, north, top, bottom = -180, 180, -90, 90, 0, -3000000
    area = (west, east, south, north)
    # 网格shape
    shape = (nlon, nlat)
    
    # 剖面图位置
    profile_h = 32
    profile_v = 65
    # 使用多个核
    ncpu = 6
    # 统计最后last个模型用于画3D图，剖面图和平面图
    last = 100
    ndraws = nsamples - last
    # misfit开始画图的采样点
    nstart = 0
    
    # add all chains to one
    model_allchain = []
    #chains
    for chain in range(0, chains):
        misfit = np.loadtxt("result/" + str(set) + str(test) + "_chain" + str(chain) + "/misfit.dat")
        # 所有模型
        model_all = np.loadtxt("result/" + str(set) + str(test) + "_chain" + str(chain) + "/model.dat")
        # all
        model_allchain.append(model_all)

        #
        misfit_total_normed = misfit[:, 3]
        misfit_data_normed = misfit[:, 4]
        misfit_model_normed = misfit[:, 5]
        # -------------误差曲线
        plt.figure()
        #plt.title("misfit")
        plt.plot(np.arange(0, nsamples), misfit_total_normed)
        plt.xlim([0, nsamples])
        plt.xlabel("iterations")
        plt.ylabel(r"$U(m)$")
        ## [left, bottom, width, height]四个参数(fractions of figure)可以非常灵活的调节子图中子图的位置
        plt.axes([0.3, 0.4, 0.5, 0.3])
        plt.title(r'Last %d iterations' % (last))
        plt.plot(np.arange(ndraws, nsamples), misfit_total_normed[ndraws:])
        plt.xlim([ndraws, nsamples])
        # plt.ylim([9, 9.5])
        ax = plt.gca()
        ax.ticklabel_format(style='plain', axis='y')  # 不使用科学记数法
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.4f')) # 坐标不再加上一个数

        plt.savefig('picture/' + str(set) + str(test) + '_chain{}-misfit_1p.png'.format(chain), dpi=500,
                    bbox_inches='tight')
        # plt.show()
        plt.close()
        # -------------误差曲线--data
        plt.figure()
        #plt.title("misfit")
        plt.plot(np.arange(0, nsamples), misfit_data_normed)
        plt.xlim([0, nsamples])
        plt.xlabel("iterations")
        plt.ylabel(r"Misfit")
        ## [left, bottom, width, height]四个参数(fractions of figure)可以非常灵活的调节子图中子图的位置
        plt.axes([0.3, 0.4, 0.5, 0.3])
        plt.title(r'last %d iterations' % (last))
        plt.plot(np.arange(ndraws, nsamples), misfit_data_normed[ndraws:])
        plt.xlim([ndraws, nsamples])
        # plt.ylim([9, 9.5])
        ax = plt.gca()
        ax.ticklabel_format(style='plain', axis='y')  # 不使用科学记数法
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.4f')) # 坐标不再加上一个数

        plt.savefig('picture/' + str(set) + str(test) + '_chain{}-misfit_data.png'.format(chain), dpi=500,
                    bbox_inches='tight')
        # plt.show()
        plt.close()
        # -------------误差曲线--model
        plt.figure()
        #plt.title("misfit")
        plt.plot(np.arange(0, nsamples), misfit_model_normed)
        plt.xlim([0, nsamples])
        plt.xlabel("iterations")
        plt.ylabel(r"Misfit")
        ## [left, bottom, width, height]四个参数(fractions of figure)可以非常灵活的调节子图中子图的位置
        plt.axes([0.3, 0.4, 0.5, 0.3])
        plt.title(r'last %d iterations' % (last))
        plt.plot(np.arange(ndraws, nsamples), misfit_model_normed[ndraws:])
        plt.xlim([ndraws, nsamples])
        # plt.ylim([9, 9.5])
        ax = plt.gca()
        ax.ticklabel_format(style='plain', axis='y')  # 不使用科学记数法
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.4f')) # 坐标不再加上一个数

        plt.savefig('picture/' + str(set) + str(test) + '_chain{}-misfit_model.png'.format(chain), dpi=500,
                    bbox_inches='tight')
        # plt.show()
        plt.close()


    # convert to array
    model_allchain = np.array(model_allchain)

    # ***********************几条链的结果一起统计
    model_allchain_last = model_allchain[:, ndraws:, :].reshape(-1, model_all.shape[1])
    print(model_allchain_last.shape)


    # ------------------对modelall做分析
    # 计算每一个网格点密度值的均值和方差
    model_inv_mean = np.mean(model_allchain_last, axis=0)
    model_inv_std = np.std(model_allchain_last, axis=0)
    # 变为3D数组
    rho3D_mean = model_inv_mean.reshape(nr, nlat, nlon)
    rho3D_std = model_inv_std.reshape(nr, nlat, nlon)
    # 均值
    mesh_inv_mean = mesher.TesseroidMesh(bounds=(west, east, south, north, top, bottom), spacing=(dr, dlat, dlon))
    mesh_inv_mean.addprop('density', model_inv_mean)
    # 标准差
    mesh_inv_std = mesher.TesseroidMesh(bounds=(west, east, south, north, top, bottom), spacing=(dr, dlat, dlon))
    mesh_inv_std.addprop('std of density', model_inv_std)
    
    # -----保存数据---
    if not os.path.exists('result/' + str(set) + str(test) + '_allchain'):
        os.mkdir('result/' + str(set) + str(test) + '_allchain')
    # -----反演模型
    # 计算模型点的坐标 -----排列顺序：先lon变，再lat变，最后r变
    plons, plats, prs = mesh_inv_mean.get_xs(), mesh_inv_mean.get_ys(), mesh_inv_mean.get_zs()
    grid_rs, grid_lats, grid_lons = np.meshgrid(prs[1:], plats[1:], plons[1:], indexing='ij')
    header = "# lon, lat, r, model_inv_mean, model_inv_std"
    with open("result/" + str(set) + str(test) + "_allchain/inversion_model.dat", "w") as f:
        np.savetxt(f, np.c_[grid_lons.ravel(), grid_lats.ravel(), grid_rs.ravel(), model_inv_mean, model_inv_std], fmt="%.5f", delimiter=' ')
    # -----异常
    # 正演
    # ---------正演
    dpre_mean = tesseroidforward.gz(lons, lats, heights, mesh_inv_mean, njobs=ncpu)
    d_error = dobs - dpre_mean
    # 保存
    header = "# lons, lats, heights, dobs, dpre_mean, d_error"
    with open("result/" + str(set) + str(test) + "_allchain/inversion_anomaly.dat", 'w') as f:
        np.savetxt(f, np.c_[lons, lats, heights, dobs, dpre_mean, d_error], fmt="%.5f", delimiter=' ')

    # # 模型mean
    # myv.figure()
    # plot = myv.tesseroids(mesh_inv_mean, prop='density')
    # myv.savefig('picture/' + str(set) + str(test) + '_allchain-rhoall-mean-3D.png')
    # myv.show()
    # # std
    # myv.figure()
    # plot = myv.tesseroids(mesh_inv_std, prop='std of density')
    # myv.savefig('picture/' + str(set) + str(test) + '_allchain-rhoall-std-3D.png')
    # myv.show()
    # 数据拟合均方根RMSD& 模型恢复均方根RMSM：
    RMSD = np.sqrt(np.linalg.norm(dobs - dpre_mean) ** 2 / len(dobs))
    RMSM = np.sqrt(np.linalg.norm(rho_true - model_inv_mean) ** 2 / model_all.shape[1])
    print("RMSD:", RMSD)
    print("RMSM:", RMSM)

    # ----------重力------------
    data_shape=  (nlon+1, nlat+1)
    # ----------异常-mean
    plt.figure(figsize=(10, 2))
    bm = mpl.basemap(area, 'robin')
    plt.subplot(131)
    plt.title("d_obs")
    mpl.contourf(lons, lats, dobs, data_shape, 15, basemap=bm)
    plt.colorbar()
    plt.subplot(132)
    plt.title("d_mean")
    mpl.contourf(lons, lats, dpre_mean, data_shape, 15, basemap=bm)
    plt.colorbar()
    plt.subplot(133)
    plt.title("d_error")
    mpl.contourf(lons, lats, d_error, data_shape, 15, basemap=bm)
    plt.colorbar()

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
    plt.savefig('picture/' + str(set) + str(test) + '-allchain-Anomaly.png', dpi=500, bbox_inches='tight')
    #plt.show()
    plt.close()

    # 模型切片图-mean
    plt.figure(figsize=(10, 2))
    for i in range(nr):
        plt.suptitle('inversion density model (mean)')
        ax = plt.subplot(2, 5, i + 1)
        plt.pcolor(rho3D_mean[i, :, :], vmin=rhomin, vmax=rhomax, cmap=plt.cm.jet)
        plt.colorbar()
        if i in list([2, 3, 4, 5, 6]):
            plt.plot(frame_plane1[:, 0], frame_plane1[:, 1], 'white')
            plt.plot(frame_plane2[:, 0], frame_plane2[:, 1], 'white')
        if i in list([2, 3, 4, 5]):  
            plt.plot(frame_plane3[:, 0], frame_plane3[:, 1], 'white')
        if i in list([2, 3, 4]): 
            plt.plot(frame_plane4[:, 0], frame_plane4[:, 1], 'white')
            plt.plot(frame_plane5[:, 0], frame_plane5[:, 1], 'white')
        plt.xticks(np.linspace(0, nlon, 5))
        plt.yticks(np.linspace(0, nlat, 5))
        plt.xlim([0, nlon])
        plt.ylim([0, nlat])
        plt.xlabel("Easting")
        plt.ylabel("Northing")


    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=1, hspace=0.5)
    plt.savefig('picture/'+ str(set) + str(test) + '-allchain-rhomeanPlane.png', dpi=500, bbox_inches='tight')
    #plt.show()
    plt.close()

    # -------剖面图profile_h
    plt.figure(figsize=(10, 5))
    ax = plt.subplot()
    plt.title('density profile: lat(layer) = {}'.format(profile_h))
    plt.pcolor(rho3D_mean[:, profile_h, :], vmin=rhomin, vmax=rhomax, cmap=plt.cm.jet)
    cb = plt.colorbar()
    # cb.set_label('$g/cm^3$')
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

    plt.savefig('picture/'+ str(set) + str(test) + 'allchain_rhomean-profile_h.png', dpi=500, bbox_inches='tight')
    #plt.show()
    plt.close()

    # -------剖面图profile_v
    plt.figure(figsize=(10, 5))
    ax = plt.subplot()
    plt.title('density profile: lon(layer) = {}'.format(profile_v))
    plt.pcolor(rho3D_mean[:, :, profile_v], vmin=rhomin, vmax=rhomax, cmap=plt.cm.jet)
    cb = plt.colorbar()
    # cb.set_label('$g/cm^3$')
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

    plt.savefig('picture/'+ str(set) + str(test) + 'allchain_rhomean-profile_v.png', dpi=500, bbox_inches='tight')
    #plt.show()
    plt.close()


    print("Successfully plot all %d chains by analysing last %d samples of each chain" % (chains, last))

if __name__ == "__main__":
    # 查看电脑的总的内存信息
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
    mspacing =  parameters[attempt]["mspacing"]
    print(set + '\t' + test)
    print("Begain to calculate at ", time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()))

    main(set, test, mspacing, rhomin, rhomax, nsamples, chains)