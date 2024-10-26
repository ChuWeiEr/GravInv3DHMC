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
    
     # 模型地形
    with open('data/topo_12d05d.dat') as f:
        lons_topo, lats_topo, data_topo = np.loadtxt(f, usecols=[0, 1, 2], unpack=True)
    # ---重力异常数据
    with open('data/gravinv_12d05d.dat') as f:
        lons, lats, heights, dobs = np.loadtxt(f, usecols=[0, 1, 2, 3], unpack=True)
    # 固定网格单元的异常（水层）
    with open('data/grasea_12d05d.dat') as f:
        grav_sea = np.loadtxt(f, usecols=[2], unpack=True)

    # mask的index
    with open("modeldata/maskindex_{}.txt".format(set)) as f:
        mask = np.loadtxt(f, usecols=[0])
    with open("modeldata/mrs_index_{}.txt".format(set)) as f:
        mrs = np.loadtxt(f)
    # ---小棱柱个数
    with open("modeldata/modelinfo_{}.txt".format(set)) as f:
        nr, nlat, nlon = np.loadtxt(f, usecols=[0, 1, 2])
    nr, nlat, nlon = int(nr), int(nlat), int(nlon)
     # 正常规则网格的密度--赋值np.nan，用于画出地形
    rho_mesh = np.ones([nr * nlat * nlon]) * np.nan
    # 研究区范围
    west, east, south, north, top, bottom = 106.5, 118.5, 16, 28, 2000, -60000
    # 网格大小
    dlon, dlat, dr = mspacing[0], mspacing[1], mspacing[2]
     # 创建模型
    mrange = (west, east, south, north, top, bottom)
    mspacing = (dr, dlat, dlon)
    mdivisionsection = [top, -5000, -15000, bottom]  # 分段
    # data网格shape
    data_shape = (25, 25)

    # 使用多个核
    ncpu = 10
    # 统计最后last个模型用于画3D图，剖面图和平面图
    last = 100
    ndraws = nsamples - last
    # add all chains to one
    model_allchain = []
    #chains
    for chain in range(0, chains):
        misfit = np.loadtxt("result/" + str(set) + str(test) + "_chain" + str(chain) + "/misfit.dat")
        # 所有模型
        #model_all = np.loadtxt("result/" + str(set) + str(test) + "_chain" + str(chain) + "/model.dat")
        with open("result/" + str(set) + str(test) + "_chain" + str(chain) + "/model.dat") as f:
            for i in range(ndraws):
                f.readline()
            model_all = np.loadtxt(f)
        # all
        model_allchain.append(model_all)
        
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


    # convert to array
    model_allchain = np.array(model_allchain)
    
    # ***********************几条链的结果一起统计
    model_allchain_last = model_allchain[:, :, :].reshape(-1, model_all.shape[1])
    print(model_allchain_last.shape)

    # ------------------对modelall做分析
    # 计算每一个网格点密度值的均值和方差
    model_inv_mean = np.mean(model_allchain_last, axis=0)
    model_inv_std = np.std(model_allchain_last, axis=0)

    # 将密度值转换回规则网格rho_mesh
    model_inv_mean_regular = utils.carve2rho(model_inv_mean, rho_mesh, mask)
    model_inv_std_regular = utils.carve2rho(model_inv_std, rho_mesh, mask)
    # 变为3D数组
    rho3D_mean = model_inv_mean_regular.reshape(nr, nlat, nlon)
    rho3D_std = model_inv_std_regular.reshape(nr, nlat, nlon)
    # 使用绝对密度画图
    real3D_model = rho3D_mean + 3
    # 均值
    mesh_inv_mean = mesher.TesseroidMeshSegment(mrange, mspacing, mdivisionsection)
    mesh_inv_mean.addprop('density', model_inv_mean_regular)
    mesh_inv_mean.carvetopo(lons_topo, lats_topo, data_topo)
    # 标准差
    mesh_inv_std = mesher.TesseroidMeshSegment(mrange, mspacing, mdivisionsection)
    mesh_inv_std.addprop('std of density', model_inv_std_regular)
    mesh_inv_std.carvetopo(lons_topo, lats_topo, data_topo)
    # -----保存数据---
    if not os.path.exists('result/' + str(set) + str(test) + '_allchain'):
        os.mkdir('result/' + str(set) + str(test) + '_allchain')
    # -----反演模型
    # 计算模型点的坐标 -----排列顺序：先lon变，再lat变，最后r变 
    plons, plats, prs = mesh_inv_mean.get_xs(), mesh_inv_mean.get_ys(), mesh_inv_mean.get_zs()
    grid_rs, grid_lats, grid_lons = np.meshgrid(prs[:-1], plats[:-1]+dlat/2, plons[:-1]+dlon/2, indexing='ij')
    header = "# lon, lat, r, model_inv_mean, model_inv_std"
    with open("result/" + str(set) + str(test) + "_allchain/inversion_model.dat", "w") as f:
        np.savetxt(f, np.c_[grid_lons.ravel(), grid_lats.ravel(), grid_rs.ravel(), 
        model_inv_mean_regular, model_inv_std_regular], fmt="%.8f", delimiter=' ')
    
    # -----异常
    dpre_mean = tesseroidforward.gz(lons, lats, heights, mesh_inv_mean, njobs=ncpu)
    dpre_all_mean = dpre_mean + grav_sea
    # 误差
    d_error = (dobs-np.mean(dobs)) - (dpre_all_mean-np.mean(dpre_all_mean))
    # 保存
    header = "#[lons, lats, heights, dobs, dpre_mean, dpre_all_mean, d_error"
    with open("result/" + str(set) + str(test) + "_allchain/inversion_anomaly.dat", 'w') as f:
        f.write(header)
        f.write('\n')
        np.savetxt(f, np.c_[lons, lats, heights, dobs, dpre_mean, dpre_all_mean, d_error], 
        fmt="%.8f", delimiter=' ')

    
    # # 模型mean3D
    # myv.figure()
    # plot = myv.tesseroids(mesh_inv_mean, prop='density')
    # myv.savefig('picture/' + str(set) + str(test) + '_allchain-rhoall-mean-3D.png')
    # myv.show()
    # # std
    # myv.figure()
    # plot = myv.tesseroids(mesh_inv_std, prop='std of density')
    # myv.savefig('picture/' + str(set) + str(test) + '_allchain-rhoall-std-3D.png')
    # myv.show()

    
    # ----------重力异常-mean
    plt.figure(figsize=(10, 2))
    bm = mpl.basemap((west, east, south, north), 'merc')
    plt.subplot(131)
    plt.title("d_obs")
    mpl.contourf(lons, lats, dobs, data_shape, 15, basemap=bm)
    plt.colorbar()
    bm.drawmeridians(np.linspace(west, east, 5), labels=[0, 0, 0, 1], linewidth=0.2)  # 经线
    bm.drawparallels(np.linspace(south, north, 5), labels=[1, 0, 0, 0], linewidth=0.2)  # 纬线
    
    plt.subplot(132)
    plt.title("d_pre")
    mpl.contourf(lons, lats, dpre_all_mean, data_shape, 15, basemap=bm)
    plt.colorbar()
    bm.drawmeridians(np.linspace(west, east, 5), labels=[0, 0, 0, 1], linewidth=0.2)  # 经线
    bm.drawparallels(np.linspace(south, north, 5), labels=[1, 0, 0, 0], linewidth=0.2)  # 纬线

    plt.subplot(133)
    plt.title("d_error")
    mpl.contourf(lons, lats, d_error, data_shape, 15, basemap=bm)
    plt.colorbar()
    bm.drawmeridians(np.linspace(west, east, 5), labels=[0, 0, 0, 1], linewidth=0.2)  # 经线
    bm.drawparallels(np.linspace(south, north, 5), labels=[1, 0, 0, 0], linewidth=0.2)  # 纬线

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
    plt.savefig('picture/' + str(set) + str(test) + '-allchain-Anomaly.png', dpi=500, bbox_inches='tight')
    #plt.show()
    plt.close()
    
    # 模型切片图-mean  vmin=rhomin, vmax=rhomax, 
    plt.figure(figsize=(20, 12))
    for i in range(nr):
        plt.suptitle('inversion density model (mean)')
        ax = plt.subplot(5, 5, i + 1)
        plt.title(mrs[i])
        plt.pcolor(real3D_model[i, :, :], cmap=plt.cm.jet)
        plt.colorbar()
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

    
    # -------剖面图 vmin=rhomin, vmax=rhomax, 
    for latpf in range(0, 24, 5):
        plt.figure(figsize=(10, 3))
        ax = plt.subplot()
        plt.title('density profile: lat(index) = {}'.format(latpf))
        plt.pcolor(real3D_model[:, latpf, :], cmap=plt.cm.jet)
        cb = plt.colorbar()
        cb.set_label('$g/cm^3$')
        plt.xticks(np.linspace(0, nlon, 5))
        plt.yticks(np.arange(0, nr, 5), mrs[::5])
        plt.xlim([0, nlon])
        plt.ylim([0, nr])
        plt.xlabel("Easting")
        plt.ylabel("Radius")
        ax.invert_yaxis()

        plt.savefig('picture/'+ str(set) + str(test) + '-allchain-rhomean-latpf{}.png'.format(latpf), dpi=500, bbox_inches='tight')
        #plt.show()
        plt.close()
    
    for lonpf in range(0, 24, 5):
        plt.figure(figsize=(10, 3))
        ax = plt.subplot()
        plt.title('density profile: lon(index) = {}'.format(lonpf))
        plt.pcolor(rho3D_mean[:, :, lonpf], cmap=plt.cm.jet)
        cb = plt.colorbar()
        cb.set_label('$g/cm^3$')
        plt.xticks(np.linspace(0, nlat, 5))
        plt.yticks(np.arange(0, nr, 5), mrs[::5])
        # plt.yticks(mrs)
        plt.xlim([0, nlat])
        plt.ylim([0, nr])
        plt.xlabel("Norting")
        plt.ylabel("Radius")
        ax.invert_yaxis()

        plt.savefig('picture/'+ str(set) + str(test) + '-allchain-rhomean-lonpf{}.png'.format(lonpf), dpi=500, bbox_inches='tight')
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