# The GravInv3DHMC package
version of 2024-06-30 by ChuWei

## 安装
conda create HMCINV
conda activate HMCINV
pip install numpy,scipy,matplotlib,numba
（出现包导入错误可以相应安装）

此外，使用前还需运行
gravmag/setup.py

## GravInv3DHMC使用说明

本手册以5个示例介绍GravInv3DHMC代码的使用方法，后续不同的建模反演可参照以上示例修改脚本，实现不同建模方式的反演。本算法中重力建模方法包括模型均匀剖分、分段剖分、等比率剖分和模型带地形剖分，具体实现分别可见example文件夹下的uniformgrid, segmentgrid, ratiogrid和carvetopogrid。此外，realdata为一个实例脚本，同时使用分段剖分和带地形剖分方式，并使用了先验模型实现三维重力反演，可作为一个综合示例。

### HMC部分
	不同建模方法均需输入重力异常数据，数据格式如下：
-	重力异常数据：四列（经度、纬度、观测面高度、异常值）
如果可获得初始模型或者先验模型作为输入数据，数据格式如下：
-	初始模型/先验模型：四列（经度、纬度、深度、密度值）
注意：初始模型/先验模型的密度值网格剖分应该与反演模型剖分方式完全一致，如不一致，可以使用Make_InitialAprior_Model.py做插值。

准备好输入数据之后，修改主程序main_*.py，调整反演参数SetPMTS.txt，在linux终端输入：bash run_main.sh，即可运行程序。
接下来以5个示例具体介绍代码用法。
#### 1	realdata

- 重力异常数据准备
data文件夹中gravity_12d05d.dat和gravobs_12d05d.dat分别为重力异常数据和重力异常观测面高度数据，使用make_grav4inv.py将这两个数据整理为反演所需格式（经度、纬度、观测面高度、异常值），得到反演重力异常数据gravinv_12d05d.dat。
建模时考虑了地形，将水层网格固定住不参与反演，因此在反演过程中需要从总的重力异常中减去水层重力异常值。因此计算水层重力异常值grasea_12d05d.dat作为输入文件。
输入地形数据topo_12d05d.dat用于建模，地形数据所在坐标系向上为正，向下为负。

-	初始模型/先验模型构建
首先确定反演的建模格式，然后使用Make_InitialAprior_Model.py构建初始模型或者先验模型。
例如已有密度模型SC_Rho3D.txt，其数据排列格式与反演建模所需不同，因此需要重新做插值，修改Make_InitialAprior_Model.py脚本模型构建（29-36行）部分的参数。修改输入、输出文件，即可得到用于反演的初始或先验模型。

-	主程序建模
main_real.py为主程序。
创建模型部分（23-29行）与先验模型构建参数保持一致。
输入数据部分（31-40行）修改相应的文件名。
实例化类部分（43-46行）：
 
dobs为输入重力异常数据；
mrange, mspacing为建模参数，分别为建模范围和单个剖分棱柱大小；
obsurface为重力异常观测面；
fixed=True判断是否固定某些网格不参与反演，默认为False，如果设置为True，需要输入不参与反演网格位置处产生的重力异常值，以grav_fix=grav_sea的方式传入；
mseg=True判断是否进行分段剖分，默认为False，如果设置为True，需要数据分段剖分的节点值，以mdivisionsection=mdivisionsection的方式传入；
coordinate="spherical", njobs=5, field="gravity"分别为建模坐标系选择（可选"spherical"或者"cartesian"），多进程计算使用核数，计算物理场（可选"gravity"或者"magnetic"）。注：目前算法不支持"spherical"和"magnetic"的组合；
wavelet=False选择是否对核矩阵做小波压缩（可选False, '1D'或'3D'），默认不使用小波压缩，即False，'1D'或'3D'分别代表使用1D小波压缩和3D小波压缩。带地形反演由于核矩阵不规则，不能做小波压缩。
mtopo=(lons_topo, lats_topo, data_topo)是当使用带地形建模时，需要输入的地形参数。
保存模型参数信息（48-64行）：
modelinfo_{}.txt保存整个模型空间在三个维度分别划分为多少个网格；
mlons_index_{}.txt、mlats_index_{}.txt、mrs_index_{}.txt保存模型空间在三个维度剖分网格点的位置信息；
maskindex_{}.txt保存了地形之上（即切掉不参与反演的）网格的位置（index）。
输入初始模型和先验模型（65-74行）：初始模型赋值为0，即不使用初始模型；读入上一步构建的结果作为先验模型，由于构建的模型为规则网格，带地形模型建模时将地形之上的网格删除，使其不参与反演，因此同样需要将输入的先验模型地形之上位置的密度值删除，代码74行即完成这一功能。
参数设置（76-98行）的参数设置不需要修改，修改SetPMTS.txt文件即可，见下一节。

-	参数设置
修改SetPMTS.txt文件中各项参数：
"set": "SC", "test": "T0"用于保存数据时的文件命名，分别为研究区名称、测试编号；
"rhomin": -0.5, "rhomax": 0.5为反演时给定物性参数范围；
"mspacing": [0.5, 0.5, [-1000, -2000, -5000]]为建模时，当个棱柱体大小；
"Lrange": [5, 20], "delta": 0.01, "Sigma": 0.01为反演参数设置，分别为步数、步长和质量矩阵；
"RegulFactor": 1, "regularization": "Damping", "beta": 0.01分别为正则化因子、正则化项选择（可选"MS"、"Damping"、"Smoothness"或"TV"），当选择"MS"、"TV"时设定beta为合适的值，当选择"Damping"、"Smoothness"时不需要修改beta的值；
"nsamples": 4000为HMC算法的采样次数。

修改以上参数，不同参数组合作为不同的测试，每个不同的参数测试为SetPMTS.txt文件中的一行。

-	代码运行

run_main.sh为运行主程序main_real.py的bash脚本，运行一次可以同时完成SetPMTS.txt中所有的参数测试，运行时，在终端输入 bash run_main.sh
代码中的参数含义：
代码1-11行用来创建文件夹modeldata和result，用于存放数据；
代码12-18行需要修改参数
 
主程序运行后的输出信息存储在logout_T*.txt文件中，修改参数name和i用于logout文件的命名；
nohup mpiexec 用于在系统后台不挂断地运行命令，退出终端不会影响程序的运行。
从上向下第3个红框中的2代表在反演计算时，设置2条链并行计算。

- 成图
plot_real_multichain.py脚本实现对反演结果成图，可以在其中修改对应参数，最终代码运行方法，在终端输入 bash run_plotMC.sh

#### 2	uniformgrid

使用均匀剖分网格建模反演
model01_singlecube.py为建模脚本，建立模型并正演重力异常；
 
研究区建模范围xmin, xmax, ymin, ymax, zmin, zmax = 0, 2000, 0, 3000, 0, 1000
均匀剖分网格大小dx, dy, dz = 100, 100, 100

main_uniform.py为反演主程序
 
dobs为输入重力异常数据；
mrange, mspacing为建模参数，分别为建模范围和单个剖分棱柱大小,
mrange = (xmin, xmax, ymin, ymax, zmin, zmax)
mspacing = (dz, dy, dx)；
obsurface为重力异常观测面；
coordinate="cartesian", njobs=5, field="gravity"分别为建模坐标系选择，多进程计算使用核数，计算物理场；
wavelet='3D'。
run_main.sh为运行主程序main_real.py的bash脚本，运行一次可以同时完成SetPMTS.txt中所有的参数测试，运行时，在终端输入 bash run_main.sh

plot_uniform.py为画图脚本，每条链单独统计成图，在终端输入 bash run_plotMC.sh实现对反演结果成图

#### 3	ratiogrid
使用深度方向等比率剖分网格建模反演
model_ratio.py为建模脚本，建立模型并正演重力异常；
 
main_ratio.py为反演主程序
 
dobs为输入重力异常数据；
mrange, mspacing为建模参数，分别为建模范围和单个剖分棱柱大小,
mrange = (xmin, xmax, ymin, ymax, zmin, zmax)
mspacing = (dz, dy, dx)；
obsurface为重力异常观测面；
mratio为等比率剖分的比率因子，默认为1；
coordinate="cartesian", njobs=5, field="gravity"分别为建模坐标系选择，多进程计算使用核数，计算物理场；
wavelet='3D'。
run_main.sh为运行主程序main_real.py的bash脚本，运行一次可以同时完成SetPMTS.txt中所有的参数测试，运行时，在终端输入 bash run_main.sh

#### 4	global

model_global.py为建模脚本，建立模型并正演重力异常；
 
main_global.py为反演主程序
 
dobs为输入重力异常数据；
mrange, mspacing为建模参数，分别为建模范围和单个剖分棱柱大小；
obsurface为重力异常观测面；
coordinate="spherical", njobs=5, field="gravity"分别为建模坐标系选择，多进程计算使用核数，计算物理场；
run_main.sh为运行主程序main_global.py的bash脚本，运行一次可以同时完成SetPMTS.txt中所有的参数测试，运行时，在终端输入 bash run_main.sh

plot_model_global.py为画图脚本，每条链单独统计成图，在终端输入 bash run_plotMC.sh实现对反演结果成图

#### 5	segmentgrid

使用分段剖分网格建模反演
model_seg.py为建模脚本，建立模型并正演重力异常；
 
网格大小dx, dy, dz = 100, 100, [100, 200, 300]
研究区建模范围xmin, xmax, ymin, ymax, zmin, zmax = 0, 2000, 0, 3000, 0, 2100
z方向使用分段剖分：
0~300m在z方向网格大小为100m；
300~900m在z方向网格大小为200m；
900~2100m在z方向网格大小为300m；

main_seg.py为反演主程序
 
dobs为输入重力异常数据；
mrange, mspacing为建模参数，分别为建模范围和单个剖分棱柱大小；
obsurface为重力异常观测面；
mseg=True判断是否进行分段剖分，默认为False，如果设置为True，需要数据分段剖分的节点值，以mdivisionsection=mdivisionsection的方式传入；
coordinate="cartesian", njobs=5, field="gravity"分别为建模坐标系选择，多进程计算使用核数，计算物理场；
wavelet='3D'；
run_main.sh为运行主程序main_real.py的bash脚本，运行一次可以同时完成SetPMTS.txt中所有的参数测试，运行时，在终端输入 bash run_main.sh

plot_seg.py为画图脚本，每条链单独统计成图，在终端输入 bash run_plotMC.sh实现对反演结果成图

其他常用命令

ps aux | grep *  查看正在运行的进程

### CG部分
 
dobs为输入重力异常数据；
mrange, mspacing为建模参数，分别为建模范围和单个剖分棱柱大小；
obsurface为重力异常观测面；

initial_model：初始模型
aprior_model：先验模型
boundary：密度值约束区间
regularization、beta：正则化项选择、参数
q：自适应正则化因子的衰减系数
maxk：最大迭代次数


##### 软件包使用方法
建议新建一个文件夹用于存放主程序，例如新建example文件夹，在其中编写建模和反演代码，运行后输出数据和图片。


##### reference
https://www.fatiando.org/
博士论文：《基于 HMC 采样的三维重力反演及其在华南陆缘密度结构研究-褚伟》