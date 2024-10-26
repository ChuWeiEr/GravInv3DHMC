'''
potential energy function of hmc
**List of classes**

* :class:`~GravMagInversion3D.inversion.potential.GravMagModule`:
gravity or magnetic inversion in cartesian or spherical coordinate using hmc

* :class:`~GravMagInversion3D.inversion.potential.JointModule`:
gravity and magnetic joint inversion in cartesian or spherical coordinate using hmc

++++++++++References
ChuWei 2022.06.30
'''

import os
import psutil
import sys
import gc

import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.pyplot import MultipleLocator
plt.switch_backend('agg')
from scipy.optimize import curve_fit
from scipy.sparse import coo_matrix, csr_matrix
from gravmag import prism, tesseroid
from gravmag import compressor1D as cp1D
from gravmag import compressor3D as cp3D
import mesher, utils
from vis import mpl, myv
# “*”运算是单纯做对应位置元素的数乘，“np.dot()”和“@”运算都可以起到矩阵乘法的作用。
class GravMagModule():
    def __init__(self, dobs, mrange, mspacing, obsurface, fixed=False, grav_fix=[],
                mratio=1, mseg=False, mdivisionsection=[], weightfactor=0.5, 
                coordinate="cartesian", njobs=1, field="gravity",
                mangle=(90, 0), wavelet=False, **kwargs):
        '''
        y->East, x->North, and z->Down
        # 球坐标系
        # mrange = (west, east, south, north, top, bottom)
        # mspacing = (dr, dlat, dlon)
        # obsurface = [lons, lats, height]
        # 笛卡尔坐标系
        # mrange = (xmin, xmax, ymin, ymax, zmin, zmax)
        # mspacing = (dz, dy, dx)
        # obsurface = [xobs, yobs, height]
        # mtopo = (x, y, topography)  # if use topography, z up is positive
        # fixed是否有网格不参与反演，若有则输入固定网格的异常grav_fixed
        # 其他参数
        # mratio=1,模型深度方向自适应剖分
        # mangle= (inc, dec) 用于磁场计算
        # coordinate="cartesian" or "spherical"
        # field="gravity"or"magnetic"
        # wavelet = False, 1D, 3D 三个选项

        '''
        # observation data
        self.dobs = dobs
        # fixed prism with gravity
        self.fixed = fixed
        self.grav_fix = grav_fix
        # model parameters
        self.mrange = mrange
        self.mspacing = mspacing
        self.mratio = mratio
        self.weightfactor = weightfactor
        # The z direction is modeled piecewise
        self.mseg = mseg
        self.mdivisionsection = mdivisionsection
        # observation surface parameters
        self.lonobs = obsurface[0]
        self.latobs = obsurface[1]
        self.heightobs = obsurface[2]
        # inc, dec of magnetic
        self.inc = mangle[0]
        self.dec = mangle[1]
        # multiple cores
        self.njobs = njobs
        # topography
        self.topocarve = False
        # wavelet
        self.wavelet = wavelet

        if coordinate == "spherical" and field == "gravity":
            # tesseroid && density
            print("Calculating {} field in {} coordinate.".format(field, coordinate))
            if self.mseg:
                mesh = mesher.TesseroidMeshSegment(self.mrange, self.mspacing, self.mdivisionsection)
            else:
                mesh = mesher.TesseroidMesh(self.mrange, self.mspacing, self.mratio)
            # topography
            for key, value in kwargs.items():
                self.topocarve = True
                mtopo = value
                mask = mesh.carvetopo(mtopo[0], mtopo[1], mtopo[2])
                self.mask = mask
            mesh.addprop('density', np.zeros(mesh.size))
            self.mesh = mesh
            print("Start of calculate kernel")
            start = time.time()
            _, kernel = tesseroid.gz(self.lonobs, self.latobs, self.heightobs, self.mesh, njobs=self.njobs)
            end = time.time()
            print("End of calculate kernel:%.6f s" % (end - start))
        elif coordinate == "spherical" and field == "magnetic":
            print("Calculating {} field in {} coordinate.".format(field, coordinate))
            pass
        elif coordinate == "cartesian" and field == "gravity":
            print("Calculating {} field in {} coordinate.".format(field, coordinate))
            if self.mseg:
                mesh = mesher.PrismMeshSegment(self.mrange, self.mspacing, self.mdivisionsection)
            else:
                mesh = mesher.PrismMesh(self.mrange, self.mspacing, self.mratio)
            # topography
            for key, value in kwargs.items():
                self.topocarve = True
                mtopo = value
                mask = mesh.carvetopo(mtopo[0], mtopo[1], mtopo[2])
                self.mask = mask
            mesh.addprop('density', np.zeros(mesh.size))
            self.mesh = mesh
            print("Start of calculate kernel")
            start = time.time()
            _, kernel = prism.gz(self.lonobs, self.latobs, self.heightobs, self.mesh, njobs=self.njobs)
            print("kernel.shape",kernel.shape)
            end = time.time()
            print("End of calculate kernel:%.6f s" % (end - start))
        elif coordinate == "cartesian" and field == "magnetic":
            print("Calculating {} field in {} coordinate.".format(field, coordinate))
            mesh = mesher.PrismMesh(self.mrange, self.mspacing, self.mratio)
            # # ---kernel Li 1996用到
            # mesh.addprop('density', np.zeros(mesh.size))
            # _, kernelNoTopo = prism.gz(self.lonobs, self.latobs, self.heightobs, mesh)
            # self.kernelNoTopo = kernelNoTopo
            # topography
            for key, value in kwargs.items():
                self.topocarve = True
                mtopo = value
                mask = mesh.carvetopo(mtopo[0], mtopo[1], mtopo[2])
                self.mask = mask
            mesh.addprop('magnetization', utils.ang2vec(np.zeros(mesh.size), self.inc, self.dec))
            self.mesh = mesh
            print("Start of calculate kernel")
            start = time.time()
            _, kernel = prism.tf(self.lonobs, self.latobs, self.heightobs,
                                           self.mesh, self.inc, self.dec, njobs=self.njobs)
            end = time.time()
            print("End of calculate kernel:", end - start)
        else:
            raise ValueError("Please choose coordinate from(cartesian, spherical) and field from(gravity, magnetic)!")

        # 各方向剖分的网格单元个数 self.mshape = [nz, ny, nx]
        self.mshape = mesh.shape
        # 各方向网格单元坐标
        self.mxs = mesh.get_xs()
        self.mys = mesh.get_ys()
        self.mzs = mesh.get_zs()
        # weight kernel
        self.A = kernel  # define kernel
        print("Start to weight kernel")
        start = time.time()
        self.sensitivityWeighting()  # 敏感核矩阵加权
        end = time.time()
        print("End of weighting kernel: %.6f s" % (end - start))
        # 释放核矩阵变量
        del kernel,self.A
        gc.collect()
        # 使用小波压缩，压缩核矩阵Aw,成为Awcp
        if wavelet == '1D':
            # 带地形必须使用1D
            print("Using {} wavelet to compress kernel.".format(wavelet))
            Awcp = cp1D.kernelcompressor(self.Aw)
            self.Awcp = Awcp
        if wavelet == '3D':
            print("Using {} wavelet to compress kernel.".format(wavelet))
            Awcp = cp3D.kernelcompressor(self.Aw, self.mshape)
            self.Awcp = Awcp
        
        # # 作图
        # self.plotmatrix()
        # self.plotdataw()

    def plotmatrix(self):
        # ------作图,画出几个矩阵
        plt.figure()
        plt.title("Wm")
        plt.contourf(self.Wm.todense(), cmap=plt.cm.jet)
        plt.colorbar()
        plt.savefig('picture/matrix of Wm.png')
        #plt.show()
        plt.close()

        plt.figure()
        plt.title("WmInv")
        plt.contourf(self.WmInv.todense(), cmap=plt.cm.jet)
        plt.colorbar()
        plt.savefig('picture/matrix of WmInv.png')
        #plt.show()
        plt.close()

        plt.figure()
        plt.title("Wmsquare")
        plt.contourf(self.WmSquare.todense(), cmap=plt.cm.jet)
        plt.colorbar()
        plt.savefig('picture/matrix of WmSquare.png')
        #plt.show()
        plt.close()

    def plotdataw(self):
        # ------作图,画出核矩阵和数据
        plt.figure()
        plt.title("A")
        plt.contourf(self.A, cmap=plt.cm.jet)
        plt.colorbar()

        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.6, hspace=0.5)
        plt.savefig('picture/kernel A.png')
        #plt.show()
        plt.close()

        plt.figure()
        plt.title("Aw")
        plt.contourf(self.Aw, cmap=plt.cm.jet)
        plt.colorbar()

        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.6, hspace=0.5)
        plt.savefig('picture/kernel Aw.png')
        #plt.show()
        plt.close()

    def sensitivityWeighting(self):
        """
        使用灵敏度矩阵加权
        输入核矩阵
        计算Wm加权之后的核矩阵
        """
        # 计算ADiag,是一个M维向量
        # 逐个元素计算，速度更快
        ADiagSquare = np.zeros(self.A.shape[1])
        for i in range(self.A.shape[1]):
            ADiagSquare[i] = 0
            for j in range(self.A.shape[0]):
                ADiagSquare[i] += self.A[j, i]**2
        ADiag = np.power(ADiagSquare, self.weightfactor)
        # 计算对角元素的倒数
        for i in range(ADiag.shape[0]):
            if abs(ADiag[i]) == 0:
                ADiagInv = 0
            else:
                ADiagInv = 1.0 / ADiag
        # 计算对角元素的平方
        ADiagSquare = ADiag * ADiag
        # 计算加权矩阵,构造稀疏矩阵存储加权矩阵
        row = np.arange(0, self.A.shape[1])  # 对角线坐标，row=col
        Wm = coo_matrix((ADiag, (row, row))).tocsr()  # 将对角元素放到矩阵对角线上，得到加权矩阵Wm
        WmInv = coo_matrix((ADiagInv, (row, row))).tocsr()  # 加权矩阵Wm的逆矩阵
        WmSquare = coo_matrix((ADiagSquare, (row, row))).tocsr()  # 加权矩阵Wm的平方
        Aw = self.A @ WmInv  # 计算新的核矩阵
        # 定义
        self.Aw = Aw
        self.Wm = Wm
        self.WmInv = WmInv
        self.WmSquare = WmSquare

    def fd3d(self, shape):
        """
        Produce a 3D finite difference matrix.

        Parameters:

        * shape : tuple = (nz, ny, nx)
            The shape of the parameter grid. Number of parameters in the z, y and x
            dimensions.

        Returns:

        * fd : sparse CSR matrix
            The finite difference matrix
        Examples:

        >>> fd3d((2, 2, 2)).todense()
        matrix([[ 1 -1  0  0  0  0  0  0]
                 [ 0  0  1 -1  0  0  0  0]
                 [ 1  0 -1  0  0  0  0  0]
                 [ 0  1  0 -1  0  0  0  0]
                 [ 0  0  0  0  1 -1  0  0]
                 [ 0  0  0  0  0  0  1 -1]
                 [ 0  0  0  0  1  0 -1  0]
                 [ 0  0  0  0  0  1  0 -1]
                 [ 1  0  0  0 -1  0  0  0]
                 [ 0  1  0  0  0 -1  0  0]
                 [ 0  0  1  0  0  0 -1  0]
                 [ 0  0  0  1  0  0  0 -1]])
        >>> fd3d((3, 2, 2)).todense()
        matrix([[ 1 -1  0  0  0  0  0  0  0  0  0  0]
                 [ 0  0  1 -1  0  0  0  0  0  0  0  0]
                 [ 1  0 -1  0  0  0  0  0  0  0  0  0]
                 [ 0  1  0 -1  0  0  0  0  0  0  0  0]
                 [ 0  0  0  0  1 -1  0  0  0  0  0  0]
                 [ 0  0  0  0  0  0  1 -1  0  0  0  0]
                 [ 0  0  0  0  1  0 -1  0  0  0  0  0]
                 [ 0  0  0  0  0  1  0 -1  0  0  0  0]
                 [ 0  0  0  0  0  0  0  0  1 -1  0  0]
                 [ 0  0  0  0  0  0  0  0  0  0  1 -1]
                 [ 0  0  0  0  0  0  0  0  1  0 -1  0]
                 [ 0  0  0  0  0  0  0  0  0  1  0 -1]
                 [ 1  0  0  0 -1  0  0  0  0  0  0  0]
                 [ 0  1  0  0  0 -1  0  0  0  0  0  0]
                 [ 0  0  1  0  0  0 -1  0  0  0  0  0]
                 [ 0  0  0  1  0  0  0 -1  0  0  0  0]
                 [ 0  0  0  0  1  0  0  0 -1  0  0  0]
                 [ 0  0  0  0  0  1  0  0  0 -1  0  0]
                 [ 0  0  0  0  0  0  1  0  0  0 -1  0]
                 [ 0  0  0  0  0  0  0  1  0  0  0 -1]])

        """
        nz, ny, nx = shape
        nderivs = ((nx - 1) * ny + (ny - 1) * nx) * nz + nx * ny * (nz - 1)
        I, J, V = [], [], []

        # 先写出所有单层
        for k in range(nz):
            deriv = 0
            param = 0
            # 某一层
            # nx方向
            for i in range(ny):
                for j in range(nx - 1):
                    I.extend([((nx - 1) * ny + (ny - 1) * nx) * k + deriv, ((nx - 1) * ny + (ny - 1) * nx) * k + deriv])
                    J.extend([nx * ny * k + param, nx * ny * k + param + 1])
                    V.extend([1, -1])
                    deriv += 1
                    param += 1
                param += 1

            # ny方向
            param = 0
            for i in range(ny - 1):
                for j in range(nx):
                    I.extend([((nx - 1) * ny + (ny - 1) * nx) * k + deriv, ((nx - 1) * ny + (ny - 1) * nx) * k + deriv])
                    J.extend([nx * ny * k + param, nx * ny * k + param + nx])
                    V.extend([1, -1])
                    deriv += 1
                    param += 1

        # 再写出两层之间
        front = ((nx - 1) * ny + (ny - 1) * nx) * nz
        for k in range(nz - 1):
            # nz方向
            deriv = 0
            param = 0
            for i in range(ny):
                for j in range(nx):
                    I.extend([front + nx * ny * k + deriv, front + nx * ny * k + deriv])
                    J.extend([nx * ny * k + param, nx * ny * k + param + nx * ny])
                    V.extend([1, -1])
                    deriv += 1
                    param += 1

        return coo_matrix((V, (I, J)), (nderivs, nx * ny * nz)).tocsr()

    def fd3dx(self, shape):
        """
                Produce a 3D finite difference matrix in dx direction.

                Parameters:

                * shape : tuple = (nz, ny, nx)
                    The shape of the parameter grid. Number of parameters in the z, y and x
                    dimensions.

                Returns:

                * fd : sparse CSR matrix
                    The finite difference matrix
                Examples:
                >>> fd3dx((2, 2, 2)).todense()
                matrix[[ 1 -1  0  0  0  0  0  0]
                     [ 0  0  1 -1  0  0  0  0]
                     [ 0  0  0  0  0  0  0  0]
                     [ 0  0  0  0  0  0  0  0]
                     [ 0  0  0  0  1 -1  0  0]
                     [ 0  0  0  0  0  0  1 -1]
                     [ 0  0  0  0  0  0  0  0]
                     [ 0  0  0  0  0  0  0  0]
                     [ 0  0  0  0  0  0  0  0]
                     [ 0  0  0  0  0  0  0  0]
                     [ 0  0  0  0  0  0  0  0]
                     [ 0  0  0  0  0  0  0  0]]
                >>> fd3dx((3, 2, 2)).todense()
                matrix[[ 1 -1  0  0  0  0  0  0  0  0  0  0]
                     [ 0  0  1 -1  0  0  0  0  0  0  0  0]
                     [ 0  0  0  0  0  0  0  0  0  0  0  0]
                     [ 0  0  0  0  0  0  0  0  0  0  0  0]
                     [ 0  0  0  0  1 -1  0  0  0  0  0  0]
                     [ 0  0  0  0  0  0  1 -1  0  0  0  0]
                     [ 0  0  0  0  0  0  0  0  0  0  0  0]
                     [ 0  0  0  0  0  0  0  0  0  0  0  0]
                     [ 0  0  0  0  0  0  0  0  1 -1  0  0]
                     [ 0  0  0  0  0  0  0  0  0  0  1 -1]
                     [ 0  0  0  0  0  0  0  0  0  0  0  0]
                     [ 0  0  0  0  0  0  0  0  0  0  0  0]
                     [ 0  0  0  0  0  0  0  0  0  0  0  0]
                     [ 0  0  0  0  0  0  0  0  0  0  0  0]
                     [ 0  0  0  0  0  0  0  0  0  0  0  0]
                     [ 0  0  0  0  0  0  0  0  0  0  0  0]
                     [ 0  0  0  0  0  0  0  0  0  0  0  0]
                     [ 0  0  0  0  0  0  0  0  0  0  0  0]
                     [ 0  0  0  0  0  0  0  0  0  0  0  0]
                     [ 0  0  0  0  0  0  0  0  0  0  0  0]]
        """
        nz, ny, nx = shape
        nderivs = ((nx - 1) * ny + (ny - 1) * nx) * nz + nx * ny * (nz - 1)
        I, J, V = [], [], []

        # dx方向
        for k in range(nz):
            deriv = 0
            param = 0
            # 某一层
            for i in range(ny):
                for j in range(nx - 1):
                    I.extend([((nx - 1) * ny + (ny - 1) * nx) * k + deriv, ((nx - 1) * ny + (ny - 1) * nx) * k + deriv])
                    J.extend([nx * ny * k + param, nx * ny * k + param + 1])
                    V.extend([1, -1])
                    deriv += 1
                    param += 1
                param += 1

        return coo_matrix((V, (I, J)), (nderivs, nx * ny * nz)).tocsr()

    def fd3dy(self, shape):
        """
                Produce a 3D finite difference matrix in dy direction.

                Parameters:

                * shape : tuple = (nz, ny, nx)
                    The shape of the parameter grid. Number of parameters in the z, y and x
                    dimensions.

                Returns:

                * fd : sparse CSR matrix
                    The finite difference matrix
                Examples:

                >>> fd3dy((2, 2, 2)).todense()
                matrix([[ 0  0  0  0  0  0  0  0]
                     [ 0  0  0  0  0  0  0  0]
                     [ 1  0 -1  0  0  0  0  0]
                     [ 0  1  0 -1  0  0  0  0]
                     [ 0  0  0  0  0  0  0  0]
                     [ 0  0  0  0  0  0  0  0]
                     [ 0  0  0  0  1  0 -1  0]
                     [ 0  0  0  0  0  1  0 -1]
                     [ 0  0  0  0  0  0  0  0]
                     [ 0  0  0  0  0  0  0  0]
                     [ 0  0  0  0  0  0  0  0]
                     [ 0  0  0  0  0  0  0  0]])
                >>> fd3dy((3, 2, 2)).todense()
                matrix([[ 0  0  0  0  0  0  0  0  0  0  0  0]
                     [ 0  0  0  0  0  0  0  0  0  0  0  0]
                     [ 1  0 -1  0  0  0  0  0  0  0  0  0]
                     [ 0  1  0 -1  0  0  0  0  0  0  0  0]
                     [ 0  0  0  0  0  0  0  0  0  0  0  0]
                     [ 0  0  0  0  0  0  0  0  0  0  0  0]
                     [ 0  0  0  0  1  0 -1  0  0  0  0  0]
                     [ 0  0  0  0  0  1  0 -1  0  0  0  0]
                     [ 0  0  0  0  0  0  0  0  0  0  0  0]
                     [ 0  0  0  0  0  0  0  0  0  0  0  0]
                     [ 0  0  0  0  0  0  0  0  1  0 -1  0]
                     [ 0  0  0  0  0  0  0  0  0  1  0 -1]
                     [ 0  0  0  0  0  0  0  0  0  0  0  0]
                     [ 0  0  0  0  0  0  0  0  0  0  0  0]
                     [ 0  0  0  0  0  0  0  0  0  0  0  0]
                     [ 0  0  0  0  0  0  0  0  0  0  0  0]
                     [ 0  0  0  0  0  0  0  0  0  0  0  0]
                     [ 0  0  0  0  0  0  0  0  0  0  0  0]
                     [ 0  0  0  0  0  0  0  0  0  0  0  0]
                     [ 0  0  0  0  0  0  0  0  0  0  0  0]])
                """
        nz, ny, nx = shape
        nderivs = ((nx - 1) * ny + (ny - 1) * nx) * nz + nx * ny * (nz - 1)
        I, J, V = [], [], []

        # 先写出所有单层
        for k in range(nz):
            deriv = 0
            param = 0
            # 某一层
            # nx方向
            for i in range(ny):
                for j in range(nx - 1):
                    deriv += 1
                    param += 1
                param += 1

            # ny方向
            param = 0
            for i in range(ny - 1):
                for j in range(nx):
                    I.extend([((nx - 1) * ny + (ny - 1) * nx) * k + deriv, ((nx - 1) * ny + (ny - 1) * nx) * k + deriv])
                    J.extend([nx * ny * k + param, nx * ny * k + param + nx])
                    V.extend([1, -1])
                    deriv += 1
                    param += 1

        return coo_matrix((V, (I, J)), (nderivs, nx * ny * nz)).tocsr()

    def fd3dz(self, shape):
        """
        Produce a 3D finite difference matrix in dz direction.

        Parameters:

        * shape : tuple = (nz, ny, nx)
            The shape of the parameter grid. Number of parameters in the z, y and x
            dimensions.

        Returns:

        * fd : sparse CSR matrix
            The finite difference matrix
        Examples:

        >>> fd3dz((2, 2, 2)).todense()
        matrix([[ 0  0  0  0  0  0  0  0]
             [ 0  0  0  0  0  0  0  0]
             [ 0  0  0  0  0  0  0  0]
             [ 0  0  0  0  0  0  0  0]
             [ 0  0  0  0  0  0  0  0]
             [ 0  0  0  0  0  0  0  0]
             [ 0  0  0  0  0  0  0  0]
             [ 0  0  0  0  0  0  0  0]
             [ 1  0  0  0 -1  0  0  0]
             [ 0  1  0  0  0 -1  0  0]
             [ 0  0  1  0  0  0 -1  0]
             [ 0  0  0  1  0  0  0 -1]])
        >>> fd3dz((3, 2, 2)).todense()
        matrix([[ 0  0  0  0  0  0  0  0  0  0  0  0]
             [ 0  0  0  0  0  0  0  0  0  0  0  0]
             [ 0  0  0  0  0  0  0  0  0  0  0  0]
             [ 0  0  0  0  0  0  0  0  0  0  0  0]
             [ 0  0  0  0  0  0  0  0  0  0  0  0]
             [ 0  0  0  0  0  0  0  0  0  0  0  0]
             [ 0  0  0  0  0  0  0  0  0  0  0  0]
             [ 0  0  0  0  0  0  0  0  0  0  0  0]
             [ 0  0  0  0  0  0  0  0  0  0  0  0]
             [ 0  0  0  0  0  0  0  0  0  0  0  0]
             [ 0  0  0  0  0  0  0  0  0  0  0  0]
             [ 0  0  0  0  0  0  0  0  0  0  0  0]
             [ 1  0  0  0 -1  0  0  0  0  0  0  0]
             [ 0  1  0  0  0 -1  0  0  0  0  0  0]
             [ 0  0  1  0  0  0 -1  0  0  0  0  0]
             [ 0  0  0  1  0  0  0 -1  0  0  0  0]
             [ 0  0  0  0  1  0  0  0 -1  0  0  0]
             [ 0  0  0  0  0  1  0  0  0 -1  0  0]
             [ 0  0  0  0  0  0  1  0  0  0 -1  0]
             [ 0  0  0  0  0  0  0  1  0  0  0 -1]])

        """
        nz, ny, nx = shape
        nderivs = ((nx - 1) * ny + (ny - 1) * nx) * nz + nx * ny * (nz - 1)
        I, J, V = [], [], []

        # 再写出两层之间
        front = ((nx - 1) * ny + (ny - 1) * nx) * nz
        for k in range(nz - 1):
            # nz方向
            deriv = 0
            param = 0
            for i in range(ny):
                for j in range(nx):
                    I.extend([front + nx * ny * k + deriv, front + nx * ny * k + deriv])
                    J.extend([nx * ny * k + param, nx * ny * k + param + nx * ny])
                    V.extend([1, -1])
                    deriv += 1
                    param += 1

        return coo_matrix((V, (I, J)), (nderivs, nx * ny * nz)).tocsr()

    def kernelw(self):
        '''
        采样循环使用；
        使用采样次数sample次
        '''
        return self.Aw, self.WmInv, self.Wm

    def data(self, x, low, high, constraint, log_fator):
        '''
        # 数据项,自适应正则化因子才用得到
        '''
        # # ----convert x to m
        if constraint == 'logarithmic':
            mw = (low + high * np.e ** (log_fator * x)) / (1 + np.e ** (log_fator * x))  # g/cm3
        elif constraint == 'mandatory':
            mw = x
        else:
            raise ValueError("Please choose right boundary constraint(mandatory, logarithmic)!")
        if self.wavelet == '1D':
            dpre = cp1D.modelcompressor(mw, self.Awcp)
        if self.wavelet == '3D':
            dpre = cp3D.modelcompressor(mw, self.Awcp, self.mshape)
        if not self.wavelet:
            dpre = np.dot(self.Aw, mw)
        data_value = np.linalg.norm((dpre-np.mean(dpre)) - (self.dobs-np.mean(self.dobs)))**2
        return data_value

    def model_MS(self, x, mwapr, low, high, constraint, log_fator, beta):
        '''
        模型项,自适应正则化因子才用得到
        minimum support stabilizing (MS) function
        模型项:最小支撑稳定泛函
        '''
        # ----convert x to mw
        if constraint == 'logarithmic':
            mw = (low + high * np.e ** (log_fator * x)) / (1 + np.e ** (log_fator * x))  # g/cm3
        elif constraint == 'mandatory':
            mw = x
        else:
            raise ValueError("Please choose right boundary constraint(mandatory, logarithmic)!")
        # ------model_value
        # mw^2
        mwSquare = (mw - mwapr) ** 2
        # 分子numerator; 分母denominator
        numerator_value = self.WmSquare @ mwSquare
        denominator_value = mwSquare + beta
        model_value = np.sum(numerator_value / denominator_value)  # 求和
        return model_value

    def model_Damping(self, x, mwapr, low, high, constraint, log_fator):
        '''
        Damping (0th order Tikhonov) regularization.
        Imposes the minimum norm of the parameter vector.
        '''
        # ----convert x to mw
        if constraint == 'logarithmic':
            mw = (low + high * np.e ** (log_fator * x)) / (1 + np.e ** (log_fator * x))  # g/cm3
        elif constraint == 'mandatory':
            mw = x
        else:
            raise ValueError("Please choose right boundary constraint(mandatory, logarithmic)!")
        # ------model_value
        model_value = np.dot((mw - mwapr).T, (mw - mwapr))
        return model_value

    def model_Smoothness(self, x, mwapr, low, high, constraint, log_fator):
        '''
        Damping (0th order Tikhonov) regularization.
        Imposes the minimum norm of the parameter vector.
        '''
        # ----convert x to mw
        if constraint == 'logarithmic':
            mw = (low + high * np.e ** (log_fator * x)) / (1 + np.e ** (log_fator * x))  # g/cm3
        elif constraint == 'mandatory':
            mw = x
        else:
            raise ValueError("Please choose right boundary constraint(mandatory, logarithmic)!")
        # The finite difference matrix
        R3d = self.fd3d(self.mshape)
        # ------model_value
        model_value = np.dot((R3d @ (mw - mwapr)).T, (R3d @ (mw - mwapr)))
        return model_value

    def model_TV(self, x, mwapr, low, high, constraint, log_fator, beta):
        '''
        模型项,自适应正则化因子才用得到
        minimum support stabilizing (MS) function
        模型项:最小支撑稳定泛函
        '''
        # ----convert x to mw
        if constraint == 'logarithmic':
            mw = (low + high * np.e ** (log_fator * x)) / (1 + np.e ** (log_fator * x))  # g/cm3
        elif constraint == 'mandatory':
            mw = x
        else:
            raise ValueError("Please choose right boundary constraint(mandatory, logarithmic)!")
        # The finite difference matrix
        R3d = self.fd3d(self.mshape)
        tmp_value1 = R3d @ (mw - mwapr)
        # ------model_value
        tmp_value2 = np.sqrt(tmp_value1 ** 2 + beta)
        model_value = np.sum(tmp_value2)
        return model_value

    def data_all(self, mw):
        '''
        # 数据项的value和gradient
        '''
        # wavelet or not
        if self.wavelet == '1D':
            dpre = cp1D.modelcompressor(mw, self.Awcp)
        if self.wavelet == '3D':
            dpre = cp3D.modelcompressor(mw, self.Awcp, self.mshape)
        if not self.wavelet:
            dpre = np.dot(self.Aw, mw)
        # 有固定不参与反演的网格
        if self.fixed:
            dinv = dpre + self.grav_fix
        else:
            dinv = dpre
        # ---减去均值 
        # ------数据项
        data_value = np.linalg.norm((dinv-np.mean(dinv)) - (self.dobs-np.mean(self.dobs))) ** 2
        # ------数据项的导数
        data_gradient = 2 * np.dot(self.Aw.T, (dinv-np.mean(dinv)) - (self.dobs-np.mean(self.dobs)))

        '''
        # ---不减去均值
        # ---数据项:
        data_value = np.linalg.norm(dpre - self.dobs) ** 2
        # ------数据项的导数
        data_gradient = 2 * np.dot(self.Aw.T, (dpre - self.dobs))
        '''
        return dpre, data_value, data_gradient

    def model_MS_all(self, mw, mwapr, beta):
        '''
        最小支撑稳定泛函 minimum support stabilizing (MS) function
        *****累加
        '''
        # ------model_value
        # mw^2
        mwSquare = (mw - mwapr) ** 2
        # 分子numerator; 分母denominator
        numerator_value = self.WmSquare @ mwSquare
        denominator_value = mwSquare + beta
        model_value = np.sum(numerator_value / denominator_value)  # 求和
        # ---------model_gradient
        # 分子numerator; 分母denominator
        numerator_gradient = 2 * beta * self.WmSquare @ (mw - mwapr)
        denominator_gradient = (mwSquare + beta)**2
        model_gradient = numerator_gradient / denominator_gradient
        return model_value, model_gradient

    def model_MS1_all(self, mw, mwapr, beta):
        '''
        最小支撑稳定泛函 minimum support stabilizing (MS) function
        *****矩阵形式
        '''
        # ------model_value
        # 分子numerator; 分母denominator
        numerator_value = self.Wm @ (mw - mwapr)
        denominator_value = np.sqrt((mw - mwapr) ** 2 + beta)
        unit_value = numerator_value / denominator_value
        model_value = np.dot(unit_value, unit_value)
        # ---------model_gradient
        # 分子numerator; 分母denominator
        numerator_gradient = 2 * beta * self.Wm.T @ self.Wm @ (mw - mwapr)
        denominator_gradient = ((mw - mwapr) ** 2 + beta)**2
        model_gradient = numerator_gradient / denominator_gradient
        return model_value, model_gradient

    def model_MStry_all(self, mw, mwapr, beta):
        '''
        最小支撑稳定泛函 minimum support stabilizing (MS) function
        一种所有m均使用Wm加权的尝试
        *****矩阵形式
        '''
        # ------model_value
        # 分子numerator; 分母denominator
        numerator_value = mw - mwapr
        denominator_value = np.sqrt((mw - mwapr) ** 2 + beta)
        unit_value = numerator_value / denominator_value
        model_value = np.dot(unit_value, unit_value)
        # ---------model_gradient
        # 分子numerator; 分母denominator
        numerator_gradient = 2 * beta * (mw - mwapr)
        denominator_gradient = ((mw - mwapr) ** 2 + beta)**2
        model_gradient = numerator_gradient / denominator_gradient
        return model_value, model_gradient

    def model_Damping_all(self, mw, mwapr):
        '''
        Damping (0th order Tikhonov) regularization.
        Imposes the minimum norm of the parameter vector.
        '''
        # ------model_value
        model_value = np.dot((mw - mwapr).T, (mw - mwapr))
        # ---------model_gradient
        model_gradient = 2 * (mw - mwapr)
        return model_value, model_gradient

    def model_Smoothness_all(self, mw, mwapr):
        '''Smoothness (1st order Tikhonov) regularization.
        Imposes that adjacent parameters have values close to each other.
        '''
        # The finite difference matrix
        R3d = self.fd3d(self.mshape)
        # ------model_value
        model_value = np.dot((R3d @ (mw - mwapr)).T, (R3d @ (mw - mwapr)))
        # ---------model_gradient
        model_gradient = 2 * R3d.T @ R3d @ (mw - mwapr)
        return model_value, model_gradient

    def model_TV_all(self, mw, mwapr, beta):
        '''Total variation regularization.
        Imposes that adjacent parameters have a few sharp transitions.
        '''
        # The finite difference matrix
        R3d = self.fd3d(self.mshape)
        tmp_value1 = R3d @ (mw - mwapr)
        # ------model_value
        tmp_value2 = np.sqrt(tmp_value1**2 + beta)
        model_value = np.sum(tmp_value2)
        # ---------model_gradient
        model_gradient = R3d.T @ (tmp_value1 / tmp_value2)
        return model_value, model_gradient

    def misfit_and_grad(self, x, mwapr, low, high, constraint, log_fator, alpha, regulization='Damping', beta=0.01):
        """
        compute misfit function and gradient
        x:
        mwapr: 经过wm加权的先验模型
        """
        # ----convert 3: x to mw
        if constraint == 'logarithmic':
            mw = (low + high * np.e ** (log_fator * x)) / (1 + np.e ** (log_fator * x))  # g/cm3
        elif constraint == 'mandatory':
            mw = x
        else:
            raise ValueError("Please choose right boundary constraint(mandatory, logarithmic)!")
        # data + model
        dpre, data_value, data_gradient = self.data_all(mw)
        if regulization == "MS":
            model_value, model_gradient = self.model_MS_all(mw, mwapr, beta)
        elif regulization == "Damping":
            model_value, model_gradient = self.model_Damping_all(mw, mwapr)
        elif regulization == "Smoothness":
            model_value, model_gradient = self.model_Smoothness_all(mw, mwapr)
        elif regulization == "TV":
            model_value, model_gradient = self.model_TV_all(mw, mwapr, beta)
        else:
            raise ValueError("Please choose regularization from 'MS','Damping', 'Smoothness', 'TV'.")
        # -----仅数据项（加权）
        # model_value, model_gradient = 0, 0
        # misfit = data_value
        # grad = data_gradient
        # ----数据项（加权）+ alpha * 模型项
        misfit = data_value + alpha * model_value
        grad = data_gradient + alpha * model_gradient

        return misfit, grad, dpre, data_value, model_value

class JointModule():
    def __init__(self, dobs_gz, dobs_tf,  mrange, mspacing, obsurface,
                 mratio=1, coordinate="cartesian", njobs=1, mangle=(90, 0),
                 wavelet=False, **kwargs):
        '''
        # dobs_gz, dobs_tf 观测数据，做数据加权
        # mrange = (west, east, south, north, top, bottom)也即（xmin, xmax, ymin, ymax, zmin, zmax）
        # mspacing = (dr, dlat, dlon)也即(dz, dy, dx)
        # obsurface = [lon, lat, height] 也即 [xobs, yobs, height]
        # mratio=1,模型深度方向自适应剖分
        # mangle= (inc, dec) 用于磁场计算
        # coordinate="prism" or "spherical"
        # field="gravity"or"magnetic"
        # wavelet = False, 1D, 3D 三个选项
        # if use topography, z up is positive
        # mtopo = (x, y, topography)
        '''
        # observation data
        self.dobs_gz = dobs_gz
        self.dobs_tf = dobs_tf
        # model parameters
        self.mrange = mrange
        self.mspacing = mspacing
        self.mratio = mratio
        # observation surface parameters
        self.lonobs = obsurface[0]
        self.latobs = obsurface[1]
        self.heightobs = obsurface[2]
        # inc, dec of magnetic
        self.inc = mangle[0]
        self.dec = mangle[1]
        # multiple cores
        self.njobs = njobs
        # topography
        self.topocarve = False
        # wavelet
        self.wavelet = wavelet

        if coordinate == "spherical":
            # tesseroid && density
            print("Joint inversion in {} coordinate.".format(coordinate))
            mesh = mesher.TesseroidMesh(self.mrange, self.mspacing, self.mratio)
            # topography
            for key, value in kwargs.items():
                self.topocarve = True
                mtopo = value
                mask = mesh.carvetopo(mtopo[0], mtopo[1], mtopo[2])
                self.mask = mask
            mesh.addprop('density', np.zeros(mesh.size))
            self.mesh = mesh
            _, kernel_gz = tesseroid.gz(self.lonobs, self.latobs, self.heightobs, self.mesh, njobs=self.njobs)

        elif coordinate == "cartesian":
            print("Joint inversion in {} coordinate.".format(coordinate))
            mesh = mesher.PrismMesh(self.mrange, self.mspacing, self.mratio)
            for key, value in kwargs.items():
                self.topocarve = True
                mtopo = value
                mask = mesh.carvetopo(mtopo[0], mtopo[1], mtopo[2])
                self.mask = mask

            # 同样的网格剖分，赋予不同的属性值
            meshrho = mesh.copy()
            meshmag = mesh.copy()
            # 赋值rho
            meshrho.addprop('density', np.zeros(mesh.size))
            self.meshrho = meshrho
            _, kernel_gz = prism.gz(self.lonobs, self.latobs, self.heightobs,
                                         self.meshrho, njobs=self.njobs)
            # 赋值mag
            meshmag.addprop('magnetization', utils.ang2vec(np.zeros(mesh.size), self.inc, self.dec))
            self.meshmag = meshmag
            _, kernel_tf = prism.tf(self.lonobs, self.latobs, self.heightobs,
                                         self.meshmag, self.inc, self.dec, njobs=self.njobs)
        else:
            raise ValueError("Please choose coordinate from(cartesian, spherical)!")

        # 各方向剖分的网格单元个数 self.mshape = [nz, ny, nx]
        self.mshape = mesh.shape
        # 各方向网格单元坐标
        self.mxs = mesh.get_xs()
        self.mys = mesh.get_ys()
        self.mzs = mesh.get_zs()
        # 分别定义2个核矩阵
        self.kernel_gz = kernel_gz
        self.kernel_tf = kernel_tf
        # weight kernel
        # 将两个kernel变成分块矩阵放在对角线位置
        kernel = np.block([
            [kernel_gz, np.zeros((kernel_gz.shape[0], kernel_tf.shape[1]))],
            [np.zeros((kernel_tf.shape[0], kernel_gz.shape[1])), kernel_tf]
        ])
        self.A = kernel  # 定义核矩阵（两个核矩阵合并）
        self.weightKDM()  # calculate Wb & Wm
        # 使用小波压缩，压缩核矩阵Aw,成为Awcp
        if wavelet == '1D':
            # 带地形必须使用1D
            print("Using {} wavelet to compress kernel.".format(wavelet))
            Awcp = cp1D.kernelcompressor(self.Aw)
            self.Awcp = Awcp
        if wavelet == '3D':
            print("Using {} wavelet to compress kernel.".format(wavelet))
            Awcp = cp3D.kernelcompressor(self.Aw, self.mshape)
            self.Awcp = Awcp

        # 作图
        # self.plotmatrix()
        # self.plotdataw()

    def plotmatrix(self):
        # ------作图,画出几个矩阵
        ''''''
        plt.figure()
        plt.title("Wm")
        plt.contourf(self.Wm.todense(), cmap=plt.cm.jet)
        plt.colorbar()
        plt.savefig('picture/matrix of Wm.png')
        plt.show()

        plt.figure()
        plt.title("WmInv")
        plt.contourf(self.WmInv.todense(), cmap=plt.cm.jet)
        plt.colorbar()
        plt.savefig('picture/matrix of WmInv.png')
        plt.show()

        plt.figure()
        plt.title("Wmsquare")
        plt.contourf(self.WmSquare.todense(), cmap=plt.cm.jet)
        plt.colorbar()
        plt.savefig('picture/matrix of WmSquare.png')
        plt.show()

    def plotdataw(self):
        # ------作图,画出核矩阵和数据
        plt.figure()
        plt.suptitle("kernel and data")
        plt.subplot(221)
        plt.title("Aw")
        plt.contourf(self.Aw, cmap=plt.cm.jet)
        plt.colorbar()
        plt.subplot(222)
        plt.title("Wb")
        plt.contourf(self.Wb.todense(), cmap=plt.cm.jet)
        plt.colorbar()
        plt.subplot(223)
        plt.title("dobs")
        plt.plot(self.dobs)
        plt.subplot(224)
        plt.title("dobsw")
        plt.plot(self.dobsw)

        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.6, hspace=0.5)
        plt.savefig('picture/kernel and data after weight.png')
        plt.show()

    def weightKDM(self):
        """
        Weight kernel、Data、Model
        对核矩阵、数据项和模型项加权
        Aw = Wb*A*WmInv
        mw = Wm*m
        dw = Wb*d
        """
        # ---------计算Wm（M*M维）-------
        # 计算ADiag,是一个M维向量
        # 逐个元素计算，速度更快
        ADiagSquare = np.zeros(self.A.shape[1])
        for i in range(self.A.shape[1]):
            ADiagSquare[i] = 0
            for j in range(self.A.shape[0]):
                ADiagSquare[i] += self.A[j, i] ** 2
        ADiag = np.sqrt(ADiagSquare)
        # 计算对角元素的倒数
        for i in range(ADiag.shape[0]):
            if ADiag[i] == 0:
                ADiagInv = 0
            else:
                ADiagInv = 1.0 / ADiag
        # 计算对角元素的平方
        ADiagSquare = ADiag * ADiag
        # 计算加权矩阵,构造稀疏矩阵存储加权矩阵
        row = np.arange(0, self.A.shape[1])  # 对角线坐标，row=col
        Wm = coo_matrix((ADiag, (row, row))).tocsr()  # 将对角元素放到矩阵对角线上，得到加权矩阵Wm
        WmInv = coo_matrix((ADiagInv, (row, row))).tocsr()  # 加权矩阵Wm的逆矩阵
        WmSquare = coo_matrix((ADiagSquare, (row, row))).tocsr()  # 加权矩阵Wm的平方
        # ---------计算Wb（D*D维）-------
        # method1---不做数据平衡
        # std_gz = 1
        # std_tf = 1
        # method2----两个数据的标准差
        # std_gz = np.std(self.dobs_gz)
        # std_tf = np.std(self.dobs_tf)
        # method3----两个核矩阵的标准差
        std_gz = np.std(self.kernel_gz)
        std_tf = np.std(self.kernel_tf)
        # 以数据gz为标准,计算Wb的对角元素
        WbDiag_gz = np.ones_like(self.dobs_gz)
        WbDiag_tf = np.ones_like(self.dobs_tf) * (std_gz / std_tf)
        # # 以数据tf为标准,计算Wb的对角元素
        # WbDiag_tf = np.ones_like(self.dobs_tf)
        # WbDiag_gz = np.ones_like(self.dobs_tf) * (std_tf / std_gz)
        # 拼接得到Wb的对角元素
        WbDiag = np.append(WbDiag_gz, WbDiag_tf)
        # 构造稀疏矩阵存储加权矩阵Wb
        row = np.arange(0, self.A.shape[0])  # 对角线坐标，row=col
        Wb = coo_matrix((WbDiag, (row, row))).tocsr()  # 将对角元素放到矩阵对角线上，得到加权矩阵Wb
        # 加权核矩阵和加权观测数据
        Aw = Wb @ self.A @ WmInv
        dobs = np.append(self.dobs_gz, self.dobs_tf)
        dobsw = Wb @ dobs
        # 定义
        self.Aw = Aw
        self.dobs = dobs
        self.dobsw = dobsw
        self.Wm = Wm
        self.WmInv = WmInv
        self.WmSquare = WmSquare
        self.Wb = Wb

    def forward(self, model):
        '''
        # 不加权的kernel(即A)和不加权的密度模型参数正演
        # 最终的平均模型或者最小误差模型正演计算使用；仅一次
        '''
        dpre = np.dot(self.A, model)
        return dpre

    def fd3djoint(self, shape):
        """
        Produce a 3D finite difference matrix.

        Parameters:

        * shape : tuple = (nz, ny, nx)
            The shape of the parameter grid. Number of parameters in the z, y and x
            dimensions.

        Returns:

        * fd : sparse CSR matrix
            The finite difference matrix
        Examples:

        >>> fd3d((2, 2, 2)).todense()
        matrix([[ 1 -1  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                 [ 0  0  1 -1  0  0  0  0  0  0  0  0  0  0  0  0]
                 [ 1  0 -1  0  0  0  0  0  0  0  0  0  0  0  0  0]
                 [ 0  1  0 -1  0  0  0  0  0  0  0  0  0  0  0  0]
                 [ 0  0  0  0  1 -1  0  0  0  0  0  0  0  0  0  0]
                 [ 0  0  0  0  0  0  1 -1  0  0  0  0  0  0  0  0]
                 [ 0  0  0  0  1  0 -1  0  0  0  0  0  0  0  0  0]
                 [ 0  0  0  0  0  1  0 -1  0  0  0  0  0  0  0  0]
                 [ 1  0  0  0 -1  0  0  0  0  0  0  0  0  0  0  0]
                 [ 0  1  0  0  0 -1  0  0  0  0  0  0  0  0  0  0]
                 [ 0  0  1  0  0  0 -1  0  0  0  0  0  0  0  0  0]
                 [ 0  0  0  1  0  0  0 -1  0  0  0  0  0  0  0  0]
                 [ 0  0  0  0  0  0  0  0  1 -1  0  0  0  0  0  0]
                 [ 0  0  0  0  0  0  0  0  0  0  1 -1  0  0  0  0]
                 [ 0  0  0  0  0  0  0  0  1  0 -1  0  0  0  0  0]
                 [ 0  0  0  0  0  0  0  0  0  1  0 -1  0  0  0  0]
                 [ 0  0  0  0  0  0  0  0  0  0  0  0  1 -1  0  0]
                 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  1 -1]
                 [ 0  0  0  0  0  0  0  0  0  0  0  0  1  0 -1  0]
                 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  1  0 -1]
                 [ 0  0  0  0  0  0  0  0  1  0  0  0 -1  0  0  0]
                 [ 0  0  0  0  0  0  0  0  0  1  0  0  0 -1  0  0]
                 [ 0  0  0  0  0  0  0  0  0  0  1  0  0  0 -1  0]
                 [ 0  0  0  0  0  0  0  0  0  0  0  1  0  0  0 -1]])
        >>> fd3d((3, 2, 2)).todense()
        matrix([[ 1 -1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                 [ 0  0  1 -1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                 [ 1  0 -1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                 [ 0  1  0 -1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                 [ 0  0  0  0  1 -1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                 [ 0  0  0  0  0  0  1 -1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                 [ 0  0  0  0  1  0 -1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                 [ 0  0  0  0  0  1  0 -1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                 [ 0  0  0  0  0  0  0  0  1 -1  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                 [ 0  0  0  0  0  0  0  0  0  0  1 -1  0  0  0  0  0  0  0  0  0  0  0  0]
                 [ 0  0  0  0  0  0  0  0  1  0 -1  0  0  0  0  0  0  0  0  0  0  0  0  0]
                 [ 0  0  0  0  0  0  0  0  0  1  0 -1  0  0  0  0  0  0  0  0  0  0  0  0]
                 [ 1  0  0  0 -1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                 [ 0  1  0  0  0 -1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                 [ 0  0  1  0  0  0 -1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                 [ 0  0  0  1  0  0  0 -1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                 [ 0  0  0  0  1  0  0  0 -1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                 [ 0  0  0  0  0  1  0  0  0 -1  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                 [ 0  0  0  0  0  0  1  0  0  0 -1  0  0  0  0  0  0  0  0  0  0  0  0  0]
                 [ 0  0  0  0  0  0  0  1  0  0  0 -1  0  0  0  0  0  0  0  0  0  0  0  0]
                 [ 0  0  0  0  0  0  0  0  0  0  0  0  1 -1  0  0  0  0  0  0  0  0  0  0]
                 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  1 -1  0  0  0  0  0  0  0  0]
                 [ 0  0  0  0  0  0  0  0  0  0  0  0  1  0 -1  0  0  0  0  0  0  0  0  0]
                 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  1  0 -1  0  0  0  0  0  0  0  0]
                 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1 -1  0  0  0  0  0  0]
                 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1 -1  0  0  0  0]
                 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0 -1  0  0  0  0  0]
                 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0 -1  0  0  0  0]
                 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1 -1  0  0]
                 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1 -1]
                 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0 -1  0]
                 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0 -1]
                 [ 0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0 -1  0  0  0  0  0  0  0]
                 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0 -1  0  0  0  0  0  0]
                 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0 -1  0  0  0  0  0]
                 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0 -1  0  0  0  0]
                 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0 -1  0  0  0]
                 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0 -1  0  0]
                 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0 -1  0]
                 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0 -1]])

        """
        nz, ny, nx = shape
        nderivs = ((nx - 1) * ny + (ny - 1) * nx) * nz + nx * ny * (nz - 1)
        I, J, V = [], [], []

        # 先写出所有单层
        for k in range(nz):
            deriv = 0
            param = 0
            # 某一层
            # nx方向
            for i in range(ny):
                for j in range(nx - 1):
                    I.extend([((nx - 1) * ny + (ny - 1) * nx) * k + deriv,
                              ((nx - 1) * ny + (ny - 1) * nx) * k + deriv,
                              nderivs + ((nx - 1) * ny + (ny - 1) * nx) * k + deriv,
                              nderivs + ((nx - 1) * ny + (ny - 1) * nx) * k + deriv])
                    J.extend([nx * ny * k + param,
                              nx * ny * k + param + 1,
                              nx * ny * nz + nx * ny * k + param,
                              nx * ny * nz + nx * ny * k + param + 1])
                    V.extend([1, -1, 1, -1])
                    deriv += 1
                    param += 1
                param += 1

            # ny方向
            param = 0
            for i in range(ny - 1):
                for j in range(nx):
                    I.extend([((nx - 1) * ny + (ny - 1) * nx) * k + deriv,
                              ((nx - 1) * ny + (ny - 1) * nx) * k + deriv,
                              nderivs + ((nx - 1) * ny + (ny - 1) * nx) * k + deriv,
                              nderivs + ((nx - 1) * ny + (ny - 1) * nx) * k + deriv])
                    J.extend([nx * ny * k + param,
                              nx * ny * k + param + nx,
                              nx * ny * nz + nx * ny * k + param,
                              nx * ny * nz + nx * ny * k + param + nx])
                    V.extend([1, -1, 1, -1])
                    deriv += 1
                    param += 1

        # 再写出两层之间
        front = ((nx - 1) * ny + (ny - 1) * nx) * nz
        for k in range(nz - 1):
            # nz方向
            deriv = 0
            param = 0
            for i in range(ny):
                for j in range(nx):
                    I.extend([front + nx * ny * k + deriv,
                              front + nx * ny * k + deriv,
                              nderivs + front + nx * ny * k + deriv,
                              nderivs + front + nx * ny * k + deriv])
                    J.extend([nx * ny * k + param,
                              nx * ny * k + param + nx * ny,
                              nx * ny * nz + nx * ny * k + param,
                              nx * ny * nz + nx * ny * k + param + nx * ny])
                    V.extend([1, -1, 1, -1])
                    deriv += 1
                    param += 1

        return coo_matrix((V, (I, J)), (2 * nderivs, 2 * nx * ny * nz)).tocsr()

    def fd3dxjoint(self, shape):
        """
                Produce a 3D finite difference matrix in dx direction.

                Parameters:

                * shape : tuple = (nz, ny, nx)
                    The shape of the parameter grid. Number of parameters in the z, y and x
                    dimensions.

                Returns:

                * fd : sparse CSR matrix
                    The finite difference matrix
                Examples:
                >>> fd3dx((2, 2, 2)).todense()
                matrix[[[ 1 -1  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  1 -1  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  1 -1  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  1 -1  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  1 -1  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  1 -1  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  1 -1  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  1 -1]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]]]
                >>> fd3dx((3, 2, 2)).todense()
                matrix[[[ 1 -1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  1 -1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  1 -1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  1 -1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  1 -1  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  1 -1  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  1 -1  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  1 -1  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1 -1  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1 -1  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1 -1  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1 -1]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]]]
        """
        nz, ny, nx = shape
        nderivs = ((nx - 1) * ny + (ny - 1) * nx) * nz + nx * ny * (nz - 1)
        I, J, V = [], [], []

        # 先写出所有单层
        for k in range(nz):
            deriv = 0
            param = 0
            # 某一层
            # nx方向
            for i in range(ny):
                for j in range(nx - 1):
                    I.extend([((nx - 1) * ny + (ny - 1) * nx) * k + deriv,
                              ((nx - 1) * ny + (ny - 1) * nx) * k + deriv,
                              nderivs + ((nx - 1) * ny + (ny - 1) * nx) * k + deriv,
                              nderivs + ((nx - 1) * ny + (ny - 1) * nx) * k + deriv])
                    J.extend([nx * ny * k + param,
                              nx * ny * k + param + 1,
                              nx * ny * nz + nx * ny * k + param,
                              nx * ny * nz + nx * ny * k + param + 1])
                    V.extend([1, -1, 1, -1])
                    deriv += 1
                    param += 1
                param += 1

        return coo_matrix((V, (I, J)), (2 * nderivs, 2 * nx * ny * nz)).tocsr()

    def fd3dyjoint(self, shape):
        """
                Produce a 3D finite difference matrix in dy direction.

                Parameters:

                * shape : tuple = (nz, ny, nx)
                    The shape of the parameter grid. Number of parameters in the z, y and x
                    dimensions.

                Returns:

                * fd : sparse CSR matrix
                    The finite difference matrix
                Examples:

                >>> fd3dy((2, 2, 2)).todense()
                matrix([[ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 1  0 -1  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  1  0 -1  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  1  0 -1  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  1  0 -1  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  1  0 -1  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  1  0 -1  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  1  0 -1  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  1  0 -1]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]])
                >>> fd3dy((3, 2, 2)).todense()
                matrix([[ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 1  0 -1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  1  0 -1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  1  0 -1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  1  0 -1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  1  0 -1  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  1  0 -1  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  1  0 -1  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  1  0 -1  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0 -1  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0 -1  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0 -1  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0 -1]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                         [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]])
                """
        nz, ny, nx = shape
        nderivs = ((nx - 1) * ny + (ny - 1) * nx) * nz + nx * ny * (nz - 1)
        I, J, V = [], [], []

        # 先写出所有单层
        for k in range(nz):
            deriv = 0
            param = 0
            # 某一层
            # nx方向
            for i in range(ny):
                for j in range(nx - 1):
                    deriv += 1
                    param += 1
                param += 1

            # ny方向
            param = 0
            for i in range(ny - 1):
                for j in range(nx):
                    I.extend([((nx - 1) * ny + (ny - 1) * nx) * k + deriv,
                              ((nx - 1) * ny + (ny - 1) * nx) * k + deriv,
                              nderivs + ((nx - 1) * ny + (ny - 1) * nx) * k + deriv,
                              nderivs + ((nx - 1) * ny + (ny - 1) * nx) * k + deriv])
                    J.extend([nx * ny * k + param,
                              nx * ny * k + param + nx,
                              nx * ny * nz + nx * ny * k + param,
                              nx * ny * nz + nx * ny * k + param + nx])
                    V.extend([1, -1, 1, -1])
                    deriv += 1
                    param += 1

        return coo_matrix((V, (I, J)), (2 * nderivs, 2 * nx * ny * nz)).tocsr()

    def fd3dzjoint(self, shape):
        """
        Produce a 3D finite difference matrix in dz direction.

        Parameters:

        * shape : tuple = (nz, ny, nx)
            The shape of the parameter grid. Number of parameters in the z, y and x
            dimensions.

        Returns:

        * fd : sparse CSR matrix
            The finite difference matrix
        Examples:

        >>> fd3dz((2, 2, 2)).todense()
        matrix([[ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                 [ 1  0  0  0 -1  0  0  0  0  0  0  0  0  0  0  0]
                 [ 0  1  0  0  0 -1  0  0  0  0  0  0  0  0  0  0]
                 [ 0  0  1  0  0  0 -1  0  0  0  0  0  0  0  0  0]
                 [ 0  0  0  1  0  0  0 -1  0  0  0  0  0  0  0  0]
                 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                 [ 0  0  0  0  0  0  0  0  1  0  0  0 -1  0  0  0]
                 [ 0  0  0  0  0  0  0  0  0  1  0  0  0 -1  0  0]
                 [ 0  0  0  0  0  0  0  0  0  0  1  0  0  0 -1  0]
                 [ 0  0  0  0  0  0  0  0  0  0  0  1  0  0  0 -1]])
        >>> fd3dz((3, 2, 2)).todense()
        matrix([[ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                 [ 1  0  0  0 -1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                 [ 0  1  0  0  0 -1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                 [ 0  0  1  0  0  0 -1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                 [ 0  0  0  1  0  0  0 -1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                 [ 0  0  0  0  1  0  0  0 -1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                 [ 0  0  0  0  0  1  0  0  0 -1  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                 [ 0  0  0  0  0  0  1  0  0  0 -1  0  0  0  0  0  0  0  0  0  0  0  0  0]
                 [ 0  0  0  0  0  0  0  1  0  0  0 -1  0  0  0  0  0  0  0  0  0  0  0  0]
                 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
                 [ 0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0 -1  0  0  0  0  0  0  0]
                 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0 -1  0  0  0  0  0  0]
                 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0 -1  0  0  0  0  0]
                 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0 -1  0  0  0  0]
                 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0 -1  0  0  0]
                 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0 -1  0  0]
                 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0 -1  0]
                 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0 -1]])
        """
        nz, ny, nx = shape
        nderivs = ((nx - 1) * ny + (ny - 1) * nx) * nz + nx * ny * (nz - 1)
        I, J, V = [], [], []

        # 再写出两层之间
        front = ((nx - 1) * ny + (ny - 1) * nx) * nz
        for k in range(nz - 1):
            # nz方向
            deriv = 0
            param = 0
            for i in range(ny):
                for j in range(nx):
                    I.extend([front + nx * ny * k + deriv,
                              front + nx * ny * k + deriv,
                              nderivs + front + nx * ny * k + deriv,
                              nderivs + front + nx * ny * k + deriv])
                    J.extend([nx * ny * k + param,
                              nx * ny * k + param + nx * ny,
                              nx * ny * nz + nx * ny * k + param,
                              nx * ny * nz + nx * ny * k + param + nx * ny])
                    V.extend([1, -1, 1, -1])
                    deriv += 1
                    param += 1

        return coo_matrix((V, (I, J)), (2 * nderivs, 2 * nx * ny * nz)).tocsr()


    def CrossGradient(self):
        pass

    def kernelw(self):
        '''
        采样循环使用；
        使用采样次数sample次
        '''
        return self.Aw, self.WmInv, self.Wm

    def data(self, x, low, high, constraint, log_fator):
        '''
        # 数据项,自适应正则化因子才用得到
        '''
        # # ----convert x to m
        if constraint == 'logarithmic':
            mw = (low + high * np.e ** (log_fator * x)) / (1 + np.e ** (log_fator * x))  # g/cm3
        elif constraint == 'mandatory':
            mw = x
        else:
            raise ValueError("Please choose right boundary constraint(mandatory, logarithmic)!")
        if self.wavelet == '1D':
            dpre = cp1D.modelcompressor(mw, self.Awcp)
        if self.wavelet == '3D':
            dpre = cp3D.modelcompressor(mw, self.Awcp, self.mshape)
        if not self.wavelet:
            dpre = np.dot(self.Aw, mw)
        data_value = np.linalg.norm(dpre - self.dobsw)**2
        return data_value

    def model_MS(self, x, mwapr, low, high, constraint, log_fator, beta):
        '''
        模型项,自适应正则化因子才用得到
        minimum support stabilizing (MS) function
        模型项:最小支撑稳定泛函
        '''
        # ----convert x to mw
        if constraint == 'logarithmic':
            mw = (low + high * np.e ** (log_fator * x)) / (1 + np.e ** (log_fator * x))  # g/cm3
        elif constraint == 'mandatory':
            mw = x
        else:
            raise ValueError("Please choose right boundary constraint(mandatory, logarithmic)!")
        # ------model_value
        # mw^2
        mwSquare = (mw - mwapr) ** 2
        # 分子numerator; 分母denominator
        numerator_value = self.WmSquare @ mwSquare
        denominator_value = mwSquare + beta
        model_value = np.sum(numerator_value / denominator_value)  # 求和
        return model_value

    def model_Damping(self, x, mwapr, low, high, constraint, log_fator):
        '''
        Damping (0th order Tikhonov) regularization.
        Imposes the minimum norm of the parameter vector.
        '''
        # ----convert x to mw
        if constraint == 'logarithmic':
            mw = (low + high * np.e ** (log_fator * x)) / (1 + np.e ** (log_fator * x))  # g/cm3
        elif constraint == 'mandatory':
            mw = x
        else:
            raise ValueError("Please choose right boundary constraint(mandatory, logarithmic)!")
        # ------model_value
        model_value = np.dot((mw - mwapr).T, (mw - mwapr))
        return model_value

    def model_Smoothness(self, x, mwapr, low, high, constraint, log_fator):
        '''
        Damping (0th order Tikhonov) regularization.
        Imposes the minimum norm of the parameter vector.
        '''
        # ----convert x to mw
        if constraint == 'logarithmic':
            mw = (low + high * np.e ** (log_fator * x)) / (1 + np.e ** (log_fator * x))  # g/cm3
        elif constraint == 'mandatory':
            mw = x
        else:
            raise ValueError("Please choose right boundary constraint(mandatory, logarithmic)!")
        # The finite difference matrix
        R3d = self.fd3d(self.mshape)
        # ------model_value
        model_value = np.dot((R3d @ (mw - mwapr)).T, (R3d @ (mw - mwapr)))
        return model_value

    def model_TV(self, x, mwapr, low, high, constraint, log_fator, beta):
        '''
        模型项,自适应正则化因子才用得到
        minimum support stabilizing (MS) function
        模型项:最小支撑稳定泛函
        '''
        # ----convert x to mw
        if constraint == 'logarithmic':
            mw = (low + high * np.e ** (log_fator * x)) / (1 + np.e ** (log_fator * x))  # g/cm3
        elif constraint == 'mandatory':
            mw = x
        else:
            raise ValueError("Please choose right boundary constraint(mandatory, logarithmic)!")
        # The finite difference matrix
        R3d = self.fd3d(self.mshape)
        tmp_value1 = R3d @ (mw - mwapr)
        # ------model_value
        tmp_value2 = np.sqrt(tmp_value1 ** 2 + beta)
        model_value = np.sum(tmp_value2)
        return model_value

    def data_all(self, mw):
        '''
        # 数据项的value和gradient
        '''
        if self.wavelet == '1D':
            dpre = cp1D.modelcompressor(mw, self.Awcp)
        if self.wavelet == '3D':
            dpre = cp3D.modelcompressor(mw, self.Awcp, self.mshape)
        if not self.wavelet:
            dpre = np.dot(self.Aw, mw)
        # ---数据项version1
        data_value = np.linalg.norm(dpre - self.dobsw) ** 2
        # ------数据项的导数
        data_gradient = 2 * np.dot(self.Aw.T, (dpre - self.dobsw))

        return dpre, data_value, data_gradient

    def model_MS_all(self, mw, mwapr, beta):
        '''
        最小支撑稳定泛函 minimum support stabilizing (MS) function
        *****累加
        '''
        # ------model_value
        # mw^2
        mwSquare = (mw - mwapr) ** 2
        # 分子numerator; 分母denominator
        numerator_value = self.WmSquare @ mwSquare
        denominator_value = mwSquare + beta
        model_value = np.sum(numerator_value / denominator_value)  # 求和
        # ---------model_gradient
        # 分子numerator; 分母denominator
        numerator_gradient = 2 * beta * self.WmSquare @ (mw - mwapr)
        denominator_gradient = (mwSquare + beta)**2
        model_gradient = numerator_gradient / denominator_gradient
        return model_value, model_gradient

    def model_MS1_all(self, mw, mwapr, beta):
        '''
        最小支撑稳定泛函 minimum support stabilizing (MS) function
        *****矩阵形式
        '''
        # ------model_value
        # 分子numerator; 分母denominator
        numerator_value = self.Wm @ (mw - mwapr)
        denominator_value = np.sqrt((mw - mwapr) ** 2 + beta)
        unit_value = numerator_value / denominator_value
        model_value = np.dot(unit_value, unit_value)
        # ---------model_gradient
        # 分子numerator; 分母denominator
        numerator_gradient = 2 * beta * self.Wm.T @ self.Wm @ (mw - mwapr)
        denominator_gradient = ((mw - mwapr) ** 2 + beta)**2
        model_gradient = numerator_gradient / denominator_gradient
        return model_value, model_gradient

    def model_MStry_all(self, mw, mwapr, beta):
        '''
        最小支撑稳定泛函 minimum support stabilizing (MS) function
        一种所有m均使用Wm加权的尝试
        *****矩阵形式
        '''
        # ------model_value
        # 分子numerator; 分母denominator
        numerator_value = mw - mwapr
        denominator_value = np.sqrt((mw - mwapr) ** 2 + beta)
        unit_value = numerator_value / denominator_value
        model_value = np.dot(unit_value, unit_value)
        # ---------model_gradient
        # 分子numerator; 分母denominator
        numerator_gradient = 2 * beta * (mw - mwapr)
        denominator_gradient = ((mw - mwapr) ** 2 + beta)**2
        model_gradient = numerator_gradient / denominator_gradient
        return model_value, model_gradient

    def model_Damping_all(self, mw, mwapr):
        '''
        Damping (0th order Tikhonov) regularization.
        Imposes the minimum norm of the parameter vector.
        '''
        # ------model_value
        model_value = np.dot((mw - mwapr).T, (mw - mwapr))
        # ---------model_gradient
        model_gradient = 2 * (mw - mwapr)
        return model_value, model_gradient

    def model_Smoothness_all(self, mw, mwapr):
        '''Smoothness (1st order Tikhonov) regularization.
        Imposes that adjacent parameters have values close to each other.
        '''
        # The finite difference matrix
        R3d = self.fd3d(self.mshape)
        # ------model_value
        model_value = np.dot((R3d @ (mw - mwapr)).T, (R3d @ (mw - mwapr)))
        # ---------model_gradient
        model_gradient = 2 * R3d.T @ R3d @ (mw - mwapr)
        return model_value, model_gradient

    def model_TV_all(self, mw, mwapr, beta):
        '''Total variation regularization.
        Imposes that adjacent parameters have a few sharp transitions.
        '''
        # The finite difference matrix
        R3d = self.fd3d(self.mshape)
        tmp_value1 = R3d @ (mw - mwapr)
        # ------model_value
        tmp_value2 = np.sqrt(tmp_value1**2 + beta)
        model_value = np.sum(tmp_value2)
        # ---------model_gradient
        model_gradient = R3d.T @ (tmp_value1 / tmp_value2)
        return model_value, model_gradient

    def misfit_and_grad(self, x, mwapr, low, high, constraint, log_fator, alpha, regulization='Damping', beta=0.01):
        """
        compute misfit function and gradient
        x:
        mwapr: 经过wm加权的先验模型
        """
        # ----convert 3: x to mw
        if constraint == 'logarithmic':
            mw = (low + high * np.e ** (log_fator * x)) / (1 + np.e ** (log_fator * x))  # g/cm3
        elif constraint == 'mandatory':
            mw = x
        else:
            raise ValueError("Please choose right boundary constraint(mandatory, logarithmic)!")
        # data + model
        dpre, data_value, data_gradient = self.data_all(mw)
        if regulization == "MS":
            model_value, model_gradient = self.model_MS_all(mw, mwapr, beta)
        elif regulization == "MS1":
            model_value, model_gradient = self.model_MS1_all(mw, mwapr, beta)
        elif regulization == "MStry":
            model_value, model_gradient = self.model_MStry_all(mw, mwapr, beta)
        elif regulization == "Damping":
            model_value, model_gradient = self.model_Damping_all(mw, mwapr)
        elif regulization == "Smoothness":
            model_value, model_gradient = self.model_Smoothness_all(mw, mwapr)
        elif regulization == "TV":
            model_value, model_gradient = self.model_TV_all(mw, mwapr, beta)
        else:
            raise ValueError("Please choose regularization from 'MS','Damping', 'Smoothness', 'TV'.")
        # -----仅数据项（加权）
        # model_value, model_gradient = 0, 0
        # misfit = data_value
        # grad = data_gradient
        # ----数据项（加权）+ alpha * 模型项
        misfit = data_value + alpha * model_value
        grad = data_gradient + alpha * model_gradient

        return misfit, grad, dpre, data_value, model_value
