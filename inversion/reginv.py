"""
regularized inversion
**List of classes**

* :class:`~GravMagInversion3D.inversion.reginv.ConjugateGradient`: regularized inversion using CG

* :class:`~GravMagInversion3D.inversion.reginv.BootStrap`: BootStrap inversion using CG

++++++++++
ChuWei 2022.06.30
"""
import time
import matplotlib.pyplot as plt
from vis import mpl
import numpy as np
from scipy.sparse import coo_matrix
from gravmag import prism, tesseroid
from gravmag import compressor1D as cp1D
from gravmag import compressor3D as cp3D
import mesher, utils

class ConjugateGradient():
    def __init__(self, dobs, mrange, mspacing, obsurface,
                 mratio=1, njobs=1, coordinate="cartesian",
                 field="gravity", mangle=(90, 0), wavelet=False, **kwargs):
        # 观测数据
        self.dobs = dobs
        # model parameters
        self.mrange = mrange
        self.mspacing = mspacing
        self.mratio = mratio
        # observation surface parameters
        self.lonobs = obsurface[0]
        self.latobs = obsurface[1]
        self.heightobs = obsurface[2]
        # 并行
        self.njobs = njobs
        # inc, dec of magnetic
        self.inc = mangle[0]
        self.dec = mangle[1]
        # 其他参数
        self.wavelet = wavelet  # wavelet

        if coordinate == "spherical" and field == "gravity":
            # tesseroid && density
            print("Calculating {} field in {} coordinate.".format(field, coordinate))
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
            mesh = mesher.PrismMesh(self.mrange, self.mspacing, self.mratio)
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
            end = time.time()
            print("End of calculate kernel:%.6f s" % (end - start))
        elif coordinate == "cartesian" and field == "magnetic":
            print("Calculating {} field in {} coordinate.".format(field, coordinate))
            mesh = mesher.PrismMesh(self.mrange, self.mspacing, self.mratio)
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
        self.dsize = kernel.shape[0]
        self.msize = kernel.shape[1]
        # 各方向网格单元坐标
        self.mxs = mesh.get_xs()
        self.mys = mesh.get_ys()
        self.mzs = mesh.get_zs()
        # weight kernel
        self.A = kernel  # define kernel
        self.newkernel()  # weight kernel
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

    def newkernel(self):
        """
        计算Wm加权之后的kernel
        用于反演
        """
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

    def data(self, mw):
        # -----数据项
        if self.wavelet == '1D':
            dpre = cp1D.modelcompressor(mw, self.Awcp)
        if self.wavelet == '3D':
            dpre = cp3D.modelcompressor(mw, self.Awcp, self.mshape)
        if not self.wavelet:
            dpre = np.dot(self.Aw, mw)
        data_value = np.linalg.norm(dpre - self.dobs) ** 2
        return data_value

    def data_gfun(self, mw):
        # ------数据项的导数
        if self.wavelet == '1D':
            dpre = cp1D.modelcompressor(mw, self.Awcp)
        if self.wavelet == '3D':
            dpre = cp3D.modelcompressor(mw, self.Awcp, self.mshape)
        if not self.wavelet:
            dpre = np.dot(self.Aw, mw)
        # ------数据项的导数
        data_gradient = 2 * np.dot(self.Aw.T, (dpre - self.dobs))
        return data_gradient

    def model_MS(self, mw, mwapr, beta):
        '''
        最小支撑稳定泛函 minimum support stabilizing (MS) function
        '''
        # mw^2
        mwSquare = (mw - mwapr) ** 2
        # 分子numerator; 分母denominator
        numerator_value = self.WmSquare @ mwSquare
        denominator_value = mwSquare + beta
        model_value = np.sum(numerator_value / denominator_value)  # 求和
        return model_value

    def model_gfun_MS(self, mw, mwapr, beta):
        '''
        最小支撑稳定泛函 minimum support stabilizing (MS) function 一阶导数
        '''
        # mw^2
        mwSquare = mw * mw
        # 分子numerator; 分母denominator
        numerator_gradient = 2 * beta * self.WmSquare @ (mw - mwapr)
        denominator_gradient = (mwSquare + beta) ** 2
        model_gradient = numerator_gradient / denominator_gradient
        return model_gradient

    def model_Damping(self, mw, mwapr):
        '''
        Damping (0th order Tikhonov) regularization.
        Imposes the minimum norm of the parameter vector.
        '''
        # ------model_value
        model_value = np.dot((mw - mwapr).T, (mw - mwapr))
        return model_value

    def model_gfun_Damping(self, mw, mwapr):
        '''
        Damping (0th order Tikhonov) regularization.
        Imposes the minimum norm of the parameter vector.
        '''
        # ---------model_gradient
        model_gradient = 2 * (mw - mwapr)
        return model_gradient

    def model_Smoothness(self, mw, mwapr):
        '''Smoothness (1st order Tikhonov) regularization.
        Imposes that adjacent parameters have values close to each other.
        '''
        # The finite difference matrix
        R3d = self.fd3d(self.mshape)
        # ------model_value
        model_value = np.dot((R3d @ (mw - mwapr)).T, (R3d @ (mw - mwapr)))
        return model_value

    def model_gfun_Smoothness(self, mw, mwapr):
        '''Smoothness (1st order Tikhonov) regularization.
        Imposes that adjacent parameters have values close to each other.
        '''
        # The finite difference matrix
        R3d = self.fd3d(self.mshape)
        # ---------model_gradient
        model_gradient = 2 * R3d.T @ R3d @ (mw - mwapr)
        return model_gradient

    def model_TV(self, mw, mwapr, beta):
        '''Total variation regularization.
        Imposes that adjacent parameters have a few sharp transitions.
        '''
        # The finite difference matrix
        R3d = self.fd3d(self.mshape)
        tmp_value1 = R3d @ (mw - mwapr)
        # ------model_value
        tmp_value2 = np.sqrt(tmp_value1**2 + beta)
        model_value = np.sum(tmp_value2)
        return model_value

    def model_gfun_TV(self, mw, mwapr, beta):
        '''Total variation regularization.
        Imposes that adjacent parameters have a few sharp transitions.
        '''
        # The finite difference matrix
        R3d = self.fd3d(self.mshape)
        tmp_value1 = R3d @ (mw - mwapr)
        tmp_value2 = np.sqrt(tmp_value1**2 + beta)
        # ---------model_gradient
        model_gradient = R3d.T @ (tmp_value1 / tmp_value2)
        return model_gradient

    def CG(self, initialModel, apriorModel, boundary, regularization='MS', beta=0.01, q=0.9, maxk=100):
        '''
        Args:
            initialModel: 初始模型
            apriorModel: 先验模型
            boundary: 密度约束
            q: 正则化因子的衰减系数
            regularization: 正则化项选择，可以为"MS" "Damping" "Smoothness" "TV"中的一个
            beta: "MS" 与 "TV"用到的参数
            maxk: 最大反演迭代次数
        Returns: model_inv, data_inv, data_misfit, model_misfit, regul_factor
        分别为：3D模型反演结果，数据拟合结果，数据误差，模型误差，正则化因子
        '''
        # -----模型加权
        mw = self.Wm @ initialModel
        mwapr = self.Wm @ apriorModel
        rhomin = boundary[0]
        rhomax = boundary[1]
        # 存误差、正则化因子
        data_misfit = []  # 数据误差
        model_misfit = []  # 模型误差
        regul_factor = []  #正则化因子
        # 开始共轭梯度法的迭代过程
        for k in range(0, maxk):
            # 输出迭代次数
            print("CG iteration: ", k+1)
            # 计算正则化因子
            if k==0:
                alpha = 0
            elif k==1:
                if regularization == "MS":
                    alpha = self.data(mw_new) / self.model_MS(mw_new, mwapr, beta)
                elif regularization == "Damping":
                    alpha = self.data(mw_new) / self.model_Damping(mw_new, mwapr)
                elif regularization == "Smoothness":
                    alpha = self.data(mw_new) / self.model_Smoothness(mw_new, mwapr)
                elif regularization == "TV":
                    alpha = self.data(mw_new) / self.model_TV(mw_new, mwapr, beta)
                else:
                    raise ValueError("Please choose regularization from 'MS','Damping', 'Smoothness', 'TV'.")
            else:
                # d_old - d_new < 0.01 * d_old,证明数据项衰减地很小，也就是数据项拟合地很好；
                # 让模型项权重变小，数据项变大？
                if self.data(mw) - self.data(mw_new) < 0.01 * self.data(mw):
                    alpha = q * alpha
                else:
                    alpha = alpha
            regul_factor.append(alpha)  # 保存正则化因子

            # ----目标函数的一阶导数
            # 第一次迭代与之后不同
            if k==0:
                data_misfit.append(self.data(mw) / self.dsize)  # 保存数据误差
                # I0&I0w
                if regularization == "MS":
                    I = self.data_gfun(mw) + alpha * self.model_gfun_MS(mw, mwapr, beta)
                    model_misfit.append(self.model_MS(mw, mwapr, beta) / self.msize)  # 模型项误差
                elif regularization == "Damping":
                    I = self.data_gfun(mw) + alpha * self.model_gfun_Damping(mw, mwapr)
                    model_misfit.append(self.model_Damping(mw, mwapr) / self.msize)  # 模型项误差
                elif regularization == "Smoothness":
                    I = self.data_gfun(mw) + alpha * self.model_gfun_Smoothness(mw, mwapr)
                    model_misfit.append(self.model_Smoothness(mw, mwapr) / self.msize)  # 模型项误差
                elif regularization == "TV":
                    I = self.data_gfun(mw) + alpha * self.model_gfun_TV(mw, mwapr, beta)
                    model_misfit.append(self.model_TV(mw, mwapr, beta) / self.msize)  # 模型项误差
                else:
                    raise ValueError("Please choose regularization from 'MS','Damping', 'Smoothness', 'TV'.")
                Iw = I
                # 步长
                kstep = np.dot(Iw.T, I)/(np.linalg.norm(self.Aw @ Iw) ** 2 + alpha * np.linalg.norm(Iw)**2)
                # 更新密度矩阵
                mw_new = mw - kstep * Iw
                # 将密度矩阵限制在合理范围内
                mtemp = self.WmInv @ mw_new
                mtemp[mtemp < rhomin] = rhomin
                mtemp[mtemp > rhomax] = rhomax
                mw_new = self.Wm @ mtemp

            if k > 0:
                # 保存上一步的值
                I_old = I
                Iw_old = Iw
                mw = mw_new
                # In
                if regularization == "MS":
                    I = self.data_gfun(mw) + alpha * self.model_gfun_MS(mw, mwapr, beta)
                elif regularization == "Damping":
                    I = self.data_gfun(mw) + alpha * self.model_gfun_Damping(mw, mwapr)
                elif regularization == "Smoothness":
                    I = self.data_gfun(mw) + alpha * self.model_gfun_Smoothness(mw, mwapr)
                elif regularization == "TV":
                    I = self.data_gfun(mw) + alpha * self.model_gfun_TV(mw, mwapr, beta)
                else:
                    raise ValueError("Please choose regularization from 'MS','Damping', 'Smoothness', 'TV'.")
                mu = np.linalg.norm(I)**2/np.linalg.norm(I_old)**2
                # 更新共轭方向
                Iw = I + mu * Iw_old
                # 更新步长
                kstep = np.dot(Iw.T, I)/(np.linalg.norm(self.Aw @ Iw) ** 2 + alpha * np.linalg.norm(Iw)**2)
                # 更新密度矩阵
                mw_new = mw - kstep * Iw
                # 将密度矩阵限制在合理范围内
                mtemp = self.WmInv @ mw_new
                mtemp[mtemp < rhomin] = rhomin
                mtemp[mtemp > rhomax] = rhomax
                mw_new = self.Wm @ mtemp

                # 分别保存并输出数据和模型误差
                data_misfit.append(self.data(mw_new) / self.dsize)
                print("Normed data error:", self.data(mw_new) / self.dsize)  # 数据项误差
                if regularization == "MS":
                    model_misfit.append(self.model_MS(mw_new, mwapr, beta)/self.msize)
                    print("Normed model error:", self.model_MS(mw_new, mwapr, beta)/self.msize)  # 模型项误差
                elif regularization == "Damping":
                    model_misfit.append(self.model_Damping(mw_new, mwapr)/self.msize)
                    print("Normed model error:", self.model_Damping(mw_new, mwapr) / self.msize)  # 模型项误差
                elif regularization == "Smoothness":
                    model_misfit.append(self.model_Smoothness(mw_new, mwapr) / self.msize)
                    print("Normed model error:", self.model_Smoothness(mw_new, mwapr) / self.msize)  # 模型项误差
                elif regularization == "TV":
                    model_misfit.append(self.model_TV(mw_new, mwapr, beta) / self.msize)
                    print("Normed model error:", self.model_TV(mw_new, mwapr, beta) / self.msize)  # 模型项误差
                else:
                    raise ValueError("Please choose regularization from 'MS','Damping', 'Smoothness', 'TV'.")

                # 当数据误差<0.1时代码停止
                if self.data(mw_new) / self.dsize < 0.001:
                    print("Normed data error is {} < 0.001, stop iteration!".format(self.data(mw_new) / self.dsize))
                    break
        # 最终将模型转换真实结果，计算异常拟合数据
        model_inv = self.WmInv @ mw_new
        data_inv = self.A @ model_inv

        return model_inv, data_inv, data_misfit, model_misfit, regul_factor


class BootStrap():
    def __init__(self, mrange, mspacing, obsurface, dobs, boundary,
                 samples=100,beta=0.01, maxk=100, mratio=1, njobs=1,
                 wavelet=False, **kwargs):
        # model parameters
        self.mrange = mrange
        self.mspacing = mspacing
        self.mratio = mratio
        # observation surface parameters
        self.lonobs = obsurface[0]
        self.latobs = obsurface[1]
        self.heightobs = obsurface[2]
        self.boundary = boundary  # 模型参数范围
        # 其他参数
        self.samples = samples  # 对观测数据随机采样次数
        self.njobs = njobs  # 并行
        self.dobs = dobs  # 观测数据
        self.maxk = maxk  # 最大迭代次数
        self.beta = beta  # MS模型项参数
        self.wavelet = wavelet  # wavelet

        print("Calculating gravity field using prism.")
        mesh = mesher.PrismMesh(self.mrange, self.mspacing, self.mratio)
        # topography
        for key, value in kwargs.items():
            mtopo = value
            mask = mesh.carvetopo(mtopo[0], mtopo[1], mtopo[2])
            self.mask = mask
        mesh.addprop('density', np.zeros(mesh.size))
        self.mesh = mesh
        print('Start to calculate kernel')
        start = time.time()
        _, kernel = prism.gz(self.lonobs, self.latobs, self.heightobs, self.mesh, njobs=self.njobs)
        end = time.time()
        print("End of calculate kernel:", end-start)
        # 各方向剖分的网格单元个数 self.mshape = [nz, ny, nx]
        self.mshape = mesh.shape
        self.dsize = kernel.shape[0]
        self.msize = kernel.shape[1]
        # 各方向网格单元坐标
        self.mxs = mesh.get_xs()
        self.mys = mesh.get_ys()
        self.mzs = mesh.get_zs()
        # define and weight kernel
        self.A = kernel  # define kernel

        print('Start to calculate kernel')
        start = time.time()
        self.newkernel()  # 敏感核加权矩阵
        end = time.time()
        print("End of calculate newkernel:", end-start)

        if wavelet == '1D':
            # 带地形必须使用1D
            print("Using {} wavelet to compress kernel.".format(wavelet))
            Awcp = cp1D.kernelcompressor(self.Aw)
            self.Awcp = Awcp
        if wavelet == '3D':
            print("Using {} wavelet to compress kernel.".format(wavelet))
            Awcp = cp3D.kernelcompressor(self.Aw, self.mshape)
            self.Awcp = Awcp

    def newkernel(self):
        """
        计算Wm加权之后的kernel
        用于反演
        """
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
        Aw = self.A @ WmInv  # 计算新的核矩阵
        # 定义
        self.Aw = Aw
        self.Wm = Wm
        self.WmInv = WmInv
        self.WmSquare = WmSquare


    def data(self, mw, Aw, dobs):
        # -----数据项
        if self.wavelet == '1D':
            dpre = cp1D.modelcompressor(mw, self.Awcp)
        if self.wavelet == '3D':
            dpre = cp3D.modelcompressor(mw, self.Awcp, self.mshape)
        if not self.wavelet:
            dpre = np.dot(Aw, mw)
        data_value = np.linalg.norm(dpre - dobs) ** 2
        return data_value

    def model_MS(self, mw):
        # mw^2
        mwSquare = mw * mw
        # 分子reg1; 分母reg2
        reg1 = self.WmSquare @ mwSquare
        reg2 = mwSquare + self.beta ** 2
        model_value = np.sum(reg1 / reg2)  # 求和
        return model_value

    def data_gfun(self, mw, Aw, dobs):
        # ------数据项的导数
        if self.wavelet == '1D':
            dpre = cp1D.modelcompressor(mw, self.Awcp)
        if self.wavelet == '3D':
            dpre = cp3D.modelcompressor(mw, self.Awcp, self.mshape)
        if not self.wavelet:
            dpre = np.dot(Aw, mw)
        # ------数据项的导数
        data_gradient = 2 * np.dot(Aw.T, (dpre - dobs))
        return data_gradient

    def model_gfun_MS(self, mw):
        # mw^2
        mwSquare = mw * mw
        # ---------regul_gradient
        # 分子reg1_gradient; 分母reg2_gradient
        reg1_gradient = 2 * self.WmSquare @ (mw * self.beta ** 2)
        r2 = mwSquare + self.beta ** 2
        reg2_gradient = r2 * r2
        model_gradient = reg1_gradient / reg2_gradient
        return model_gradient

    def CG(self, Aw, dobs, initialModel):
        # -----模型加权
        mw = self.Wm @ initialModel
        rhomin = self.boundary[0]
        rhomax = self.boundary[1]
        # 设置正则化因子的衰减系数
        q = 0.9
        # 存误差、正则化因子
        data_misfit = []  # 数据误差
        model_misfit = []  # 模型误差
        regul_factor = []  #正则化因子

        # 开始共轭梯度法的迭代过程
        for k in range(0, self.maxk):
            # 计算正则化因子
            # alpha = 1
            if k==0:
                alpha = 0
            elif k==1:
                alpha = self.data(mw_new, Aw, dobs)/self.model_MS(mw_new)
            else:
                # d_old - d_new < 0.01 * d_old,证明数据项衰减地很小，也就是数据项拟合地很好；
                # 让模型项权重变小，数据项变大？
                if self.data(mw, Aw, dobs) - self.data(mw_new, Aw, dobs) < 0.01 * self.data(mw, Aw, dobs):
                    alpha = q * alpha
                else:
                    alpha = alpha
            regul_factor.append(alpha)

            # ----目标函数的一阶导数
            # 第一次迭代与之后不同
            if k==0:
                # I0&I0w
                I = self.data_gfun(mw, Aw, dobs) + alpha * self.model_gfun_MS(mw)
                Iw = I
                # 步长
                kstep = np.dot(Iw.T, I)/(np.linalg.norm(Aw @ Iw) ** 2 + alpha * np.linalg.norm(Iw)**2)
                # 更新密度矩阵
                mw_new = mw - kstep * Iw
                # 将密度矩阵限制在合理范围内
                mtemp = self.WmInv @ mw_new
                mtemp[mtemp < rhomin] = rhomin
                mtemp[mtemp > rhomax] = rhomax
                mw_new = self.Wm @ mtemp

            if k > 0:
                # 保存上一步的值
                I_old = I
                Iw_old = Iw
                mw = mw_new
                # In
                I = self.data_gfun(mw, Aw, dobs) + alpha * self.model_gfun_MS(mw)
                mu = np.linalg.norm(I)**2/np.linalg.norm(I_old)**2
                # 更新共轭方向
                Iw = I + mu * Iw_old
                # 更新步长
                kstep = np.dot(Iw.T, I)/(np.linalg.norm(Aw @ Iw) ** 2 + alpha * np.linalg.norm(Iw)**2)
                # 更新密度矩阵
                mw_new = mw - kstep * Iw
                # 将密度矩阵限制在合理范围内
                mtemp = self.WmInv @ mw_new
                mtemp[mtemp < rhomin] = rhomin
                mtemp[mtemp > rhomax] = rhomax
                mw_new = self.Wm @ mtemp

                # 当数据误差<0.1时代码停止
                if self.data(mw_new, Aw, dobs) < 0.1:
                    print("Data error is {} < 0.1, stop iteration!".format(self.data(mw_new, Aw, dobs)))
                    break

                # 保存并输出
                data_misfit.append(self.data(mw_new, Aw, dobs)/self.dsize)
                model_misfit.append(self.model_MS(mw_new)/self.msize)
                print(self.data(mw_new, Aw, dobs)/self.dsize)  # 数据项误差
                print(self.model_MS(mw_new)/self.msize)  # 查看模型项误差

            # 输出迭代次数
            print("CG iteration: ", k)

        model_inv = self.WmInv @ mw_new
        # data_inv = self.A @ model_inv

        return model_inv, data_misfit, model_misfit, regul_factor

    def BSCG(self, initialModel):
        '''
        Bootstrap采样：
        对观测数据做self.samples次的有放回采样，每次采样反演出来一个结果
        Returns:

        '''
        # save all inversion models & misfit & regul
        model_inv_all = np.zeros((self.samples, self.msize))
        data_misfit_all = np.zeros((self.samples, self.maxk-1))
        model_misfit_all = np.zeros((self.samples, self.maxk-1))
        regul_factor_all = np.zeros((self.samples, self.maxk))
        for sample in range(self.samples):
            print("*********Sample {}*********".format(sample+1))
            np.random.seed(sample)
            # 对索引做有放回采样
            index = np.arange(0, self.dsize)
            indexSample = np.random.choice(index, size=self.dsize, replace=True, p=None)
            # 根据索引取出dobs和加权核矩阵Aw
            dobsSample = np.zeros_like(self.dobs)
            AwSample = np.zeros_like(self.Aw)
            for i in range(self.dsize):
                dobsSample[i] = self.dobs[indexSample[i]]
                AwSample[i, :] = self.Aw[indexSample[i], :]

            # 调用CG计算
            model_inv, data_misfit, model_misfit, regul_factor = self.CG(AwSample, dobsSample, initialModel)
            # 保存模型与误差
            model_inv_all[sample, :] = model_inv
            data_misfit_all[sample, :] = data_misfit
            model_misfit_all[sample, :] = model_misfit
            regul_factor_all[sample, :] = regul_factor

        return model_inv_all, data_misfit_all, model_misfit_all, regul_factor_all







