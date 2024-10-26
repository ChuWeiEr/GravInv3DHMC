'''
Hamiltonian Monte Carlo Sampling
**List of classes**

* :class:`~GravMagInversion3D.inversion.hmc.HamitonianMC`:
        :fun: sample
        Hamiltonian Monte Carlo Sampling using Leapfrog

* :fun: HMCSample

++++++++++
ChuWei 2022.06.30
'''

import os
import psutil

os.environ['OPENBLAS_NUM_THREADS'] = "1"
os.environ['MKL_NUM_THREADS'] = '1'
import numpy as np
import sys
import time
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from datetime import datetime
# Adaptive regularization fator

class HamitonianMC:
    def __init__(self, UserDefinedModel):
        self.invert_Mass = None
        self.model = UserDefinedModel
        self.dobs = np.zeros(2)
        self.boundaries = np.zeros((2, 2))
        self.dt = None
        self.Lrange = [10, 50]

        self.seed = None
        self.myrank = None
        self.save_folder = None

        self.cache = {}

    def _kinetic(self, p):
        """
        Kinetic energy
        """
        K = np.dot(self.invert_Mass @ p, p) * 0.5

        return K

    
    # for adaptive alpha
    def _data(self, x):
        return self.model.data(x, self.low, self.high, self.constraint, self.log_factor)

    def _model_MS(self, x):
        return self.model.model_MS(x, self.aprior_model, self.low, self.high,
                                   self.constraint, self.log_factor, beta=self.beta)
    def _model_Damping(self, x):
        return self.model.model_Damping(x, self.aprior_model, self.low, self.high,
                                        self.constraint, self.log_factor)
    def _model_Smoothness(self, x):
        return self.model.model_Smoothness(x, self.aprior_model, self.low, self.high,
                                        self.constraint, self.log_factor)
    def _model_TV(self, x):
        return self.model.model_TV(x, self.aprior_model, self.low, self.high,
                                        self.constraint, self.log_factor, beta=self.beta)

    # for leapfrog
    def _misfit_and_grad(self, x, alpha):
        """
        compute misfit function and corresponding gradient
        """
        misfit, grad, dsyn, data_misfit, model_misfit = \
            self.model.misfit_and_grad(x, self.aprior_model, self.low, self.high, self.constraint,self.log_factor,
                                       alpha, regulization=self.regularization, beta=self.beta)
        return misfit, grad, dsyn, data_misfit, model_misfit
    
    # for convert
    def _kernelw(self):
        Aw, WmInv, Wm = self.model.kernelw()
        return Aw, WmInv, Wm

    def _leapfrog(self, xcur, dt, L, alpha, fignum):
        """
        leap frog scheme
        """
        # 存下来xnew用于作图，查看采样点轨迹 2210/1330;2714/5411;2765/6820
        im1 = self.im[0]
        im2 = self.im[1]
        xnewList1 = []
        xnewList2 = []
        n = len(xcur)
        pcur = np.random.randn(n) * self.Sigma
        # initialize xnew and pnew
        pnew = pcur * 1.0
        xnew = xcur * 1.0
        xnewList1.append(xnew[im1]) # ****
        xnewList2.append(xnew[im2])  # ****
        # print("pnew", pnew)
        # print("xnew", xnew)
        # compute current Hamiltonian
        K = self._kinetic(pnew)
        U, grad, dsyn, U_data, U_model = self._misfit_and_grad(xnew, alpha)  # 第一次正演
        Hcur = K + U
        # print("kinetic", K)
        # print("potential", U)
        # print("hamilton", Hcur)
        # save current potential and synthetics
        dsyn_new = dsyn.copy()
        Unew = U
        # update
        pnew -= dt * grad * 0.5
        # print("grad", grad[200:250])
        # print('pnew', pnew[200:250])
        for i in range(L):
            xnew += dt * pnew  # update xnew
            xnewList1.append(xnew[im1])  # ****
            xnewList2.append(xnew[im2])  # ****
            if self.constraint == 'mandatory':
                # check boundaries
                xtmp = xnew.copy()
                ptmp = pnew.copy()
                idx1 = xtmp > self.high
                idx2 = xtmp < self.low
                # 只要有一个密度值超出了所给的范围，就需要如下操作
                # while np.sum(np.logical_or(idx1, idx2)) > 0:
                #     xtmp[idx1] = 2 * self.high[idx1] - xtmp[idx1]
                #     ptmp[idx1] = -ptmp[idx1]
                #     xtmp[idx2] = 2 * self.low[idx2] - xtmp[idx2]
                #     ptmp[idx2] = -ptmp[idx2]
                #     idx1 = xtmp > self.high
                #     idx2 = xtmp < self.low
                while np.sum(np.logical_or(idx1, idx2)) > 0:
                    xtmp[idx1] = self.high[idx1]
                    ptmp[idx1] = -ptmp[idx1]
                    xtmp[idx2] = self.low[idx2]
                    ptmp[idx2] = -ptmp[idx2]
                    idx1 = xtmp > self.high
                    idx2 = xtmp < self.low
                # end while
                pnew = ptmp.copy()
                xnew = xtmp.copy()
                # end check boundaries

            # update pnew
            Unew, grad, dsyn_new, Unew_data, Unew_model = self._misfit_and_grad(xnew, alpha) # 第二次正演
            if i < L - 1:
                pnew -= dt * grad
            else:
                pnew -= dt * grad * 0.5
                # print("grad", grad[200:250])
                # print('pnew', pnew[200:250])
        # end for loop
        pnew = -pnew
        # update Hamiltonian
        Knew = self._kinetic(pnew)
        Hnew = Knew + Unew
        # print("kinetic", Knew)
        # print("potential", Unew)
        # print("hamilton", Hnew)
        # accept or not
        AcceptFlag = False
        u = np.random.rand()
        # print("u", u)
        if Hnew < Hcur or u < np.exp(-(Hnew - Hcur)):
            xcur = xnew
            U = Unew
            dsyn = dsyn_new
            AcceptFlag = True
            U_data = Unew_data
            U_model = Unew_model
        # 作图
        if self.plotsamples:
            self._plot_samples(xnewList1, xnewList2, im1, im2, dt, L,  fignum)
        return xcur, U, dsyn, AcceptFlag, U_data, U_model

    def _plot_samples(self, sample1, sample2, im1, im2, dt, L,  fignum):
        plt.figure(figsize=(10, 8))
        plt.suptitle("Samples in one Leapfrog [dt = {}]".format(dt))
        plt.subplot(131)
        plt.title("sample x[{}]".format(im1))
        plt.plot(sample1, color='k', linewidth=1, alpha=0.5)
        plt.scatter(np.arange(0, len(sample1)), sample1, s=6, c='green', alpha=0.5)
        plt.xlabel("steps")
        plt.ylabel("Density")
        plt.xlim([0, len(sample1)])

        plt.subplot(132)
        plt.title("L={}".format(L))
        plt.scatter(sample1[0], sample2[0], s=50, marker='*', c='blue', label='StartPoint')
        plt.scatter(sample1[-1], sample2[-1], s=50, marker='*', c='red', label='EndPoint')
        plt.scatter(sample1, sample2, s=6, c='green', alpha=0.5)
        plt.plot(sample1, sample2, c='k', alpha=0.5)
        plt.legend()
        plt.xlabel("x[{}]".format(im1))
        plt.ylabel(" x[{}]".format(im2))
        #plt.xlim([self.low[0], self.high[0]])
        #plt.ylim([self.low[0], self.high[0]])
        #plt.ylim([-1, 1])
        # plt.xlim([0, 1])
        # plt.ylim([0, 1])

        plt.subplot(133)
        plt.title("L={}".format(L))
        plt.scatter(sample1[0], sample2[0], s=50, marker='*', c='blue', label='StartPoint')
        plt.scatter(sample1[-1], sample2[-1], s=50, marker='*', c='red', label='EndPoint')
        plt.scatter(sample1, sample2, s=6, c='green', alpha=0.5)
        plt.plot(sample1, sample2, c='k', alpha=0.5)
        plt.legend()
        plt.xlabel("x[{}]".format(im1))
        plt.ylabel(" x[{}]".format(im2))
        plt.xlim([self.low[im1], self.high[im1]])
        plt.ylim([self.low[im2], self.high[im2]])
        #plt.ylim([-1, 1])
        # plt.xlim([0, 1])
        # plt.ylim([0, 1])

        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
        plt.savefig('picture/leapfrog-{}.png'.format(fignum))

    def _save_models(self, x, note):
        # save model
        f = open(self.save_folder + "/" + "model" + str(note) + ".dat", "w")
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                f.write("%f\t" % (x[i, j]))
            f.write("\n")
        f.close()

    def _save_misfit(self, misfit):
        # save misfit:total, data, model
        f = open(self.save_folder + "/" + "misfit.dat", "w")
        for i in range(misfit.shape[0]):
            for j in range(misfit.shape[1]):
                f.write("%f\t" % (misfit[i, j]))
            f.write("\n")
        f.close()

    def _save_models_add(self, x):
        # add to
        with open(self.save_folder + "/" + "model"+ ".dat", "a") as f:
            np.savetxt(f, x, fmt='%.8f', delimiter=' ')

    def _save_misfit_add(self, misfit):
        # add to
        with open(self.save_folder + "/" + "misfit"+ ".dat", "a") as f:
            np.savetxt(f, misfit, fmt='%.8f', delimiter=' ')


    def sample(self, nsamples, ndraws, **kwargs):
        # save synthetics and each sample
        if not os.path.exists(self.save_folder):
            os.mkdir(self.save_folder)
        # 文件用于追加，因此删除之前存在的文件
        if os.path.exists(self.save_folder + "/" + "model"+ ".dat"):
            os.remove(self.save_folder + "/" + "model"+ ".dat")
        # set random seed
        np.random.seed(self.seed)
        # ---------model parameters
        _, WmInv, _ = self._kernelw()
        # mw:加权模型(初始以及)
        # x: 优化对象,x=ln[(mw-a)/(b-mw)]
        # m:（最终输出模型）
        # ----init model m0
        mw = self.initial_model
        print("initial mw:", mw)
        print("mw boundaryies:", self.high, self.low)
        # ----convert 2:mw to x
        if self.constraint == 'logarithmic':
            x = (1/self.log_factor) * np.log((mw - self.low)/(self.high - mw))  # g/cm3
            print("Using logarithmic boundary constraint.")
        elif self.constraint == 'mandatory':
            x = mw
            print("Using mandatory boundary constraint.")
        else:
            raise ValueError("Please choose right boundary constraint(mandatory, logarithmic)!")
        # 计算data和model的size
        data_size = self.dobs.shape[0]
        model_size = self.initial_model.shape[0]
        # print("x", x)
        # time.sleep(10)
        # create list to save misift and model
        misfit = np.zeros((1, 7))
        # sample posterior distributions
        m_cache = np.zeros((1, len(x)))
        # ncount:进行采样的次数
        ncount = 0
        # i:采样被成功接受的次数
        i = 0
        # regularization factor
        alpha = self.RegulFactor
        # end of initialize alpha
        while i < ndraws+nsamples:
            # 给出一个随机的步数
            L = np.random.randint(self.Lrange[0], self.Lrange[1] + 1)
            #print("step", L)
            # all sampling are based on x(mw)
            x, U, _, AcceptFlag, U_data, U_model = self._leapfrog(x, self.dt, L, alpha, i)
            # normed data and model
            U_data_normed = U_data/data_size
            U_model_normed = U_model/model_size
            U_normed = U_data_normed + alpha * U_model_normed

            if AcceptFlag:  # accept this new sample
                if i >= ndraws:
                    # self._save_results(x,dsyn,i-ndraws)
                    # save to list
                    misfit[0, 0] = U
                    misfit[0, 1] = U_data
                    misfit[0, 2] = U_model
                    misfit[0, 3] = U_normed
                    misfit[0, 4] = U_data_normed
                    misfit[0, 5] = U_model_normed
                    misfit[0, 6] = alpha
                    # --------Append the result to the file 'misfit.dat'
                    self._save_misfit_add(misfit)
                    # ----convert 3: x to mw
                    if self.constraint == 'logarithmic':
                        mw = (self.low + self.high *
                              np.e ** (self.log_factor * x))/(1 + np.e ** (self.log_factor*x))  # g/cm3
                    elif self.constraint == 'mandatory':
                        mw = x
                    else:
                        raise ValueError("Please choose right boundary constraint(mandatory, logarithmic)!")
                    # ----convert 4: mw to m
                    m = WmInv @ mw
                    # save m
                    m_cache[0, :] = m.copy()
                    # --------Append the result to the file 'model.dat'
                    self._save_models_add(m_cache)

                i += 1
            ncount += 1
            if i > -1:
                msg = "chain {}: {:.2%}, misfit(total, data, alpha, model)=({:.7f},{:.7f},{:.2f},{:.7f}) " \
                      "-- accept ratio {:.2%}\n". \
                    format(self.myrank, i / (ndraws + nsamples), U_normed, U_data_normed, alpha, U_model_normed, i / ncount)
                print(msg)
                #print(u'hmc进程内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
                sys.stdout.flush()

        # end i-loop

        # 保存数据
        # # 若数据过大，保存误差最小的nbest个模型
        # nbests = self.nbest
        # idx = np.argsort(misfit)
        # if nsamples > 10000:
        #     self._save_models(m_cache[idx[:nbests], :], 'nbest{}'.format(nbests))
        
        # 保存misfit
        # self._save_misfit(misfit)
        


def HMCSample(model, nsamples, ndraws, delta, Lrange,
              initial_model, aprior_model, boundaries, constraint, log_factor, dobs,
              adaptiveRegul, RegulRate, RegulFactor, regularization, beta,
              seed, Sigma, nbest=100, myrank=0,save_folder="mychain",
              plotsamples=False, im=[0, 0]):
    """
    HMC sampling function
    """
    chain = HamitonianMC(model)
    chain.myrank = myrank
    chain.save_folder = save_folder + str(myrank)
    chain.seed = seed + myrank
    chain.nbest = nbest
    # boundary
    nt = boundaries.shape[0]
    chain.boundaries = boundaries
    chain.constraint = constraint  # 边界约束方法
    chain.log_factor = log_factor  # 对数约束的系数
    # leapfrog
    chain.Lrange = Lrange
    chain.dt = delta
    chain.Sigma = Sigma
    # regularization
    chain.adaptiveRegul = adaptiveRegul
    chain.RegulRate = RegulRate
    chain.RegulFactor = RegulFactor
    chain.regularization = regularization
    chain.beta = beta
    # invert_Mass需要稀疏存储
    row = np.arange(0, nt)
    invert_Mass_data = np.ones(nt)
    chain.invert_Mass = coo_matrix((invert_Mass_data, (row, row))).tocsr()
    # 将m的boundaries变成mw的boundaries
    _, _, Wm = chain._kernelw()
    chain.low = Wm @ chain.boundaries[:, 0]
    chain.high = Wm @ chain.boundaries[:, 1]
    chain.im = im
    # model ----convert 1: m to mw
    chain.initial_model = Wm @ initial_model
    chain.aprior_model = Wm @ aprior_model
    # data
    chain.dobs = dobs
    # plot samples
    chain.plotsamples = plotsamples

    chain.sample(nsamples, ndraws)


