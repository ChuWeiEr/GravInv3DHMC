# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 23:22:13 2020
@author: emad ghalenoei
revised by ChuWei 2022/5/22
wname: pywt.wavelist(kind='discrete')
Wmode: pywt.Modes.modes
"""
import numpy as np
from scipy.sparse import csr_matrix
import pywt
import warnings

warnings.filterwarnings('ignore')


def kernelcompressor(Kernel_Grv):
    # inputs are
    # Kernel_Grv: original kernel matrix that is going to be compressed;2D排列
    # Nmodelpoints, number of prisms

    wname = 'db4'  # name of wavelet
    wv = pywt.Wavelet(wname)
    Nlevel = 2  # level of wavelet comprerssion
    thrg = 0.001  # thresholding value
    Wmode = 'periodization'  # mode of wavelet compression
    Ndatapoints = Kernel_Grv.shape[0] #number of data point.
    Gkernel = [] # will contain wavelet coefficients

    for irow in np.arange(Ndatapoints):
        Kernelsplit = Kernel_Grv[irow, :].copy()  # take one row of kernel
        Gi_coeff = pywt.wavedec(Kernelsplit, wv, mode=Wmode, level=Nlevel)  # apply 1D wavelet compression
        Gi_1D_coeff = pywt.coeffs_to_array(Gi_coeff)[0]  # extract wavelet coeff and insert to array
        Gi_1D_coeff[abs(Gi_1D_coeff) < thrg] = 0  # zeroing values under thrg
        # Gkernel[irow, :] = Gi_1D_coeff  # put values to the corresponding row
        Gkernel.append(Gi_1D_coeff) # put values to the corresponding row

    Gkernel = np.array(Gkernel)  # list to array
    Gkernel2D = Gkernel.reshape(Ndatapoints, -1)  # wavelet 2D kernel
    Gkernelsp = csr_matrix(Gkernel2D)  # sparness

    return Gkernelsp


def modelcompressor(DensityModel, Gkernelsp):
    # inputs are
    # DensityModel: 1D matrix as an input model, prism的排列方式，x先变，y再变，z最后变
    # Gkernelsp: compressed kernel matrix (output of function "kernelcompressor")

    wname = 'db4'  # name of wavelet
    wv = pywt.Wavelet(wname)
    Nlevel = 2  # level of wavelet compression
    Wmode = 'periodization'  # mode of wavelet compression
    Model_coeff = pywt.wavedec(DensityModel, wv, mode=Wmode, level=Nlevel) # apply 1D wavelet compression
    Model_1D_coeff = pywt.coeffs_to_array(Model_coeff)[0] # extract wavelet coeff and insert to array
    # generated data in real domain.
    # Note that the multiplication of two matrices in the wavelet domain gives output in the real domain
    data_g = Gkernelsp @ Model_1D_coeff

    return data_g
