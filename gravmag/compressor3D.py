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


def kernelcompressor(Kernel_Grv, mshape):
    # inputs are
    # Kernel_Grv: original kernel matrix that is going to be compressed;2D排列
    # mshape=(CZ, CY, CX), CX, CY, CZ are number of prisms in x , y , and z axis.

    wname = 'db4'  # name of wavelet
    wv = pywt.Wavelet(wname)
    Nlevel = 2  # level of wavelet comprerssion
    thrg = 0.001  # thresholding value
    Wmode = 'periodization'  # mode of wavelet compression
    CZ, CY, CX = mshape[0], mshape[1], mshape[2]
    Ndatapoints = Kernel_Grv.shape[0] #number of data point.
    Gkernel = []  # will contain wavelet coefficients

    for irow in np.arange(Ndatapoints):
        Kernelsplit = Kernel_Grv[irow, :].copy()  # take one row of kernel
        Gi = Kernelsplit.reshape((CZ, CY, CX))  # reshape kernel to 3D array
        Gi_coeff = pywt.wavedecn(Gi, wv, mode=Wmode, level=Nlevel)  # apply 3D wavelet compression
        Gi_3D_coeff = pywt.coeffs_to_array(Gi_coeff)[0]  # extract wavelet coeff and insert to array
        Gi_3D_coeff[abs(Gi_3D_coeff) < thrg] = 0  # zeroing values under thrg
        Gi_3D_coeff_row = Gi_3D_coeff.reshape((1, -1))  # reshape back to 1D array
        Gkernel.append(Gi_3D_coeff_row)  # put values to the corresponding row

    Gkernel = np.array(Gkernel)  # list to array
    Gkernel2D = Gkernel.reshape(Ndatapoints, -1)  # wavelet 2D kernel
    Gkernelsp = csr_matrix(Gkernel2D)  # sparness

    return Gkernelsp


def modelcompressor(DensityModel, Gkernelsp, mshape):
    # inputs are
    # DensityModel: 1D matrix as an input model, prism的排列方式，x先变，y再变，z最后变
    # Gkernelsp: compressed kernel matrix (output of function "kernelcompressor")
    # mshape=(CZ, CY, CX)

    wname = 'db4'  # name of wavelet
    wv = pywt.Wavelet(wname)
    Nlevel = 2  # level of wavelet compression
    Wmode = 'periodization'  # mode of wavelet compression
    # CX, CY, CZ are number of prisms in x , y , and z axis.
    CZ, CY, CX = mshape[0], mshape[1], mshape[2]
    DensityModel_3D = DensityModel.reshape((CZ, CY, CX))
    Model_coeff = pywt.wavedecn(DensityModel_3D, wv, mode=Wmode, level=Nlevel) # apply 3D wavelet compression
    Model_3D_coeff = pywt.coeffs_to_array(Model_coeff)[0] # extract wavelet coeff and insert to array
    Model_3D_coeff_row = Model_3D_coeff.reshape((-1, 1))  # reshape back to 1D array
    # generated data in real domain.
    # Note that the multiplication of two matrices in the wavelet domain gives output in the real domain
    data_g_wave = Gkernelsp @ Model_3D_coeff_row
    data_g = np.squeeze(data_g_wave)  # delete dimension=1

    return data_g
