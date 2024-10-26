# 整理异常数据为反演所需格式（经度、纬度、观测面高度、异常值）
import numpy as np

# load gravity data
with open('data/gravity_12d05d.dat') as f:
    lon, lat, grav = np.loadtxt(f, usecols=[0, 1, 2], unpack=True)

# load Observation surface height data
with open('data/gravobs_12d05d.dat') as f:
    gravobs = np.loadtxt(f, usecols=[2], unpack=True)

# Output the gravity inversion data
with open('data/gravinv_12d05d.dat', 'w') as f:
    np.savetxt(f, np.c_[lon, lat, gravobs, grav], fmt='%.8f', delimiter=' ')