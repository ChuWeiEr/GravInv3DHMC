#### The GravInv3DHMC package
version of 2024-06-30 by ChuWei

##### modelling
Modules for gridding, meshing, visualization and universal utilities:

- mesher: Mesh generation and definition of geometric elements
    - geometry: defines geometric primitives
        - Prism
        - Tesseroid
    - mesh:  Mesh generation and definition of geometric elements
        - PrismRelief
        - PrismMesh
        - TesseroidMesh
        - PrismMeshSegment
        - TesseroidMeshSegment
- vis: Plotting utilities for 2D (using matplotlib) and 3D (using mayavi)
    - mpl
    - myv
- utils: Miscelaneous utilities like mathematical functions, unit conversion, etc
    - kernel2UBC
    - rho2carve
    - carve2rho
- constants: Physical constants and unit conversions

##### forward
Modules for gravity and magnetic forward and wavelet compression
- gravmag
    - prism  
      - potential/gx/gy/gz/gxx/gxy/gxz/gyy/gyz/gzz  
      - tf
    - tesseroid 
      - potential/gx/gy/gz/gxx/gxy/gxz/gyy/gyz/gzz
    - compressor1D: 1-D wavelet compression
    - compressor3D: 3-D wavelet compression

##### inversion
Modules for gravity and magnetic inversion using Conjugate Gradient、BootStrap and Hamiltonian Monte Carlo Sampling
- inversion
    - reginv: regularization inversion method
      - ConjugateGradient  
      - BootStrap
    - potential: potential energy function of hmc
      - GravMagModule
      - JointModule(not completed)
    - hmc: Hamiltonian Monte Carlo Sampling using Leapfrog

##### 软件包使用方法
建议新建一个文件夹用于存放主程序，例如新建example文件夹，在其中编写建模和反演代码，运行后输出数据和图片。

##### reference
https://www.fatiando.org/