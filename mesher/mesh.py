"""
Meshes (collections) of geometric objects.
Meshes behave like lists/arrays of geometric objects (they are iterables).

References
++++++++++

Uieda, L., V. Barbosa, and C. Braitenberg (2016), Tesseroids: Forward-modeling
gravitational fields in spherical coordinates, Geophysics, F41-F48,
doi:10.1190/geo2015-0204.1

ChuWei 2022.06.30
"""
import numpy as np
import scipy.special
import scipy.interpolate
import copy as cp

import utils
from mesher.geometry import Prism, Tesseroid


class PrismRelief(object):
    """
    A 3D model of a relief (topography) using prisms.

    Use to generate:
    * topographic model
    * basin model
    * Moho model
    * etc

    PrismRelief can used as list of prisms. It acts as an iteratior (so you
    can loop over prisms). It also has a ``__getitem__`` method to access
    individual elements in the mesh.
    In practice, PrismRelief should be able to be passed to any function that
    asks for a list of prisms, like :func:`fatiando.gravmag.prism.gz`.

    Parameters:

    * ref : float
        Reference level. Prisms will have:
            * bottom on zref and top on z if z > zref;
            * bottom on z and top on zref otherwise.
    * dims :  tuple = (dy, dx)
        Dimensions of the prisms in the y and x directions
    * nodes : list of lists = [x, y, z]
        Coordinates of the center of the top face of each prism.x, y, and z are
        lists with the x, y and z coordinates on a regular grid.

    """

    def __init__(self, ref, dims, nodes):
        x, y, z = nodes
        if len(x) != len(y) != len(z):
            raise ValueError(
                "nodes has x, y, z coordinate arrays of different lengths")
        self.x, self.y, self.z = x, y, z
        self.size = len(x)
        self.ref = ref
        self.dy, self.dx = dims
        self.props = {}
        # The index of the current prism in an iteration. Needed when mesh is
        # used as an iterator
        self.i = 0

    def __len__(self):
        return self.size

    def __iter__(self):
        self.i = 0
        return self

    def __getitem__(self, index):
        # To walk backwards in the list
        if index < 0:
            index = self.size + index
        xc, yc, zc = self.x[index], self.y[index], self.z[index]
        x1 = xc - 0.5 * self.dx
        x2 = xc + 0.5 * self.dx
        y1 = yc - 0.5 * self.dy
        y2 = yc + 0.5 * self.dy
        if zc <= self.ref:
            z1 = zc
            z2 = self.ref
        else:
            z1 = self.ref
            z2 = zc
        props = dict([p, self.props[p][index]] for p in self.props)
        return Prism(x1, x2, y1, y2, z1, z2, props=props)

    def __next__(self):
        if self.i >= self.size:
            raise StopIteration
        prism = self.__getitem__(self.i)
        self.i += 1
        return prism

    def addprop(self, prop, values):
        """
        Add physical property values to the prisms.

        .. warning:: If the z value of any point in the relief is below the
            reference level, its corresponding prism will have the physical
            property value with oposite sign than was assigned to it.

        Parameters:

        * prop : str
            Name of the physical property.
        * values : list
            List or array with the value of this physical property in each
            prism of the relief.

        """
        def correct(v, i):
            if self.z[i] > self.ref:
                return -v
            return v
        self.props[prop] = [correct(v, i) for i, v in enumerate(values)]

    def copy(self):
        """ Return a deep copy of the current instance."""
        return cp.deepcopy(self)

class PrismMesh(object):
    """
    A 3D self-adaptive regular mesh of right rectangular prisms.

    Prisms are ordered as follows: first layers (z coordinate),
    then EW rows (y) and finaly x coordinate (NS).

    .. note:: Remember that the coordinate system is x->North, y->East and
        z->Down

    Ex: in a mesh with shape ``(3,3,3)`` the 15th element (index 14) has z
    index 1 (second layer), y index 1 (second row), and x index 2 (third
    element in the column).

    :class:`mesher.PrismMesh` can used as list of prisms. It acts
    as an iteratior (so you can loop over prisms). It also has a
    ``__getitem__`` method to access individual elements in the mesh.
    In practice, :class:`mesher.PrismMesh` should be able to be
    passed to any function that asks for a list of prisms, like
    :func:gravmag.prism.gz`.

    To make the mesh incorporate a topography, use
    :meth:`mesher.PrismMesh.carvetopo`

    Parameters:

    * bounds : list = [xmin, xmax, ymin, ymax, zmin, zmax]
        Boundaries of the mesh.
    * spacing : tuple = (dz, dy, dx)
        spacing of prisms in the x, y, and z directions.
    * ratio: dz 的变化率，一般来说向下网格变大，则ratio>=1;默认ratio=1
    * props :  dict
        Physical properties of each prism in the mesh.
        Each key should be the name of a physical property. The corresponding
        value should be a list with the values of that particular property on
        each prism of the mesh.
    """

    celltype = Prism

    def __init__(self, bounds, spacing, ratio=1, props=None):
        dz, dy, dx = spacing
        x1, x2, y1, y2, z1, z2 = bounds
        self.dims = (dx, dy, dz)
        self.ratio = ratio

        # 若不整除，则建模范围xmax, ymax, zmax稍大于给定范围;
        # 使用ratio时，zmax保持原范围
        nx = int(np.ceil((x2 - x1)/dx)) # 向上取整
        ny = int(np.ceil((y2 - y1)/dy))  # 向上取整
        # 求nz
        if ratio==1:
            nz = int(np.ceil((z2 - z1)/dz))
            bounds_big = x1, x1 + nx * dx, y1, y1 + ny * dy, z1, z1 + nz * dz
            print("Uniform grid with new boundaries: {}".format(bounds_big))
        else:
            Flag = True  # 剖分标志
            z_SubNum = 1
            while Flag:
                # z_SubDepth（底深）为等比序列，通项zn=dz*ratio**(n-1);
                # 求和Sn=dz*(1-ratio**n)/(1-ratio)
                z_SubDepth = z1 + dz * (1 - ratio ** z_SubNum) / (1 - ratio)
                #print(z_SubDepth)
                # 继续剖分条件：底界面深度<zmax && 剩余深度 > dz
                # 避免最后一个网格非常小，若最后一个网格很小就不需要剖分了
                if z_SubDepth < z2 and (z2 - z_SubDepth) > dz:
                    z_SubNum += 1
                else:
                    Flag = False

            # 最后z方向剖分的网格个数
            nz = int(z_SubNum)
            bounds_big = x1, x1 + nx * dx, y1, y1 + ny * dy, z1, z2
            print("Nonuniform grid(ratio: {})with new boundaries: {}".format(ratio, bounds_big))
            # ----信息查询
            # 输出按ratio剖分最后一个网格底界面，zmax之上的界面
            z_SubDepth_STL = z1 + dz * (1 - ratio ** (nz-1)) / (1 - ratio)
            print("Last depth above zmax with ratio:{:.2f}m".format(z_SubDepth_STL))
            # 输出假如按ratio剖分的最后一个网格底界面
            print("Last depth if ratio:{:.2f}m".format(z_SubDepth))

        self.bounds = bounds_big  # 实际剖分模型范围
        shape = (nz, ny, nx)
        size = int(nx * ny * nz)
        self.shape = tuple(int(i) for i in shape)
        self.size = size

        if props is None:
            self.props = {}
        else:
            self.props = props
        # The index of the current prism in an iteration. Needed when mesh is
        # used as an iterator
        self.i = 0
        # List of masked prisms. Will return None if trying to access them
        self.mask = []
        # Whether or not to change heights to z coordinate
        self.zdown = True


    def __len__(self):
        return self.size

    def __getitem__(self, index):
        if index >= self.size or index < -self.size:
            raise IndexError('mesh index out of range')
        # To walk backwards in the list
        if index < 0:
            index = self.size + index
        if index in self.mask:
            return None
        nz, ny, nx = self.shape
        k = index//(nx*ny)
        j = (index - k*(nx*ny))//nx
        i = (index - k*(nx*ny) - j*nx)
        x1 = self.bounds[0] + self.dims[0] * i
        x2 = x1 + self.dims[0]
        y1 = self.bounds[2] + self.dims[1] * j
        y2 = y1 + self.dims[1]
        # z方向
        if self.ratio == 1:
            # *---均匀剖分，k=0,...,nz-1
            # 顶界面--不受最后一行高度不定的影响
            z1 = self.bounds[4] + self.dims[2] * k
            # 底界面
            if k < nz - 1:
                # 当k=0,...,nz-2时，直接累加
                z2 = z1 + self.dims[2]
            else:
                # 当k=nz-1时，最后一行底界面为zmax
                z2 = self.bounds[5]
        else:
            # *---按照ratio自适应剖分，k=0,...,nz-1
            # （底深）为等比序列,因此先求底深，再根据底深求顶深
            # z2（底深）为等比序列，通项zn=dz*ratio**k;求和Sk=dz*(1-ratio**(k+1))/(1-ratio)
            # 当k=0,...,nz-1时，自适应剖分等比数列求和得底、顶界面
            z2 = self.bounds[4] + self.dims[2] * (1 - self.ratio ** (k + 1)) / (1 - self.ratio)  # 等比数列求和
            # 顶界面--不受最后一行高度不定的影响
            z1 = z2 - self.dims[2] * self.ratio ** k   # index=k的单元厚度:dz*ratio**n
            if k == nz - 1:
                # 当k=nz-1时，最后一行底界面为zmax
                z2 = self.bounds[5]

        props = dict([p, self.props[p][index]] for p in self.props)
        return self.celltype(x1, x2, y1, y2, z1, z2, props=props)

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.size:
            raise StopIteration
        prism = self.__getitem__(self.i)
        self.i += 1
        return prism

    def addprop(self, prop, values):
        """
        Add physical property values to the cells in the mesh.

        Different physical properties of the mesh are stored in a dictionary.

        Parameters:

        * prop : str
            Name of the physical property.
        * values :  list or array
            Value of this physical property in each prism of the mesh. For the
            ordering of prisms in the mesh see
            :class:`mesher.PrismMesh`

        """
        self.props[prop] = values

    def carvetopo(self, x, y, height, below=False):
        """
        Mask (remove) prisms from the mesh that are above the topography.

        Accessing the ith prism will return None if it was masked (above the
        topography).
        Also mask prisms outside of the topography grid provided.
        The topography height information does not need to be on a regular
        grid, it will be interpolated.

        Parameters:

        * x, y : lists
            x and y coordinates of the grid points
        * height : list or array
            Array with the height of the topography
        * below : boolean
            Will mask prisms below the input surface if set to *True*.

        """
        nz, ny, nx = self.shape
        x1, x2, y1, y2, z1, z2 = self.bounds
        dx, dy, dz = self.dims
        # The coordinates of the centers of the cells
        xc = np.arange(x1, x2, dx) + 0.5 * dx
        # Sometimes arange returns more due to rounding
        if len(xc) > nx:
            xc = xc[:-1]
        yc = np.arange(y1, y2, dy) + 0.5 * dy
        if len(yc) > ny:
            yc = yc[:-1]

        # *********计算zc的中心点
        if self.ratio == 1:
            # 均匀剖分
            zc = np.arange(z1, z2, dz) + 0.5 * dz
        else:
            # 以ratio剖分
            zc = np.zeros(nz)
            # k = 0,...,nz-2时
            for k in range(0, nz-1):
                # 等比数列求和得底界面深度；中点深度=bottom-0.5*第k个棱柱深度
                bottom = self.bounds[4] + self.dims[2] * (1 - self.ratio ** (k + 1)) / (1 - self.ratio)
                midpoint = bottom - 0.5 * self.dims[2] * self.ratio ** k
                zc[k] = midpoint
            # k = nz-1时, 中点深度=bottom+0.5*第k个棱柱深度
            zc[nz-1] = bottom + 0.5*(z2 - bottom)
        #*******end method1
        '''
        # *********计算zc的上界，只要地形位于该模型单元内就切掉
        if self.ratio == 1:
            # 均匀剖分
            zc = np.arange(z1, z2, dz)
        else:
            # 以ratio剖分
            zc = np.zeros(nz)
            # k = 0,...,nz-1时
            for k in range(0, nz):
                # 等比数列求和得底界面深度；顶界面深度=bottom-第k个棱柱深度
                bottom = self.bounds[4] + self.dims[2] * (1 - self.ratio ** (k + 1)) / (1 - self.ratio)
                midpoint = bottom - self.dims[2] * self.ratio ** k
                zc[k] = midpoint
        # ********end method2
        '''
        if len(zc) > nz:
            zc = zc[:-1]
        XC, YC = np.meshgrid(xc, yc)
        topo = scipy.interpolate.griddata((x, y), height, (XC, YC), method='cubic').ravel()
        if self.zdown:
            # -1 if to transform height into z coordinate
            topo = -1 * topo
        np.savetxt('carve_topo_interp.txt', np.c_[XC.ravel(), YC.ravel(), topo], fmt='%.8f', delimiter=' ')
        # griddata returns a masked array. If the interpolated point is out of
        # of the data range, mask will be True. Use this to remove all cells
        # below a masked topo point (ie, one with no height information)
        if np.ma.isMA(topo):
            topo_mask = topo.mask
        else:
            topo_mask = [False for i in range(len(topo))]
        c = 0
        for cellz in zc:
            for h, masked in zip(topo, topo_mask):
                if below:
                    if (masked or
                            (cellz > h and self.zdown) or
                            (cellz < h and not self.zdown)):
                        self.mask.append(c)
                else:
                    if (masked or
                            (cellz < h and self.zdown) or
                            (cellz > h and not self.zdown)):
                        self.mask.append(c)
                c += 1
        return self.mask

    def get_xs(self):
        """
        Return an array with the x coordinates of the prisms in mesh.
        """
        x1, x2, y1, y2, z1, z2 = self.bounds
        dx, dy, dz = self.dims
        nz, ny, nx = self.shape
        xs = np.arange(x1, x2 + dx, dx)
        if xs.size > nx + 1:
            return xs[:-1]
        return xs

    def get_ys(self):
        """
        Return an array with the y coordinates of the prisms in mesh.
        """
        x1, x2, y1, y2, z1, z2 = self.bounds
        dx, dy, dz = self.dims
        nz, ny, nx = self.shape
        ys = np.arange(y1, y2 + dy, dy)
        if ys.size > ny + 1:
            return ys[:-1]
        return ys

    def get_zs(self):
        """
        Return an array with the z coordinates of the prisms in mesh.
        """
        x1, x2, y1, y2, z1, z2 = self.bounds
        dx, dy, dz = self.dims
        nz, ny, nx = self.shape

        if self.ratio == 1:
            # 均匀剖分
            zs = np.arange(z1, z2 + dz, dz)
        else:
            # 以ratio剖分
            zs = np.zeros(nz+1)
            # k = 0,..., nz-1时
            for k in range(0, nz):
                # 等比数列求和得底界面深度；顶界面深度topo=bottom - 棱柱k（index）深度ak=dz*q**k
                bottom = self.bounds[4] + self.dims[2] * (1 - self.ratio ** (k + 1)) / (1 - self.ratio)
                topo = bottom - self.dims[2] * self.ratio ** k
                zs[k] = topo
            # 最后一个深度即zmax
            zs[nz] = z2

        if zs.size > nz + 1:
            return zs[:-1]
        return zs

    def get_layer(self, i):
        """
        Return the set of prisms corresponding to the ith layer of the mesh.
        Parameters:
        * i : int
            The index of the layer
        Returns:
        * prisms : list of :class:`mesher.Prism`
            The prisms in the ith layer
        """
        nz, ny, nx = self.shape
        if i >= nz or i < 0:
            raise IndexError('Layer index %d is out of range.' % (i))
        start = i * nx * ny
        end = (i + 1) * nx * ny
        layer = [self.__getitem__(p) for p in range(start, end)]
        return layer

    def layers(self):
        """
        Returns an iterator over the layers of the mesh.
        """
        nz, ny, nx = self.shape
        for i in range(nz):
            yield self.get_layer(i)

    def dump(self, meshfile, propfile, prop):
        r"""
        Dump the mesh to a file in the format required by UBC-GIF program
        MeshTools3D.

        Parameters:

        * meshfile : str or file
            Output file to save the mesh. Can be a file name or an open file.
        * propfile : str or file
            Output file to save the physical properties *prop*. Can be a file
            name or an open file.
        * prop : str
            The name of the physical property in the mesh that will be saved to
            *propfile*.

        .. note:: Uses -10000000 as the dummy value for plotting topography
        """
        if prop not in self.props:
            raise ValueError("mesh doesn't have a '%s' property." % (prop))
        isstr = False
        if isinstance(meshfile, str):
            isstr = True
            meshfile = open(meshfile, 'w')
        nz, ny, nx = self.shape
        x1, x2, y1, y2, z1, z2 = self.bounds
        dx, dy, dz = self.dims
        meshfile.writelines([
            "%d %d %d\n" % (ny, nx, nz),
            "%g %g %g\n" % (y1, x1, -z1),
            "%d*%g\n" % (ny, dy),
            "%d*%g\n" % (nx, dx),
            "%d*%g" % (nz, dz)])
        if isstr:
            meshfile.close()
        values = np.fromiter(self.props[prop], dtype=np.float)
        # Replace the masked cells with a dummy value
        values[self.mask] = -10000000
        reordered = np.ravel(np.reshape(values, self.shape), order='F')
        np.savetxt(propfile, reordered, fmt='%.4f')

    def copy(self):
        """ Return a deep copy of the current instance."""
        return cp.deepcopy(self)

class TesseroidMesh(PrismMesh):
    """
    A 3D self-adaptive regular mesh of tesseroids.

    Tesseroids are ordered as follows: first layers (height coordinate),
    then N-S rows and finaly E-W.

    Ex: in a mesh with shape ``(3,3,3)`` the 15th element (index 14) has height
    index 1 (second layer), y index 1 (second row), and x index 2 (
    third element in the column).

    This class can used as list of tesseroids. It acts
    as an iteratior (so you can loop over tesseroids).
    It also has a ``__getitem__``
    method to access individual elements in the mesh.
    In practice, it should be able to be
    passed to any function that asks for a list of tesseroids, like
    :func:`gravmag.tesseroid.gz`.

    To make the mesh incorporate a topography, use
    :meth:`mesher.TesseroidMesh.carvetopo`

    Parameters:

    * bounds : list = [w, e, s, n, top, bottom]
        Boundaries of the mesh. ``w, e, s, n`` in degrees, ``top`` and
        ``bottom`` are heights (positive upward) and in meters.
    * spacing : tuple = (dr, dlat, dlon)
        Spacing of tesseroids in the radial, latitude, and longitude directions.
    * props :  dict
        Physical properties of each tesseroid in the mesh.
        Each key should be the name of a physical property. The corresponding
        value should be a list with the values of that particular property on
        each tesseroid of the mesh.
    """

    celltype = Tesseroid

    def __init__(self, bounds, spacing, ratio=1, props=None):
        super().__init__(bounds, spacing, ratio, props)
        self.zdown = False
        self.dump = None

class PrismMeshSegment(object):
    """
    A 3D self-adaptive regular mesh of right rectangular prisms.

    Prisms are ordered as follows: first layers (z coordinate),
    then EW rows (y) and finaly x coordinate (NS).

    .. note:: Remember that the coordinate system is x->North, y->East and
        z->Down

    Ex: in a mesh with shape ``(3,3,3)`` the 15th element (index 14) has z
    index 1 (second layer), y index 1 (second row), and x index 2 (third
    element in the column).

    :class:`mesher.PrismMesh` can used as list of prisms. It acts
    as an iteratior (so you can loop over prisms). It also has a
    ``__getitem__`` method to access individual elements in the mesh.
    In practice, :class:`mesher.PrismMesh` should be able to be
    passed to any function that asks for a list of prisms, like
    :func:gravmag.prism.gz`.

    To make the mesh incorporate a topography, use
    :meth:`mesher.PrismMesh.carvetopo`

    Parameters:

    * bounds : list = [xmin, xmax, ymin, ymax, zmin, zmax]
        Boundaries of the mesh.
    * spacing : tuple = ([dz1, dz2, dz3, ...], dy, dx)
        spacing of prisms in the x, y, and z directions.
    * segment: dz is a list, 每段的dz
    * props :  dict
        Physical properties of each prism in the mesh.
        Each key should be the name of a physical property. The corresponding
        value should be a list with the values of that particular property on
        each prism of the mesh.
    """

    celltype = Prism

    def __init__(self, bounds, spacing, divisionsection, props=None):
        x1, x2, y1, y2, z1, z2 = bounds
        dzlist, dy, dx = spacing
        self.dims = (dx, dy, dzlist)
        self.segment = len(dzlist)
        self.divisionsection = divisionsection
        print(dzlist)
        # 若不整除，则建模范围xmax, ymax稍大于给定范围;zmax保持原范围
        nx = int(np.ceil((x2 - x1)/dx))  # 向上取整
        ny = int(np.ceil((y2 - y1)/dy))  # 向上取整
        # 求nz
        nz = 0
        nzlist = np.zeros(self.segment)
        nzsumlist = np.zeros(self.segment)
        for i in range(self.segment):
            # 共分为i段，每段的dz不同，计算每段的nz
            nzlist[i] = int(np.ceil((divisionsection[i+1] - divisionsection[i])/dzlist[i]))
            nz = nz + nzlist[i]
            nzsumlist[i] = nz

        print("nz is ", nz)
        print("nzlist is", nzlist)
        print("nzsumlist is", nzsumlist)
        bounds_big = x1, x1 + nx * dx, y1, y1 + ny * dy, z1, self.divisionsection[-2] + nzlist[-1] * dzlist[-1]
        print("Uniform Segment grid with new boundaries: {}".format(bounds_big))

        self.nzlist = nzlist
        self.nzsumlist = nzsumlist
        self.bounds = bounds_big  # 实际剖分模型范围
        shape = (nz, ny, nx)
        size = int(nx * ny * nz)
        self.shape = tuple(int(i) for i in shape)
        self.size = size

        if props is None:
            self.props = {}
        else:
            self.props = props
        # The index of the current prism in an iteration. Needed when mesh is
        # used as an iterator
        self.i = 0
        # List of masked prisms. Will return None if trying to access them
        self.mask = []
        # Whether or not to change heights to z coordinate
        self.zdown = True


    def __len__(self):
        return self.size

    def __getitem__(self, index):
        if index >= self.size or index < -self.size:
            raise IndexError('mesh index out of range')
        # To walk backwards in the list
        if index < 0:
            index = self.size + index
        if index in self.mask:
            return None
        nz, ny, nx = self.shape
        k = index//(nx*ny)
        j = (index - k*(nx*ny))//nx
        i = (index - k*(nx*ny) - j*nx)
        x1 = self.bounds[0] + self.dims[0] * i
        x2 = x1 + self.dims[0]
        y1 = self.bounds[2] + self.dims[1] * j
        y2 = y1 + self.dims[1]
        # z方向
        # 判断位于哪一段
        for iseg in range(self.segment):
            if k < self.nzsumlist[iseg]:
                kloc = iseg
                break
        if kloc == 0:
            # 第一段: 顶界面
            z1 = self.bounds[4] + self.dims[2][kloc] * k
            # 底界面
            z2 = z1 + self.dims[2][kloc]
        if kloc >= 1:
            # 第2段及之后
            # 顶界面
            z1 = self.divisionsection[kloc] + self.dims[2][kloc] * (k-self.nzsumlist[kloc-1])
            # 底界面
            z2 = z1 + self.dims[2][kloc]

        props = dict([p, self.props[p][index]] for p in self.props)
        return self.celltype(x1, x2, y1, y2, z1, z2, props=props)

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.size:
            raise StopIteration
        prism = self.__getitem__(self.i)
        self.i += 1
        return prism

    def addprop(self, prop, values):
        """
        Add physical property values to the cells in the mesh.

        Different physical properties of the mesh are stored in a dictionary.

        Parameters:

        * prop : str
            Name of the physical property.
        * values :  list or array
            Value of this physical property in each prism of the mesh. For the
            ordering of prisms in the mesh see
            :class:`mesher.PrismMesh`

        """
        self.props[prop] = values

    def carvetopo(self, x, y, height, below=False):
        """
        Mask (remove) prisms from the mesh that are above the topography.

        Accessing the ith prism will return None if it was masked (above the
        topography).
        Also mask prisms outside of the topography grid provided.
        The topography height information does not need to be on a regular
        grid, it will be interpolated.

        Parameters:

        * x, y : lists
            x and y coordinates of the grid points
        * height : list or array
            Array with the height of the topography
        * below : boolean
            Will mask prisms below the input surface if set to *True*.

        """
        nz, ny, nx = self.shape
        x1, x2, y1, y2, z1, z2 = self.bounds
        dx, dy, dzlist = self.dims
        # The coordinates of the centers of the cells
        xc = np.arange(x1, x2, dx) + 0.5 * dx
        # Sometimes arange returns more due to rounding
        if len(xc) > nx:
            xc = xc[:-1]
        yc = np.arange(y1, y2, dy) + 0.5 * dy
        if len(yc) > ny:
            yc = yc[:-1]

       # 计算zc
        zc = []
        # # method1-----计算z方向的中心点
        # # 判断位于哪一段
        # for iseg in range(self.segment):
        #     tmp = list(np.arange(self.divisionsection[iseg],
        #                          self.divisionsection[iseg + 1], dzlist[iseg]) + dzlist[iseg]/2.0)
        #     zc.extend(tmp)
        # # end of method1
        # method 2----计算zc的上界，只要地形位于该模型单元内就切掉，只要zc位于棱柱内部即去掉
        # 判断位于哪一段
        for iseg in range(self.segment):
            tmp = list(np.arange(self.divisionsection[iseg],
                                 self.divisionsection[iseg + 1], dzlist[iseg]))
            zc.extend(tmp)
        # end of method 2

        zc = np.array(zc)
        if len(zc) > nz:
            zc = zc[:-1]
        XC, YC = np.meshgrid(xc, yc)
        topo = scipy.interpolate.griddata((x, y), height, (XC, YC), method='nearest').ravel()

        if self.zdown:
            # -1 if to transform height into z coordinate
            topo = -1 * topo
        np.savetxt('carve_topo_interp.txt', np.c_[XC.ravel(), YC.ravel(), topo], fmt='%.8f', delimiter=' ')
        # griddata returns a masked array. If the interpolated point is out of
        # of the data range, mask will be True. Use this to remove all cells
        # below a masked topo point (ie, one with no height information)
        if np.ma.isMA(topo):
            topo_mask = topo.mask
        else:
            topo_mask = [False for i in range(len(topo))]
        c = 0
        for cellz in zc:
            for h, masked in zip(topo, topo_mask):
                if below:
                    if (masked or
                            (cellz > h and self.zdown) or
                            (cellz < h and not self.zdown)):
                        self.mask.append(c)
                else:
                    if (masked or
                            (cellz < h and self.zdown) or
                            (cellz > h and not self.zdown)):
                        self.mask.append(c)
                c += 1
        return self.mask

    def get_xs(self):
        """
        Return an array with the x coordinates of the prisms in mesh.
        """
        x1, x2, y1, y2, z1, z2 = self.bounds
        dx, dy, dz = self.dims
        nz, ny, nx = self.shape
        xs = np.arange(x1, x2 + dx, dx)
        if xs.size > nx + 1:
            return xs[:-1]
        return xs

    def get_ys(self):
        """
        Return an array with the y coordinates of the prisms in mesh.
        """
        x1, x2, y1, y2, z1, z2 = self.bounds
        dx, dy, dz = self.dims
        nz, ny, nx = self.shape
        ys = np.arange(y1, y2 + dy, dy)
        if ys.size > ny + 1:
            return ys[:-1]
        return ys

    def get_zs(self):
        """
        Return an array with the z coordinates of the prisms in mesh.
        """
        x1, x2, y1, y2, z1, z2 = self.bounds
        dx, dy, dzlist = self.dims
        nz, ny, nx = self.shape
        # 初始化
        zs = []
        # 判断位于哪一段
        for iseg in range(self.segment):
            tmp = list(np.arange(self.divisionsection[iseg], self.divisionsection[iseg+1], dzlist[iseg]))
            zs.extend(tmp)
        zs.append(z2)
        zs = np.array(zs)

        if zs.size > nz + 1:
            return zs[:-1]
        return zs

    def get_layer(self, i):
        """
        Return the set of prisms corresponding to the ith layer of the mesh.
        Parameters:
        * i : int
            The index of the layer
        Returns:
        * prisms : list of :class:`mesher.Prism`
            The prisms in the ith layer
        """
        nz, ny, nx = self.shape
        if i >= nz or i < 0:
            raise IndexError('Layer index %d is out of range.' % (i))
        start = i * nx * ny
        end = (i + 1) * nx * ny
        layer = [self.__getitem__(p) for p in range(start, end)]
        return layer

    def layers(self):
        """
        Returns an iterator over the layers of the mesh.
        """
        nz, ny, nx = self.shape
        for i in range(nz):
            yield self.get_layer(i)

    def dump(self, meshfile, propfile, prop):
        r"""
        Dump the mesh to a file in the format required by UBC-GIF program
        MeshTools3D.

        Parameters:

        * meshfile : str or file
            Output file to save the mesh. Can be a file name or an open file.
        * propfile : str or file
            Output file to save the physical properties *prop*. Can be a file
            name or an open file.
        * prop : str
            The name of the physical property in the mesh that will be saved to
            *propfile*.

        .. note:: Uses -10000000 as the dummy value for plotting topography
        """
        if prop not in self.props:
            raise ValueError("mesh doesn't have a '%s' property." % (prop))
        isstr = False
        if isinstance(meshfile, str):
            isstr = True
            meshfile = open(meshfile, 'w')
        nz, ny, nx = self.shape
        x1, x2, y1, y2, z1, z2 = self.bounds
        dx, dy, dz = self.dims
        meshfile.writelines([
            "%d %d %d\n" % (ny, nx, nz),
            "%g %g %g\n" % (y1, x1, -z1),
            "%d*%g\n" % (ny, dy),
            "%d*%g\n" % (nx, dx),
            "%d*%g" % (nz, dz)])
        if isstr:
            meshfile.close()
        values = np.fromiter(self.props[prop], dtype=np.float)
        # Replace the masked cells with a dummy value
        values[self.mask] = -10000000
        reordered = np.ravel(np.reshape(values, self.shape), order='F')
        np.savetxt(propfile, reordered, fmt='%.4f')

    def copy(self):
        """ Return a deep copy of the current instance."""
        return cp.deepcopy(self)

class TesseroidMeshSegment(PrismMeshSegment):
    """
    A 3D self-adaptive regular mesh of tesseroids.

    Tesseroids are ordered as follows: first layers (height coordinate),
    then N-S rows and finaly E-W.

    Ex: in a mesh with shape ``(3,3,3)`` the 15th element (index 14) has height
    index 1 (second layer), y index 1 (second row), and x index 2 (
    third element in the column).

    This class can used as list of tesseroids. It acts
    as an iteratior (so you can loop over tesseroids).
    It also has a ``__getitem__``
    method to access individual elements in the mesh.
    In practice, it should be able to be
    passed to any function that asks for a list of tesseroids, like
    :func:`gravmag.tesseroid.gz`.

    To make the mesh incorporate a topography, use
    :meth:`mesher.TesseroidMesh.carvetopo`

    Parameters:

    * bounds : list = [w, e, s, n, top, bottom]
        Boundaries of the mesh. ``w, e, s, n`` in degrees, ``top`` and
        ``bottom`` are heights (positive upward) and in meters.
    * spacing : tuple = (dr, dlat, dlon)
        Spacing of tesseroids in the radial, latitude, and longitude directions.
    * props :  dict
        Physical properties of each tesseroid in the mesh.
        Each key should be the name of a physical property. The corresponding
        value should be a list with the values of that particular property on
        each tesseroid of the mesh.
    """

    celltype = Tesseroid

    def __init__(self, bounds, spacing, divisionsection, props=None):
        super().__init__(bounds, spacing, divisionsection, props)
        self.zdown = False
        self.dump = None
