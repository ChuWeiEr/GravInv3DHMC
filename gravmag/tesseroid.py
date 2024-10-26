r"""
Forward model the gravitational fields of a tesseroid (spherical prism).

Functions in this module calculate the gravitational fields of a tesseroid with
respect to the local North-oriented coordinate system of the computation point.
See the figure below.

.. admonition:: Coordinate systems

    The gravitational attraction
    and gravity gradient tensor
    are calculated with respect to
    the local coordinate system of the computation point.
    This system has **x -> North**, **y -> East**, **z -> up**
    (radial direction).

.. warning:: The :math:`g_z` component is an **exception** to this.
    In order to conform with the regular convention
    of z-axis pointing toward the center of the Earth,
    **this component only** is calculated with **z -> Down**.
    This way, gravity anomalies of
    tesseroids with positive density
    are positive, not negative.

Gravity
-------

Forward modeling of gravitational fields is performed by functions:

:func:`gravmag.tesseroid.potential`,
:func:`gravmag.tesseroid.gx`,
:func:`gravmag.tesseroid.gy`,
:func:`gravmag.tesseroid.gz`,
:func:`gravmag.tesseroid.gxx`,
:func:`gravmag.tesseroid.gxy`,
:func:`gravmag.tesseroid.gxz`,
:func:`gravmag.tesseroid.gyy`,
:func:`gravmag.tesseroid.gyz`,
:func:`gravmag.tesseroid.gzz`

The fields are calculated using Gauss-Legendre Quadrature integration and the
adaptive discretization algorithm of Uieda et al. (2016). The accuracy of the
integration is controlled by the ``ratio`` argument. Larger values cause finer
discretization and more accuracy but slower computation times. The defaults
values are the ones suggested in the paper and guarantee an accuracy of
approximately 0.1%.

.. warning::

    The integration error may be larger than this if the computation
    points are closer than 1 km of the tesseroids. This effect is more
    significant in the gravity gradient components.

References
++++++++++

Uieda, L., V. Barbosa, and C. Braitenberg (2016), Tesseroids: Forward-modeling
gravitational fields in spherical coordinates, Geophysics, F41-F48,
doi:10.1190/geo2015-0204.1

----
"""

import multiprocessing
import warnings
# 解决溢出问题
from gravmag import pickle4reducer
import multiprocessing as mp
ctx = mp.get_context()
ctx.reducer = pickle4reducer.Pickle4Reducer()

import numpy as np
from gravmag import _tesseroid_numba
from constants import SI2MGAL, SI2EOTVOS, MEAN_EARTH_RADIUS, G, g0, Gs

RATIO_V = 1
RATIO_G = 1.6
RATIO_GG = 8
STACK_SIZE = 100

def _check_input(lon, lat, height, model, dens, ratio, njobs, pool):
    """
    Check if the inputs are as expected and generate the output array.

    Returns:

    * results : 1d-array, zero filled
    * kernel2d: 2d-array, zero filled
    """
    assert lon.shape == lat.shape == height.shape, \
        "Input coordinate arrays must have same shape"
    assert ratio > 0, "Invalid ratio {}. Must be > 0.".format(ratio)
    assert njobs > 0, "Invalid number of jobs {}. Must be > 0.".format(njobs)
    if njobs == 1:
        assert pool is None, "njobs should be number of processes in the pool"
    # initialize
    result = np.zeros_like(lon)
    # 计算小棱柱的数量,将carvetopo的小棱柱去除
    tessnum = 0    
    for p in model:
        if p is None or ('density' not in model.props and dens is None):
            continue
        tessnum += 1
    print("Number of effective tesseroids", tessnum)
    kernel2d = np.zeros([lon.shape[0], tessnum])
    return result, kernel2d


def _convert_coords(lon, lat, height):
    """
    Convert angles to radians and heights to radius.

    Pre-compute the sine and cosine of latitude because that is what we need
    from it.
    """
    # Convert things to radians
    lon = np.radians(lon)
    lat = np.radians(lat)
    sinlat = np.sin(lat)
    coslat = np.cos(lat)
    # Transform the heights into radius
    radius = MEAN_EARTH_RADIUS + height
    return lon, sinlat, coslat, radius


def _check_tesseroid(tesseroid, dens):
    """
    Check if the tesseroid is valid and get the right density to use.

    Returns None if the tesseroid should be ignored. Else, return the density
    that should be used.
    """
    if tesseroid is None:
        return None
    if 'density' not in tesseroid.props and dens is None:
        return None
    w, e, s, n, top, bottom = tesseroid.get_bounds()
    # Check if the dimensions given are valid
    assert w <= e and s <= n and top >= bottom, \
        "Invalid tesseroid dimensions {}".format(tesseroid.get_bounds())
    # Check if the tesseroid has volume > 0
    if (e - w <= 1e-6) or (n - s <= 1e-6) or (top - bottom <= 1e-3):
        msg = ("Encountered tesseroid with dimensions smaller than the " +
               "numerical threshold (1e-6 degrees or 1e-3 m). " +
               "Ignoring this tesseroid.")
        warnings.warn(msg, RuntimeWarning)

        return None
    if dens is not None:
        density = dens
    else:
        density = tesseroid.props['density']
    return density


def _dispatcher(field, lon, lat, height, model, **kwargs):
    """
    Dispatch the computation of *field* to the appropriate function.
    Returns:
    * result : 1d-array
    """
    njobs = kwargs.get('njobs', 1)
    pool = kwargs.get('pool', None)
    dens = kwargs['dens']
    ratio = kwargs['ratio']
    result, kernel2d = _check_input(lon, lat, height, model, dens, ratio, njobs, pool)
    if njobs > 1 and pool is None:
        pool = multiprocessing.Pool(njobs)
        created_pool = True
    else:
        created_pool = False
    if pool is None:
        # 不使用多进程，则result, kernel2d一起输出
        _forward_model([lon, lat, height, result, kernel2d, model, dens, ratio,
                        field])
    else:
        # 使用多进程，result, kernel2d分别输出
        chunks = _split_arrays(arrays=[lon, lat, height, result],
                               arrays2D=[kernel2d],
                               extra_args=[model, dens, ratio, field],
                               nparts=njobs)
        result = np.hstack(pool.map(_forward_model_result, chunks))
        kernel2d = np.vstack(pool.map(_forward_model_kernel2d, chunks))
    if created_pool:
        pool.close()
    return result, kernel2d


def _forward_model(args):
    """
    Run the computations on the model for a given list of arguments.

    This is used because multiprocessing.Pool.map can only use functions that
    receive a single argument.

    Arguments should be, in order:

    lon, lat, height, result, model, dens, ratio, field
    """
    lon, lat, height, result, kernel2d, model, dens, ratio, field = args
    lon, sinlat, coslat, radius = _convert_coords(lon, lat, height)
    func = getattr(_tesseroid_numba, field)
    warning_msg = (
        "Stopped dividing a tesseroid because it's dimensions would be " +
        "below the minimum numerical threshold (1e-6 degrees or 1e-3 m). " +
        "Will compute without division. Cannot guarantee the accuracy of " +
        "the solution.")
    # Arrays needed by the kernel. Can't allocate them inside the kernel
    # because numba doesn't like that.
    stack = np.empty((STACK_SIZE, 6), dtype='float')
    lonc = np.empty(2, dtype='float')
    sinlatc = np.empty(2, dtype='float')
    coslatc = np.empty(2, dtype='float')
    rc = np.empty(2)
    # 小棱柱循环计算！！！
    # 给小棱柱计数
    tessnum = 0
    for tesseroid in model:
        density = _check_tesseroid(tesseroid, dens)
        if density is None:
            continue
        # bounds是一个小棱柱的bounds
        bounds = np.array(tesseroid.get_bounds())
        # 这里返回计算结果，ratio即D
        error = func(lon, sinlat, coslat, radius, bounds, density, ratio,
                     stack, lonc, sinlatc, coslatc, rc, result, kernel2d, tessnum)
        # 如果小棱柱的维度<0.1 # in meters. ~1e-6  degrees则报错
        if error != 0:
            warnings.warn(warning_msg, RuntimeWarning)
        # 小棱柱的序号，即开始计算下一个小棱柱
        tessnum += 1
    return result, kernel2d

def _forward_model_result(args):
    """
    只返回result
    """
    lon, lat, height, result, kernel2d, model, dens, ratio, field = args
    lon, sinlat, coslat, radius = _convert_coords(lon, lat, height)
    func = getattr(_tesseroid_numba, field)
    warning_msg = (
        "Stopped dividing a tesseroid because it's dimensions would be " +
        "below the minimum numerical threshold (1e-6 degrees or 1e-3 m). " +
        "Will compute without division. Cannot guarantee the accuracy of " +
        "the solution.")
    # Arrays needed by the kernel. Can't allocate them inside the kernel
    # because numba doesn't like that.
    stack = np.empty((STACK_SIZE, 6), dtype='float')
    lonc = np.empty(2, dtype='float')
    sinlatc = np.empty(2, dtype='float')
    coslatc = np.empty(2, dtype='float')
    rc = np.empty(2)
    # 小棱柱循环计算！！！
    # 给小棱柱计数
    tessnum = 0
    for tesseroid in model:
        density = _check_tesseroid(tesseroid, dens)
        if density is None:
            continue
        # bounds是一个小棱柱的bounds
        bounds = np.array(tesseroid.get_bounds())
        # 这里返回计算结果，ratio即D
        error = func(lon, sinlat, coslat, radius, bounds, density, ratio,
                     stack, lonc, sinlatc, coslatc, rc, result, kernel2d, tessnum)
        # 如果小棱柱的维度<0.1 # in meters. ~1e-6  degrees则报错
        if error != 0:
            warnings.warn(warning_msg, RuntimeWarning)
        # 小棱柱的序号，即开始计算下一个小棱柱
        tessnum += 1
    return result

def _forward_model_kernel2d(args):
    """
    只返回kernel2d
    """
    lon, lat, height, result, kernel2d, model, dens, ratio, field = args
    lon, sinlat, coslat, radius = _convert_coords(lon, lat, height)
    func = getattr(_tesseroid_numba, field)
    warning_msg = (
        "Stopped dividing a tesseroid because it's dimensions would be " +
        "below the minimum numerical threshold (1e-6 degrees or 1e-3 m). " +
        "Will compute without division. Cannot guarantee the accuracy of " +
        "the solution.")
    # Arrays needed by the kernel. Can't allocate them inside the kernel
    # because numba doesn't like that.
    stack = np.empty((STACK_SIZE, 6), dtype='float')
    lonc = np.empty(2, dtype='float')
    sinlatc = np.empty(2, dtype='float')
    coslatc = np.empty(2, dtype='float')
    rc = np.empty(2)
    # 小棱柱循环计算！！！
    # 给小棱柱计数
    tessnum = 0
    for tesseroid in model:
        density = _check_tesseroid(tesseroid, dens)
        if density is None:
            continue
        # bounds是一个小棱柱的bounds
        bounds = np.array(tesseroid.get_bounds())
        # 这里返回计算结果，ratio即D
        error = func(lon, sinlat, coslat, radius, bounds, density, ratio,
                     stack, lonc, sinlatc, coslatc, rc, result, kernel2d, tessnum)
        # 如果小棱柱的维度<0.1 # in meters. ~1e-6  degrees则报错
        if error != 0:
            warnings.warn(warning_msg, RuntimeWarning)
        # 小棱柱的序号，即开始计算下一个小棱柱
        tessnum += 1
    return kernel2d


def _split_arrays(arrays, arrays2D, extra_args, nparts):
    """
    Split the coordinate arrays into nparts. Add extra_args to each part.
    """
    size = len(arrays[0])
    n = size//nparts
    strides = [(i*n, (i + 1)*n) for i in range(nparts - 1)]
    strides.append((strides[-1][-1], size))
    chunks = [[x[low:high] for x in arrays] + [y[low:high][:] for y in arrays2D] + extra_args
              for low, high in strides]
    return chunks


def potential(lon, lat, height, model, dens=None, ratio=RATIO_V,
              njobs=1, pool=None):
    """
    Calculate the gravitational potential due to a tesseroid model.

    .. warning:: Tesseroids with dimensions < 10 cm will be ignored to avoid
        numerical errors.

    Implements the method of Uieda et al. (2016).

    Parameters:

    * lon, lat, height : arrays
        Arrays with the longitude, latitude and height coordinates of the
        computation points.
    * model : list of :class:`mesher.Tesseroid`
        The density model used to calculate the gravitational effect.
        Tesseroids must have the property ``'density'``. Those that don't have
        this property will be ignored in the computations. Elements that are
        None will also be ignored.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the tesseroids. Use this, e.g., for sensitivity matrix building.
    * ratio : float
        Will divide each tesseroid until the distance between it and the
        computation points is < ratio*size of tesseroid. Used to guarantee the
        accuracy of the numerical integration.
    * njobs : int
        Split the computation into *njobs* parts and run it in parallel using
        ``multiprocessing``. If ``njobs=1`` will run the computation in serial.
    * pool : None or multiprocessing.Pool object
        If not None, will use this pool to run the computation in parallel
        instead of creating a new one. You must still specify *njobs* as the
        number of processes in the pool. Use this to avoid spawning processes
        on each call to this functions, which can have significant overhead.

    Returns:

    * res : array
        The calculated field in SI units

    References:

    Uieda, L., V. Barbosa, and C. Braitenberg (2016), Tesseroids:
    Forward-modeling gravitational fields in spherical coordinates, Geophysics,
    F41-F48, doi:10.1190/geo2015-0204.1

    """
    field = 'potential'
    result, kernel2d = _dispatcher(field, lon, lat, height, model, dens=dens,
                         ratio=ratio, njobs=njobs, pool=pool)
    result *= G
    kernel2d *= G
    return result, kernel2d


def geoid(lon, lat, height, model, dens=None, ratio=RATIO_V,
          njobs=1, pool=None):
    """
    Calculate geoid.
    geoid = potential/g0
    """
    field = 'potential'
    result, kernel2d = _dispatcher(field, lon, lat, height, model, dens=dens,
                         ratio=ratio, njobs=njobs, pool=pool)
    result *= G/g0
    kernel2d *= G/g0
    return result, kernel2d


def gx(lon, lat, height, model, dens=None, ratio=RATIO_G,
       njobs=1, pool=None):
    """
    Calculate the North component of the gravitational attraction.
    """
    field = 'gx'
    result, kernel2d = _dispatcher(field, lon, lat, height, model, dens=dens,
                         ratio=ratio, njobs=njobs, pool=pool)
    result = result*SI2MGAL*G
    kernel2d = kernel2d*SI2MGAL*G
    return result, kernel2d


def gy(lon, lat, height, model, dens=None, ratio=RATIO_G,
       njobs=1, pool=None):
    """
    Calculate the East component of the gravitational attraction.

    """
    field = 'gy'
    result, kernel2d = _dispatcher(field, lon, lat, height, model, dens=dens,
                         ratio=ratio, njobs=njobs, pool=pool)
    result = result*SI2MGAL*Gs
    kernel2d = kernel2d*SI2MGAL*Gs
    return result, kernel2d


def gz(lon, lat, height, model, dens=None, ratio=RATIO_G,
       njobs=1, pool=None):
    """
    Calculate the radial component of the gravitational attraction.
    """
    field = 'gz'
    result, kernel2d = _dispatcher(field, lon, lat, height, model, dens=dens,
                         ratio=ratio, njobs=njobs, pool=pool)
    result = result*SI2MGAL*G
    kernel2d = kernel2d*SI2MGAL*G
    return result, kernel2d

def gxx(lon, lat, height, model, dens=None, ratio=RATIO_GG,
        njobs=1, pool=None):
    """
    Calculate the xx component of the gravity gradient tensor.
    """
    field = 'gxx'
    result, kernel2d = _dispatcher(field, lon, lat, height, model, dens=dens,
                         ratio=ratio, njobs=njobs, pool=pool)
    result = result*SI2EOTVOS*G
    kernel2d = kernel2d*SI2EOTVOS*G
    return result, kernel2d


def gxy(lon, lat, height, model, dens=None, ratio=RATIO_GG,
        njobs=1, pool=None):
    """
    Calculate the xy component of the gravity gradient tensor.
    """
    field = 'gxy'
    result, kernel2d = _dispatcher(field, lon, lat, height, model, dens=dens,
                         ratio=ratio, njobs=njobs, pool=pool)
    result = result*SI2EOTVOS*G
    kernel2d = kernel2d * SI2EOTVOS * G
    return result, kernel2d


def gxz(lon, lat, height, model, dens=None, ratio=RATIO_GG,
        njobs=1, pool=None):
    """
    Calculate the xz component of the gravity gradient tensor.
    """
    field = 'gxz'
    result, kernel2d = _dispatcher(field, lon, lat, height, model, dens=dens,
                         ratio=ratio, njobs=njobs, pool=pool)
    result = result*SI2EOTVOS*G
    kernel2d = kernel2d * SI2EOTVOS * G
    return result, kernel2d


def gyy(lon, lat, height, model, dens=None, ratio=RATIO_GG,
        njobs=1, pool=None):
    """
    Calculate the yy component of the gravity gradient tensor.
    """
    field = 'gyy'
    result, kernel2d = _dispatcher(field, lon, lat, height, model, dens=dens,
                         ratio=ratio, njobs=njobs, pool=pool)
    result = result*SI2EOTVOS*G
    kernel2d = kernel2d * SI2EOTVOS * G
    return result, kernel2d


def gyz(lon, lat, height, model, dens=None, ratio=RATIO_GG,
        njobs=1, pool=None):
    """
    Calculate the yz component of the gravity gradient tensor.
    """
    field = 'gyz'
    result, kernel2d = _dispatcher(field, lon, lat, height, model, dens=dens,
                         ratio=ratio, njobs=njobs, pool=pool)
    result = result*SI2EOTVOS*G
    kernel2d = kernel2d * SI2EOTVOS * G
    return result, kernel2d


def gzz(lon, lat, height, model, dens=None, ratio=RATIO_GG,
        njobs=1, pool=None):
    """
    Calculate the zz component of the gravity gradient tensor.
    """
    field = 'gzz'
    result, kernel2d = _dispatcher(field, lon, lat, height, model, dens=dens,
                         ratio=ratio, njobs=njobs, pool=pool)
    result = result*SI2EOTVOS*G
    kernel2d = kernel2d * SI2EOTVOS * G
    return result, kernel2d

# magnetic forward
