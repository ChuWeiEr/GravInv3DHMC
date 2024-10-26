r"""
Calculate the potential fields of the 3D right rectangular prism.

.. note:: All input units are SI. Output is in conventional units: SI for the
    gravitatonal potential, mGal for gravity, Eotvos for gravity gradients, nT
    for magnetic total field anomalies.

.. note:: The coordinate system of the input parameters is x -> North,
    y -> East and z -> Down.

**Gravity**

The gravitational fields are calculated using the formula of Nagy et al.
(2000). Available functions are:

* :func:`gravmag.prism.potential`
* :func:`gravmag.prism.gx`
* :func:`gravmag.prism.gy`
* :func:`gravmag.prism.gz`
* :func:`gravmag.prism.gxx`
* :func:`gravmag.prism.gxy`
* :func:`gravmag.prism.gxz`
* :func:`gravmag.prism.gyy`
* :func:`gravmag.prism.gyz`
* :func:`gravmag.prism.gzz`

.. warning::

    The gxy, gxz, and gyz components have singularities when the computation
    point is aligned with the corners of the prism on the bottom, east, and
    north sides, respectively. In these cases, the above functions will move
    the computation point slightly to avoid these singularities. Unfortunately,
    this means that the result will not be as accurate **on those points**.


**Magnetic**

Available fields are the total-field anomaly (using the formula of
Bhattacharyya, 1964) and x, y, z components of the magnetic induction:

* :func:`gravmag.prism.tf`
* :func:`gravmag.prism.bx`
* :func:`gravmag.prism.by`
* :func:`gravmag.prism.bz`

**Auxiliary Functions**

Calculates the second derivatives of the function

.. math::

    \phi(x,y,z) = \int\int\int \frac{1}{r}
                  \mathrm{d}\nu \mathrm{d}\eta \mathrm{d}\zeta

with respect to the variables :math:`x`, :math:`y`, and :math:`z`.
In this equation,

.. math::

    r = \sqrt{(x - \nu)^2 + (y - \eta)^2 + (z - \zeta)^2}

and :math:`\nu`, :math:`\eta`, :math:`\zeta` are the Cartesian
coordinates of an element inside the volume of a 3D prism.
These second derivatives are used to calculate
the total field anomaly and the gravity gradient tensor
components.

* :func:`gravmag.prism.kernelxx`
* :func:`gravmag.prism.kernelxy`
* :func:`gravmag.prism.kernelxz`
* :func:`gravmag.prism.kernelyy`
* :func:`gravmag.prism.kernelyz`
* :func:`gravmag.prism.kernelzz`

**References**

Bhattacharyya, B. K. (1964), Magnetic anomalies due to prism-shaped bodies with
arbitrary polarization, Geophysics, 29(4), 517, doi: 10.1190/1.1439386.

Nagy, D., G. Papp, and J. Benedek (2000), The gravitational potential and its
derivatives for the prism: Journal of Geodesy, 74, 552--560,
doi: 10.1007/s001900000116.

----
"""
#from __future__ import division, absolute_import

import numpy
import numpy as np
import multiprocessing

import utils
from constants import G, SI2EOTVOS, CM, T2NT, SI2MGAL, g0
try:
    from . import _prism
    #print("_prism is right")
except ImportError:
    _prism = None
    #print("error")


def _potential(xp, yp, zp, res, kernel2d, prisms, dens=None):
    """
    Calculates the gravitational potential.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> **DOWN**.

    .. note:: All input and output values in **SI** units(!)!

    Parameters:

    * xp, yp, zp : arrays
        Arrays with the x, y, and z coordinates of the computation points.
    * prisms : list of :class:`mesher.Prism`
        The density model used to calculate the gravitational effect.
        Prisms must have the property ``'density'``. Prisms that don't have
        this property will be ignored in the computations. Elements of *prisms*
        that are None will also be ignored. *prisms* can also be a
        :class:`mesher.PrismMesh`.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the prisms. Use this, e.g., for sensitivity matrix building.

    Returns:

    * res : array
        The field calculated on xp, yp, zp

    """
    if xp.shape != yp.shape or xp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same length!")
    size = len(xp)
    k = 0
    for prism in prisms:
        if prism is None or ('density' not in prism.props and dens is None):
            continue
        if dens is None:
            density = prism.props['density']
        else:
            density = dens
        x1, x2 = prism.x1, prism.x2
        y1, y2 = prism.y1, prism.y2
        z1, z2 = prism.z1, prism.z2
        kernel1D = numpy.zeros(size, dtype=numpy.float)
        _prism.potential(xp, yp, zp, x1, x2, y1, y2, z1, z2, density, res, kernel1D)
        kernel2d[:, k] = kernel1D
        k += 1

    res *= G
    kernel2d *= G
    return res, kernel2d


def _geoid(xp, yp, zp, res, kernel2d, prisms, dens=None):
    """
    Calculates the geoid.
    """
    if xp.shape != yp.shape or xp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same length!")
    size = len(xp)
    k = 0
    for prism in prisms:
        if prism is None or ('density' not in prism.props and dens is None):
            continue
        if dens is None:
            density = prism.props['density']
        else:
            density = dens
        x1, x2 = prism.x1, prism.x2
        y1, y2 = prism.y1, prism.y2
        z1, z2 = prism.z1, prism.z2
        kernel1D = numpy.zeros(size, dtype=numpy.float)
        _prism.potential(xp, yp, zp, x1, x2, y1, y2, z1, z2, density, res, kernel1D)
        kernel2d[:, k] = kernel1D
        k += 1

    res *= G/g0
    kernel2d *= G/g0

    return res, kernel2d

def _gx(xp, yp, zp, res, kernel2d, prisms, dens=None):
    """
    Calculates the :math:`g_x` gravity acceleration component.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> **DOWN**.

    .. note:: All input values in **SI** units(!) and output in **mGal**!

    Parameters:

    * xp, yp, zp : arrays
        Arrays with the x, y, and z coordinates of the computation points.
    * prisms : list of :class:`mesher.Prism`
        The density model used to calculate the gravitational effect.
        Prisms must have the property ``'density'``. Prisms that don't have
        this property will be ignored in the computations. Elements of *prisms*
        that are None will also be ignored. *prisms* can also be a
        :class:`mesher.PrismMesh`.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the prisms. Use this, e.g., for sensitivity matrix building.

    Returns:

    * res : array
        The field calculated on xp, yp, zp

    """
    if xp.shape != yp.shape or xp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same length!")
    size = len(xp)
    k = 0
    for prism in prisms:
        if prism is None or ('density' not in prism.props and dens is None):
            continue
        if dens is None:
            density = prism.props['density']
        else:
            density = dens
        x1, x2 = prism.x1, prism.x2
        y1, y2 = prism.y1, prism.y2
        z1, z2 = prism.z1, prism.z2
        kernel1D = numpy.zeros(size, dtype=numpy.float)
        _prism.gx(xp, yp, zp, x1, x2, y1, y2, z1, z2, density, res, kernel1D)
        kernel2d[:, k] = kernel1D
        k += 1

    res *= G * SI2MGAL
    kernel2d *= G * SI2MGAL

    return res, kernel2d


def _gy(xp, yp, zp, res, kernel2d, prisms, dens=None):
    """
    Calculates the :math:`g_y` gravity acceleration component.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> **DOWN**.

    .. note:: All input values in **SI** units(!) and output in **mGal**!

    Parameters:

    * xp, yp, zp : arrays
        Arrays with the x, y, and z coordinates of the computation points.
    * prisms : list of :class:`mesher.Prism`
        The density model used to calculate the gravitational effect.
        Prisms must have the property ``'density'``. Prisms that don't have
        this property will be ignored in the computations. Elements of *prisms*
        that are None will also be ignored. *prisms* can also be a
        :class:`mesher.PrismMesh`.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the prisms. Use this, e.g., for sensitivity matrix building.

    Returns:

    * res : array
        The field calculated on xp, yp, zp

    """
    if xp.shape != yp.shape or xp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same length!")
    size = len(xp)
    k = 0
    for prism in prisms:
        if prism is None or ('density' not in prism.props and dens is None):
            continue
        if dens is None:
            density = prism.props['density']
        else:
            density = dens
        x1, x2 = prism.x1, prism.x2
        y1, y2 = prism.y1, prism.y2
        z1, z2 = prism.z1, prism.z2
        kernel1D = numpy.zeros(size, dtype=numpy.float)
        _prism.gy(xp, yp, zp, x1, x2, y1, y2, z1, z2, density, res, kernel1D)
        kernel2d[:, k] = kernel1D
        k += 1

    res *= G * SI2MGAL
    kernel2d *= G * SI2MGAL

    return res, kernel2d


def _gz(xp, yp, zp, res, kernel2d, prisms, dens=None):
    """
    Calculates the :math:`g_z` gravity acceleration component.
    """
    if xp.shape != yp.shape or xp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same length!")
    size = len(xp)
    k = 0  # 记录小棱柱个数
    for prism in prisms:
        if prism is None or ('density' not in prism.props and dens is None):
            continue
        if dens is None:
            density = prism.props['density']
        else:
            density = dens
        x1, x2 = prism.x1, prism.x2
        y1, y2 = prism.y1, prism.y2
        z1, z2 = prism.z1, prism.z2
        kernel1D = numpy.zeros(size, dtype=numpy.float)
        _prism.gz(xp, yp, zp, x1, x2, y1, y2, z1, z2, density, res, kernel1D)
        kernel2d[:, k] = kernel1D
        k += 1

    res *= G * SI2MGAL
    kernel2d *= G * SI2MGAL
    return res, kernel2d


def _gxx(xp, yp, zp, res, kernel2d, prisms, dens=None):
    """
    Calculates the :math:`g_{xx}` gravity gradient tensor component.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> **DOWN**.

    .. note:: All input values in **SI** units(!) and output in **Eotvos**!

    Parameters:

    * xp, yp, zp : arrays
        Arrays with the x, y, and z coordinates of the computation points.
    * prisms : list of :class:`mesher.Prism`
        The density model used to calculate the gravitational effect.
        Prisms must have the property ``'density'``. Prisms that don't have
        this property will be ignored in the computations. Elements of *prisms*
        that are None will also be ignored. *prisms* can also be a
        :class:`mesher.PrismMesh`.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the prisms. Use this, e.g., for sensitivity matrix building.

    Returns:

    * res : array
        The field calculated on xp, yp, zp

    """
    if xp.shape != yp.shape or xp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same length!")
    size = len(xp)
    k = 0
    for prism in prisms:
        if prism is None or ('density' not in prism.props and dens is None):
            continue
        if dens is None:
            density = prism.props['density']
        else:
            density = dens
        x1, x2 = prism.x1, prism.x2
        y1, y2 = prism.y1, prism.y2
        z1, z2 = prism.z1, prism.z2
        kernel1D = numpy.zeros(size, dtype=numpy.float)
        _prism.gxx(xp, yp, zp, x1, x2, y1, y2, z1, z2, density, res, kernel1D)
        kernel2d[:, k] = kernel1D
        k += 1

    res *= G * SI2EOTVOS
    kernel2d *= G * SI2EOTVOS
    return res, kernel2d


def _gxy(xp, yp, zp, res, kernel2d, prisms, dens=None):
    """
    Calculates the :math:`g_{xy}` gravity gradient tensor component.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> **DOWN**.

    .. note:: All input values in **SI** units(!) and output in **Eotvos**!

    .. warning::

        This component has singularities when the computation
        point is aligned with the corners of the prism on the bottom side.
        In these cases, the computation point slightly to avoid these
        singularities. Unfortunately, this means that the result will not be as
        accurate **on those points**.

    Parameters:

    * xp, yp, zp : arrays
        Arrays with the x, y, and z coordinates of the computation points.
    * prisms : list of :class:`mesher.Prism`
        The density model used to calculate the gravitational effect.
        Prisms must have the property ``'density'``. Prisms that don't have
        this property will be ignored in the computations. Elements of *prisms*
        that are None will also be ignored. *prisms* can also be a
        :class:`mesher.PrismMesh`.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the prisms. Use this, e.g., for sensitivity matrix building.

    Returns:

    * res : array
        The field calculated on xp, yp, zp

    """
    if xp.shape != yp.shape or xp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same length!")
    size = len(xp)
    k = 0
    for prism in prisms:
        if prism is None or ('density' not in prism.props and dens is None):
            continue
        if dens is None:
            density = prism.props['density']
        else:
            density = dens
        x1, x2 = prism.x1, prism.x2
        y1, y2 = prism.y1, prism.y2
        z1, z2 = prism.z1, prism.z2
        kernel1D = numpy.zeros(size, dtype=numpy.float)
        _prism.gxy(xp, yp, zp, x1, x2, y1, y2, z1, z2, density, res, kernel1D)
        kernel2d[:, k] = kernel1D
        k += 1

    res *= G * SI2EOTVOS
    kernel2d *= G * SI2EOTVOS

    return res, kernel2d


def _gxz(xp, yp, zp, res, kernel2d, prisms, dens=None):
    """
    Calculates the :math:`g_{xz}` gravity gradient tensor component.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> **DOWN**.

    .. note:: All input values in **SI** units(!) and output in **Eotvos**!

    .. warning::

        This component has singularities when the computation
        point is aligned with the corners of the prism on the east side.
        In these cases, the computation point slightly to avoid these
        singularities. Unfortunately, this means that the result will not be as
        accurate **on those points**.

    Parameters:

    * xp, yp, zp : arrays
        Arrays with the x, y, and z coordinates of the computation points.
    * prisms : list of :class:`mesher.Prism`
        The density model used to calculate the gravitational effect.
        Prisms must have the property ``'density'``. Prisms that don't have
        this property will be ignored in the computations. Elements of *prisms*
        that are None will also be ignored. *prisms* can also be a
        :class:`mesher.PrismMesh`.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the prisms. Use this, e.g., for sensitivity matrix building.

    Returns:

    * res : array
        The field calculated on xp, yp, zp

    """
    if xp.shape != yp.shape or xp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same length!")
    size = len(xp)
    k = 0
    for prism in prisms:
        if prism is None or ('density' not in prism.props and dens is None):
            continue
        if dens is None:
            density = prism.props['density']
        else:
            density = dens
        x1, x2 = prism.x1, prism.x2
        y1, y2 = prism.y1, prism.y2
        z1, z2 = prism.z1, prism.z2
        kernel1D = numpy.zeros(size, dtype=numpy.float)
        _prism.gxz(xp, yp, zp, x1, x2, y1, y2, z1, z2, density, res, kernel1D)
        kernel2d[:, k] = kernel1D
        k += 1

    res *= G * SI2EOTVOS
    kernel2d *= G * SI2EOTVOS

    return res, kernel2d


def _gyy(xp, yp, zp, res, kernel2d, prisms, dens=None):
    """
    Calculates the :math:`g_{yy}` gravity gradient tensor component.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> **DOWN**.

    .. note:: All input values in **SI** units(!) and output in **Eotvos**!

    Parameters:

    * xp, yp, zp : arrays
        Arrays with the x, y, and z coordinates of the computation points.
    * prisms : list of :class:`mesher.Prism`
        The density model used to calculate the gravitational effect.
        Prisms must have the property ``'density'``. Prisms that don't have
        this property will be ignored in the computations. Elements of *prisms*
        that are None will also be ignored. *prisms* can also be a
        :class:`mesher.PrismMesh`.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the prisms. Use this, e.g., for sensitivity matrix building.

    Returns:

    * res : array
        The field calculated on xp, yp, zp

    """
    if xp.shape != yp.shape or xp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same length!")
    size = len(xp)
    k = 0
    for prism in prisms:
        if prism is None or ('density' not in prism.props and dens is None):
            continue
        if dens is None:
            density = prism.props['density']
        else:
            density = dens
        x1, x2 = prism.x1, prism.x2
        y1, y2 = prism.y1, prism.y2
        z1, z2 = prism.z1, prism.z2
        kernel1D = numpy.zeros(size, dtype=numpy.float)
        _prism.gyy(xp, yp, zp, x1, x2, y1, y2, z1, z2, density, res, kernel1D)
        kernel2d[:, k] = kernel1D
        k += 1

    res *= G * SI2EOTVOS
    kernel2d *= G * SI2EOTVOS
    return res, kernel2d


def _gyz(xp, yp, zp, res, kernel2d, prisms, dens=None):
    """
    Calculates the :math:`g_{yz}` gravity gradient tensor component.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> **DOWN**.

    .. note:: All input values in **SI** units(!) and output in **Eotvos**!

    .. warning::

        This component has singularities when the computation
        point is aligned with the corners of the prism on the north side.
        In these cases, the computation point slightly to avoid these
        singularities. Unfortunately, this means that the result will not be as
        accurate **on those points**.

    Parameters:

    * xp, yp, zp : arrays
        Arrays with the x, y, and z coordinates of the computation points.
    * prisms : list of :class:`mesher.Prism`
        The density model used to calculate the gravitational effect.
        Prisms must have the property ``'density'``. Prisms that don't have
        this property will be ignored in the computations. Elements of *prisms*
        that are None will also be ignored. *prisms* can also be a
        :class:`mesher.PrismMesh`.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the prisms. Use this, e.g., for sensitivity matrix building.

    Returns:

    * res : array
        The field calculated on xp, yp, zp

    """
    if xp.shape != yp.shape or xp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same length!")
    size = len(xp)
    k = 0
    for prism in prisms:
        if prism is None or ('density' not in prism.props and dens is None):
            continue
        if dens is None:
            density = prism.props['density']
        else:
            density = dens
        x1, x2 = prism.x1, prism.x2
        y1, y2 = prism.y1, prism.y2
        z1, z2 = prism.z1, prism.z2
        kernel1D = numpy.zeros(size, dtype=numpy.float)
        _prism.gyz(xp, yp, zp, x1, x2, y1, y2, z1, z2, density, res, kernel1D)
        kernel2d[:, k] = kernel1D
        k += 1

    res *= G * SI2EOTVOS
    kernel2d *= G * SI2EOTVOS

    return res, kernel2d


def _gzz(xp, yp, zp, res, kernel2d, prisms, dens=None):
    """
    Calculates the :math:`g_{zz}` gravity gradient tensor component.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> **DOWN**.

    .. note:: All input values in **SI** units(!) and output in **Eotvos**!

    Parameters:

    * xp, yp, zp : arrays
        Arrays with the x, y, and z coordinates of the computation points.
    * prisms : list of :class:`mesher.Prism`
        The density model used to calculate the gravitational effect.
        Prisms must have the property ``'density'``. Prisms that don't have
        this property will be ignored in the computations. Elements of *prisms*
        that are None will also be ignored. *prisms* can also be a
        :class:`mesher.PrismMesh`.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the prisms. Use this, e.g., for sensitivity matrix building.

    Returns:

    * res : array
        The field calculated on xp, yp, zp

    """
    if xp.shape != yp.shape or xp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same length!")
    size = len(xp)
    k = 0
    for prism in prisms:
        if prism is None or ('density' not in prism.props and dens is None):
            continue
        if dens is None:
            density = prism.props['density']
        else:
            density = dens
        x1, x2 = prism.x1, prism.x2
        y1, y2 = prism.y1, prism.y2
        z1, z2 = prism.z1, prism.z2
        kernel1D = numpy.zeros(size, dtype=numpy.float)
        _prism.gzz(xp, yp, zp, x1, x2, y1, y2, z1, z2, density, res, kernel1D)
        kernel2d[:, k] = kernel1D
        k += 1

    res *= G * SI2EOTVOS
    kernel2d *= G * SI2EOTVOS

    return res, kernel2d


def _tf(xp, yp, zp, res, kernel2d, prisms, inc, dec, pmag=None):
    """
    Calculate the total-field magnetic anomaly of prisms.

    .. note:: Input units are SI. Output is in nT

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> Down.

    Parameters:

    * xp, yp, zp : arrays
        Arrays with the x, y, and z coordinates of the computation points.
    * prisms : list of :class:`mesher.Prism`
        The model used to calculate the total field anomaly.
        Prisms without the physical property ``'magnetization'`` will
        be ignored. *prisms* can also be a :class:`mesher.PrismMesh`.
    * inc : float
        The inclination of the regional field (in degrees)
    * dec : float
        The declination of the regional field (in degrees)
    * pmag : [mx, my, mz] or None
        A magnetization vector. If not None, will use this value instead of the
        ``'magnetization'`` property of the prisms. Use this, e.g., for
        sensitivity matrix building.

    Returns:

    * res : array
        The field calculated on xp, yp, zp

    """
    if xp.shape != yp.shape or xp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same length!")

    # Calculate the 3 components of the unit vector in the direction of the
    # regional field
    fx, fy, fz = utils.dircos(inc, dec)
    size = len(xp)
    k = 0
    if pmag is not None:
        if isinstance(pmag, float) or isinstance(pmag, int):
            mx, my, mz = pmag * fx, pmag * fy, pmag * fz
        else:
            mx, my, mz = pmag
    for prism in prisms:
        if (prism is None or
                ('magnetization' not in prism.props and pmag is None)):
            continue
        if pmag is None:
            mag = prism.props['magnetization']
            if isinstance(mag, float) or isinstance(mag, int):
                mx, my, mz = mag * fx, mag * fy, mag * fz
            else:
                mx, my, mz = mag
        x1, x2 = prism.x1, prism.x2
        y1, y2 = prism.y1, prism.y2
        z1, z2 = prism.z1, prism.z2
        kernel1D = numpy.zeros(size, dtype=numpy.float)
        _prism.tf(xp, yp, zp, x1, x2, y1, y2, z1, z2, mx, my, mz, fx, fy, fz,
                  res, kernel1D)
        kernel2d[:, k] = kernel1D
        k += 1

    res *= CM * T2NT
    kernel2d *= CM * T2NT

    return res, kernel2d


def _bx(xp, yp, zp, prisms, pmag=None):
    """
    Calculates the x component of the magnetic induction produced by
    rectangular prisms.

    .. note:: Input units are SI. Output is in nT

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the anomaly will be calculated
    * prisms : list of :class:`fatiando.mesher.Prism`
        The model used to calculate the total field anomaly.
        Prisms without the physical property ``'magnetization'`` will
        be ignored. The ``'magnetization'`` must be a vector.
    * pmag : [mx, my, mz] or None
        A magnetization vector. If not None, will use this value instead of the
        ``'magnetization'`` property of the prisms. Use this, e.g., for
        sensitivity matrix building.

    Returns:

    * bx: array
        The x component of the magnetic induction

    """
    if xp.shape != yp.shape or xp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    if pmag is not None:
        mx, my, mz = pmag
    size = len(xp)
    res = numpy.zeros(size, dtype=numpy.float)
    for prism in prisms:
        if (prism is None or
                ('magnetization' not in prism.props and pmag is None)):
            continue
        if pmag is None:
            mx, my, mz = prism.props['magnetization']
        x1, x2 = prism.x1, prism.x2
        y1, y2 = prism.y1, prism.y2
        z1, z2 = prism.z1, prism.z2
        _prism.bx(xp, yp, zp, x1, x2, y1, y2, z1, z2, mx, my, mz, res)
    res *= CM * T2NT
    return res


def _by(xp, yp, zp, prisms, pmag=None):
    """
    Calculates the y component of the magnetic induction produced by
    rectangular prisms.

    .. note:: Input units are SI. Output is in nT

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the anomaly will be calculated
    * prisms : list of :class:`fatiando.mesher.Prism`
        The model used to calculate the total field anomaly.
        Prisms without the physical property ``'magnetization'`` will
        be ignored. The ``'magnetization'`` must be a vector.
    * pmag : [mx, my, mz] or None
        A magnetization vector. If not None, will use this value instead of the
        ``'magnetization'`` property of the prisms. Use this, e.g., for
        sensitivity matrix building.

    Returns:

    * by: array
        The y component of the magnetic induction

    """
    if xp.shape != yp.shape or xp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    if pmag is not None:
        mx, my, mz = pmag
    size = len(xp)
    res = numpy.zeros(size, dtype=numpy.float)
    for prism in prisms:
        if (prism is None or
                ('magnetization' not in prism.props and pmag is None)):
            continue
        if pmag is None:
            mx, my, mz = prism.props['magnetization']
        x1, x2 = prism.x1, prism.x2
        y1, y2 = prism.y1, prism.y2
        z1, z2 = prism.z1, prism.z2
        _prism.by(xp, yp, zp, x1, x2, y1, y2, z1, z2, mx, my, mz, res)
    res *= CM * T2NT
    return res


def _bz(xp, yp, zp, prisms, pmag=None):
    """
    Calculates the z component of the magnetic induction produced by
    rectangular prisms.

    .. note:: Input units are SI. Output is in nT

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the anomaly will be calculated
    * prisms : list of :class:`fatiando.mesher.Prism`
        The model used to calculate the total field anomaly.
        Prisms without the physical property ``'magnetization'`` will
        be ignored. The ``'magnetization'`` must be a vector.
    * pmag : [mx, my, mz] or None
        A magnetization vector. If not None, will use this value instead of the
        ``'magnetization'`` property of the prisms. Use this, e.g., for
        sensitivity matrix building.

    Returns:

    * bz: array
        The z component of the magnetic induction

    """
    if xp.shape != yp.shape or xp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    if pmag is not None:
        mx, my, mz = pmag
    size = len(xp)
    res = numpy.zeros(size, dtype=numpy.float)
    for prism in prisms:
        if (prism is None or
                ('magnetization' not in prism.props and pmag is None)):
            continue
        if pmag is None:
            mx, my, mz = prism.props['magnetization']
        x1, x2 = prism.x1, prism.x2
        y1, y2 = prism.y1, prism.y2
        z1, z2 = prism.z1, prism.z2
        _prism.bz(xp, yp, zp, x1, x2, y1, y2, z1, z2, mx, my, mz, res)
    res *= CM * T2NT
    return res


# --------------使用multiprocessing计算
# ------重力场
def potential(xp, yp, zp, prisms, dens=None, njobs=1, pool=None):
    """
    Calculate the radial component of the gravitational attraction.
    """
    field = 'potential'
    result, kernel2d = _dispatcher_gravity(field, xp, yp, zp, prisms,
                                   dens=dens, njobs=njobs, pool=pool)
    return result, kernel2d

def geoid(xp, yp, zp, prisms, dens=None, njobs=1, pool=None):
    """
    Calculate the radial component of the gravitational attraction.
    """
    field = 'geoid'
    result, kernel2d = _dispatcher_gravity(field, xp, yp, zp, prisms,
                                   dens=dens, njobs=njobs, pool=pool)
    return result, kernel2d

def gx(xp, yp, zp, prisms, dens=None, njobs=1, pool=None):
    """
    Calculate the radial component of the gravitational attraction.
    """
    field = 'gx'
    result, kernel2d = _dispatcher_gravity(field, xp, yp, zp, prisms,
                                   dens=dens, njobs=njobs, pool=pool)
    return result, kernel2d

def gy(xp, yp, zp, prisms, dens=None, njobs=1, pool=None):
    """
    Calculate the radial component of the gravitational attraction.
    """
    field = 'gy'
    result, kernel2d = _dispatcher_gravity(field, xp, yp, zp, prisms,
                                   dens=dens, njobs=njobs, pool=pool)
    return result, kernel2d

def gz(xp, yp, zp, prisms, dens=None, njobs=1, pool=None):
    """
    Calculate the radial component of the gravitational attraction.
    """
    field = 'gz'
    result, kernel2d = _dispatcher_gravity(field, xp, yp, zp, prisms,
                                   dens=dens, njobs=njobs, pool=pool)
    return result, kernel2d

def gxx(xp, yp, zp, prisms, dens=None, njobs=1, pool=None):
    """
    Calculate the radial component of the gravitational attraction.
    """
    field = 'gxx'
    result, kernel2d = _dispatcher_gravity(field, xp, yp, zp, prisms,
                                   dens=dens, njobs=njobs, pool=pool)
    return result, kernel2d

def gxy(xp, yp, zp, prisms, dens=None, njobs=1, pool=None):
    """
    Calculate the radial component of the gravitational attraction.
    """
    field = 'gxy'
    result, kernel2d = _dispatcher_gravity(field, xp, yp, zp, prisms,
                                   dens=dens, njobs=njobs, pool=pool)
    return result, kernel2d

def gxz(xp, yp, zp, prisms, dens=None, njobs=1, pool=None):
    """
    Calculate the radial component of the gravitational attraction.
    """
    field = 'gxz'
    result, kernel2d = _dispatcher_gravity(field, xp, yp, zp, prisms,
                                   dens=dens, njobs=njobs, pool=pool)
    return result, kernel2d

def gyy(xp, yp, zp, prisms, dens=None, njobs=1, pool=None):
    """
    Calculate the radial component of the gravitational attraction.
    """
    field = 'gyy'
    result, kernel2d = _dispatcher_gravity(field, xp, yp, zp, prisms,
                                   dens=dens, njobs=njobs, pool=pool)
    return result, kernel2d

def gyz(xp, yp, zp, prisms, dens=None, njobs=1, pool=None):
    """
    Calculate the radial component of the gravitational attraction.
    """
    field = 'gyz'
    result, kernel2d = _dispatcher_gravity(field, xp, yp, zp, prisms,
                                   dens=dens, njobs=njobs, pool=pool)
    return result, kernel2d

def gzz(xp, yp, zp, prisms, dens=None, njobs=1, pool=None):
    """
    Calculate the radial component of the gravitational attraction.
    """
    field = 'gzz'
    result, kernel2d = _dispatcher_gravity(field, xp, yp, zp, prisms,
                                   dens=dens, njobs=njobs, pool=pool)
    return result, kernel2d

#--------磁场
def tf(xp, yp, zp, prisms, inc, dec, pmag=None, njobs=1, pool=None):
    """
    Calculate the radial component of the gravitational attraction.
    """
    field = 'tf'
    result, kernel2d = _dispatcher_magnetic(field, xp, yp, zp, prisms, inc, dec,
                                   pmag=pmag, njobs=njobs, pool=pool)
    return result, kernel2d

#--------并行计算
#--------重力
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

def _dispatcher_gravity(field, xp, yp, zp, prisms, **kwargs):
    """
    Dispatch the computation of *field* to the appropriate function.
    Returns:
    * result : 1d-array
    * kernel2d : 2d-array
    """
    dens = kwargs['dens']
    njobs = kwargs.get('njobs', 1)
    pool = kwargs.get('pool', None)
    # ------ 计算数组维度
    size = len(xp)  # size是观测点个数
    # 有效小棱柱的数量: prisms.size（总共）- prismcarve
    prismcarve = 0
    for prism in prisms:
        if prism is None or ('density' not in prism.props and dens is None):
            prismcarve += 1
    # 定义数组
    result = numpy.zeros(size, dtype=numpy.float)
    kernel2d = numpy.zeros([size, prisms.size-prismcarve], dtype=numpy.float)
    # 多进程
    if njobs > 1 and pool is None:
        pool = multiprocessing.Pool(njobs)
        created_pool = True
    else:
        created_pool = False
    if pool is None:
        # 不使用多进程，则result, kernel2d一起输出
         _forward_gravity([xp, yp, zp, result, kernel2d, prisms, dens, field])
    else:
        # 使用多进程，result, kernel2d分别输出
        chunks = _split_arrays(arrays=[xp, yp, zp, result],
                               arrays2D=[kernel2d],
                               extra_args=[prisms, dens, field],
                               nparts=njobs)
        result = np.hstack(pool.map(_forward_gravity_result, chunks))
        kernel2d = np.vstack(pool.map(_forward_gravity_kernel2d, chunks))
    if created_pool:
        pool.close()

    return result, kernel2d

def _forward_gravity(args):
    xp, yp, zp, result, kernel2d, prisms, dens, field = args
    if field == 'potential':
        _potential(xp, yp, zp, result, kernel2d, prisms, dens)
    elif field == 'geoid':
        _geoid(xp, yp, zp, result, kernel2d, prisms, dens)
    elif field == 'gx':
        _gx(xp, yp, zp, result, kernel2d, prisms, dens)
    elif field == 'gy':
        _gy(xp, yp, zp, result, kernel2d, prisms, dens)
    elif field == 'gz':
        _gz(xp, yp, zp, result, kernel2d, prisms, dens)
    elif field == 'gxx':
        _gxx(xp, yp, zp, result, kernel2d, prisms, dens)
    elif field == 'gxy':
        _gxy(xp, yp, zp, result, kernel2d, prisms, dens)
    elif field == 'gxz':
        _gxz(xp, yp, zp, result, kernel2d, prisms, dens)
    elif field == 'gyy':
        _gyy(xp, yp, zp, result, kernel2d, prisms, dens)
    elif field == 'gyz':
        _gyz(xp, yp, zp, result, kernel2d, prisms, dens)
    elif field == 'gzz':
        _gzz(xp, yp, zp, result, kernel2d, prisms, dens)
    else:
        print("Please choose right field")

    return result, kernel2d

def _forward_gravity_result(args):
    '''
    Args:
        args:
    Returns: result

    '''
    xp, yp, zp, result, kernel2d, prisms, dens, field = args
    if field == 'potential':
        _potential(xp, yp, zp, result, kernel2d, prisms, dens)
    elif field == 'geoid':
        _geoid(xp, yp, zp, result, kernel2d, prisms, dens)
    elif field == 'gx':
        _gx(xp, yp, zp, result, kernel2d, prisms, dens)
    elif field == 'gy':
        _gy(xp, yp, zp, result, kernel2d, prisms, dens)
    elif field == 'gz':
        _gz(xp, yp, zp, result, kernel2d, prisms, dens)
    elif field == 'gxx':
        _gxx(xp, yp, zp, result, kernel2d, prisms, dens)
    elif field == 'gxy':
        _gxy(xp, yp, zp, result, kernel2d, prisms, dens)
    elif field == 'gxz':
        _gxz(xp, yp, zp, result, kernel2d, prisms, dens)
    elif field == 'gyy':
        _gyy(xp, yp, zp, result, kernel2d, prisms, dens)
    elif field == 'gyz':
        _gyz(xp, yp, zp, result, kernel2d, prisms, dens)
    elif field == 'gzz':
        _gzz(xp, yp, zp, result, kernel2d, prisms, dens)
    else:
        print("Please choose right field")

    return result

def _forward_gravity_kernel2d(args):
    '''
    Args:
        args:
    Returns: kernel2d
    '''
    xp, yp, zp, result, kernel2d, prisms, dens, field = args
    if field == 'potential':
        _potential(xp, yp, zp, result, kernel2d, prisms, dens)
    elif field == 'geoid':
        _geoid(xp, yp, zp, result, kernel2d, prisms, dens)
    elif field == 'gx':
        _gx(xp, yp, zp, result, kernel2d, prisms, dens)
    elif field == 'gy':
        _gy(xp, yp, zp, result, kernel2d, prisms, dens)
    elif field == 'gz':
        _gz(xp, yp, zp, result, kernel2d, prisms, dens)
    elif field == 'gxx':
        _gxx(xp, yp, zp, result, kernel2d, prisms, dens)
    elif field == 'gxy':
        _gxy(xp, yp, zp, result, kernel2d, prisms, dens)
    elif field == 'gxz':
        _gxz(xp, yp, zp, result, kernel2d, prisms, dens)
    elif field == 'gyy':
        _gyy(xp, yp, zp, result, kernel2d, prisms, dens)
    elif field == 'gyz':
        _gyz(xp, yp, zp, result, kernel2d, prisms, dens)
    elif field == 'gzz':
        _gzz(xp, yp, zp, result, kernel2d, prisms, dens)
    else:
        print("Please choose right field")

    return kernel2d


#--------磁场
def _dispatcher_magnetic(field, xp, yp, zp, prisms, inc, dec, **kwargs):
    """
    Dispatch the computation of *field* to the appropriate function.
    Returns:
    * result : 1d-array
    * kernel2d : 2d-array
    """
    pmag = kwargs['pmag']
    njobs = kwargs.get('njobs', 1)
    pool = kwargs.get('pool', None)
    # ------ 计算数组维度
    size = len(xp)  # size是观测点个数
    # 有效小棱柱的数量: prisms.size（总共）- prismcarve
    prismcarve = 0
    for prism in prisms:
        if prism is None or ('magnetization' not in prism.props and pmag is None):
            prismcarve += 1
    # 定义数组
    result = numpy.zeros(size, dtype=numpy.float)
    kernel2d = numpy.zeros([size, prisms.size - prismcarve], dtype=numpy.float)
    # 多进程
    if njobs > 1 and pool is None:
        pool = multiprocessing.Pool(njobs)
        created_pool = True
    else:
        created_pool = False
    if pool is None:
        # 不使用多进程，则result, kernel2d一起输出
         _forward_magnetic([xp, yp, zp, result, kernel2d, prisms, pmag, inc, dec, field])
    else:
        # 使用多进程，result, kernel2d分别输出
        chunks = _split_arrays(arrays=[xp, yp, zp, result],
                               arrays2D=[kernel2d],
                               extra_args=[prisms, pmag, inc, dec, field],
                               nparts=njobs)
        result = np.hstack(pool.map(_forward_magnetic_result, chunks))
        kernel2d = np.vstack(pool.map(_forward_magnetic_kernel2d, chunks))
    if created_pool:
        pool.close()

    return result, kernel2d

def _forward_magnetic(args):
    xp, yp, zp, result, kernel2d, prisms, pmag, inc, dec, field = args
    if field == 'tf':
        _tf(xp, yp, zp, result, kernel2d, prisms, inc, dec, pmag)
    else:
        print("Please choose right field")

    return result, kernel2d

def _forward_magnetic_result(args):
    xp, yp, zp, result, kernel2d, prisms, pmag, inc, dec, field = args
    if field == 'tf':
        _tf(xp, yp, zp, result, kernel2d, prisms, inc, dec, pmag)
    else:
        print("Please choose right field")

    return result

def _forward_magnetic_kernel2d(args):
    xp, yp, zp, result, kernel2d, prisms, pmag, inc, dec, field = args
    if field == 'tf':
        _tf(xp, yp, zp, result, kernel2d, prisms, inc, dec, pmag)
    else:
        print("Please choose right field")

    return kernel2d