"""
Defines geometric primitives like prism, tesseroid, etc.

References
++++++++++

Uieda, L., V. Barbosa, and C. Braitenberg (2016), Tesseroids: Forward-modeling
gravitational fields in spherical coordinates, Geophysics, F41-F48,
doi:10.1190/geo2015-0204.1


"""

import copy as cp
import numpy as np


class GeometricElement(object):
    """
    Base class for all geometric elements.
    """

    def __init__(self, props):
        self.props = {}
        if props is not None:
            for p in props:
                self.props[p] = props[p]

    def addprop(self, prop, value):
        """
        Add a physical property to this geometric element.

        If it already has the property, the given value will overwrite the
        existing one.

        Parameters:

        * prop : str
            Name of the physical property.
        * value : float
            The value of this physical property.

        """
        self.props[prop] = value

    def copy(self):
        """ Return a deep copy of the current instance."""
        return cp.deepcopy(self)


class Prism(GeometricElement):
    """
    A 3D right rectangular prism.

    .. note:: The coordinate system used is x -> North, y -> East and z -> Down

    Parameters:


    * x1, x2 : float
        South and north borders of the prism
    * y1, y2 : float
        West and east borders of the prism
    * z1, z2 : float
        Top and bottom of the prism
    * props : dict
        Physical properties assigned to the prism.
        Ex: ``props={'density':10, 'magnetization':10000}``
    """

    def __init__(self, x1, x2, y1, y2, z1, z2, props=None):
        super().__init__(props)
        self.x1 = float(x1)
        self.x2 = float(x2)
        self.y1 = float(y1)
        self.y2 = float(y2)
        self.z1 = float(z1)
        self.z2 = float(z2)

    def __str__(self):
        """Return a string representation of the prism."""
        names = [('x1', self.x1), ('x2', self.x2), ('y1', self.y1),
                 ('y2', self.y2), ('z1', self.z1), ('z2', self.z2)]
        names.extend((p, self.props[p]) for p in sorted(self.props))
        return ' | '.join('%s:%g' % (n, v) for n, v in names)

    def get_bounds(self):
        """
        Get the bounding box of the prism (i.e., the borders of the prism).
        Returns:
        * bounds : list
            ``[x1, x2, y1, y2, z1, z2]``, the bounds of the prism
        """
        return [self.x1, self.x2, self.y1, self.y2, self.z1, self.z2]

    def center(self):
        """
        Return the coordinates of the center of the prism.
        Returns:
        * coords : list = [xc, yc, zc]
            Coordinates of the center
        """
        xc = 0.5 * (self.x1 + self.x2)
        yc = 0.5 * (self.y1 + self.y2)
        zc = 0.5 * (self.z1 + self.z2)
        return np.array([xc, yc, zc])


class Tesseroid(GeometricElement):
    """
    A tesseroid (spherical prism).

    Parameters:

    * w, e : float
        West and east borders of the tesseroid in decimal degrees
    * s, n : float
        South and north borders of the tesseroid in decimal degrees
    * top, bottom : float
        Bottom and top of the tesseroid with respect to the mean earth radius
        in meters. Ex: if the top is 100 meters above the mean earth radius,
        ``top=100``, if 100 meters below ``top=-100``.
    * props : dict
        Physical properties assigned to the tesseroid.
        Ex: ``props={'density':10, 'magnetization':10000}``

    Examples:

    """

    def __init__(self, w, e, s, n, top, bottom, props=None):
        super().__init__(props)
        self.w = float(w)
        self.e = float(e)
        self.s = float(s)
        self.n = float(n)
        self.bottom = float(bottom)
        self.top = float(top)

    def __str__(self):
        """Return a string representation of the tesseroid."""
        names = [('w', self.w), ('e', self.e), ('s', self.s),
                 ('n', self.n), ('top', self.top), ('bottom', self.bottom)]
        names.extend((p, self.props[p]) for p in sorted(self.props))
        return ' | '.join('%s:%g' % (n, v) for n, v in names)

    def get_bounds(self):
        """
        Get the bounding box of the tesseroid (i.e., the borders).
        Returns:
        * bounds : list
            ``[w, e, s, n, top, bottom]``, the bounds of the tesseroid
        """
        return [self.w, self.e, self.s, self.n, self.top, self.bottom]

    def half(self, lon=True, lat=True, r=True):
        """
        Divide the tesseroid in 2 halfs for each dimension (total 8)

        The smaller tesseroids will share the large one's props.

        Parameters:
        * lon, lat, r : True or False
            Dimensions along which the tesseroid will be split in half.
        Returns:
        * tesseroids : list
            A list of maximum 8 tesseroids that make up the larger one.
        """
        dlon = 0.5 * (self.e - self.w)
        dlat = 0.5 * (self.n - self.s)
        dh = 0.5 * (self.top - self.bottom)
        wests = [self.w, self.w + dlon]
        souths = [self.s, self.s + dlat]
        bottoms = [self.bottom, self.bottom + dh]
        if not lon:
            dlon *= 2
            wests.pop()
        if not lat:
            dlat *= 2
            souths.pop()
        if not r:
            dh *= 2
            bottoms.pop()
        split = [
            Tesseroid(i, i + dlon, j, j + dlat, k + dh, k, props=self.props)
            for i in wests for j in souths for k in bottoms]
        return split

    def split(self, nlon, nlat, nh):
        """
        Split the tesseroid into smaller ones.
        The smaller tesseroids will share the large one's props.
        Parameters:
        * nlon, nlat, nh : int
            The number of sections to split in the longitudinal, latitudinal,
            and vertical dimensions
        Returns:
        * tesseroids : list
            A list of nlon*nlat*nh tesseroids that make up the larger one.
        """
        wests = np.linspace(self.w, self.e, nlon + 1)
        souths = np.linspace(self.s, self.n, nlat + 1)
        bottoms = np.linspace(self.bottom, self.top, nh + 1)
        dlon = wests[1] - wests[0]
        dlat = souths[1] - souths[0]
        dh = bottoms[1] - bottoms[0]
        tesseroids = [
            Tesseroid(i, i + dlon, j, j + dlat, k + dh, k, props=self.props)
            for i in wests[:-1] for j in souths[:-1] for k in bottoms[:-1]]
        return tesseroids


