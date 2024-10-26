"""
Wrappers for :mod:`matplotlib` functions to facilitate plotting grids,
2D objects, etc.

This module loads all functions from :mod:`matplotlib.pyplot`, adds new
functions and overwrites some others (like :func:`vis.mpl.contour`,
:func:`vis.mpl.pcolor`, etc).


**Grids**

* :func:`vis.mpl.contour`
* :func:`vis.mpl.contourf`
* :func:`vis.mpl.pcolor`


**Basemap (map projections)**

* :func:`vis.mpl.basemap`
* :func:`vis.mpl.draw_geolines`
* :func:`vis.mpl.draw_countries`
* :func:`vis.mpl.draw_coastlines`

**Auxiliary**

* :func:`vis.mpl.m2km`

----
References
++++++++++

Uieda, L., V. Barbosa, and C. Braitenberg (2016), Tesseroids: Forward-modeling
gravitational fields in spherical coordinates, Geophysics, F41-F48,
doi:10.1190/geo2015-0204.1

"""

import warnings
import numpy
from matplotlib import pyplot

import utils

warnings.filterwarnings("ignore")

# Dummy variable to lazy import the basemap toolkit (slow)
Basemap = None


def draw_geolines(area, dlon, dlat, basemap, linewidth=1):
    """
    Draw the parallels and meridians on a basemap plot.

    Parameters:

    * area : list
        ``[west, east, south, north]``, i.e., the area where the lines will
        be plotted
    * dlon, dlat : float
        The spacing between the lines in the longitude and latitude directions,
        respectively (in decimal degrees)
    * basemap : mpl_toolkits.basemap.Basemap
        The basemap used for plotting (see :func:`vis.mpl.basemap`)
    * linewidth : float
        The width of the lines

    """
    west, east, south, north = area
    basemap.drawmeridians(numpy.arange(west, east, dlon), labels=[0, 0, 0, 1],
                          linewidth=linewidth)
    basemap.drawparallels(numpy.arange(south, north, dlat),
                          labels=[1, 0, 0, 0], linewidth=linewidth)


def draw_countries(basemap, linewidth=1, style='dashed'):
    """
    Draw the country borders using the given basemap.

    Parameters:

    * basemap : mpl_toolkits.basemap.Basemap
        The basemap used for plotting (see :func:`vis.mpl.basemap`)
    * linewidth : float
        The width of the lines
    * style : str
        The style of the lines. Can be: 'solid', 'dashed', 'dashdot' or
        'dotted'

    """
    lines = basemap.drawcountries(linewidth=linewidth)
    lines.set_linestyles(style)


def draw_coastlines(basemap, linewidth=1, style='solid'):
    """
    Draw the coastlines using the given basemap.

    Parameters:

    * basemap : mpl_toolkits.basemap.Basemap
        The basemap used for plotting (see :func:`vis.mpl.basemap`)
    * linewidth : float
        The width of the lines
    * style : str
        The style of the lines. Can be: 'solid', 'dashed', 'dashdot' or
        'dotted'

    """
    lines = basemap.drawcoastlines(linewidth=linewidth)
    lines.set_linestyles(style)


def basemap(area, projection, resolution='c'):
    """
    Make a basemap to use when plotting with map projections.

    Uses the matplotlib basemap toolkit.

    Parameters:

    * area : list
        ``[west, east, south, north]``, i.e., the area of the data that is
        going to be plotted
    * projection : str
        The name of the projection you want to use. Choose from:

        * 'ortho': Orthographic
        * 'geos': Geostationary
        * 'robin': Robinson
        * 'cass': Cassini
        * 'merc': Mercator
        * 'poly': Polyconic
        * 'lcc': Lambert Conformal
        * 'stere': Stereographic

    * resolution : str
        The resolution for the coastlines. Can be 'c' for crude, 'l' for low,
        'i' for intermediate, 'h' for high

    Returns:

    * basemap : mpl_toolkits.basemap.Basemap
        The basemap

    """
    if projection not in ['ortho', 'aeqd', 'geos', 'robin', 'cass', 'merc',
                          'poly', 'lcc', 'stere']:
        raise ValueError("Unsuported projection '%s'" % (projection))
    global Basemap
    if Basemap is None:
        try:
            from mpl_toolkits.basemap import Basemap
        except ImportError:
            raise
    west, east, south, north = area
    lon_0 = 0.5 * (east + west)
    lat_0 = 0.5 * (north + south)
    if projection == 'ortho':
        bm = Basemap(projection=projection, lon_0=lon_0, lat_0=lat_0,
                     resolution=resolution)
    elif projection == 'geos' or projection == 'robin':
        bm = Basemap(projection=projection, lon_0=lon_0, resolution=resolution)
    elif (projection == 'cass' or
          projection == 'poly'):
        bm = Basemap(projection=projection, llcrnrlon=west, urcrnrlon=east,
                     llcrnrlat=south, urcrnrlat=north, lat_0=lat_0,
                     lon_0=lon_0, resolution=resolution)
    elif projection == 'merc':
        bm = Basemap(projection=projection, llcrnrlon=west, urcrnrlon=east,
                     llcrnrlat=south, urcrnrlat=north, lat_ts=lat_0,
                     resolution=resolution)
    elif projection == 'lcc':
        bm = Basemap(projection=projection, llcrnrlon=west, urcrnrlon=east,
                     llcrnrlat=south, urcrnrlat=north, lat_0=lat_0,
                     lon_0=lon_0, rsphere=(6378137.00, 6356752.3142),
                     lat_1=lat_0, resolution=resolution)
    elif projection == 'stere':
        bm = Basemap(projection=projection, llcrnrlon=west, urcrnrlon=east,
                     llcrnrlat=south, urcrnrlat=north, lat_0=lat_0,
                     lon_0=lon_0, lat_ts=lat_0, resolution=resolution)
    return bm


def m2km(axis=None):
    """
    Convert the x and y tick labels from meters to kilometers.

    Parameters:

    * axis : matplotlib axis instance
        The plot.

    .. tip:: Use ``vis.gca()`` to get the current axis. Or the value
        returned by ``vis.subplot`` or ``matplotlib.pyplot.subplot``.

    """
    if axis is None:
        axis = pyplot.gca()
    axis.set_xticklabels(['%g' % (0.001 * l) for l in axis.get_xticks()])
    axis.set_yticklabels(['%g' % (0.001 * l) for l in axis.get_yticks()])


def layers(thickness, values, style='-k', z0=0., linewidth=1, label=None,
           **kwargs):
    """
    Plot a series of layers and values associated to each layer.

    Parameters:

    * thickness : list
        The thickness of each layer in order of increasing depth
    * values : list
        The value associated with each layer in order of increasing
        depth
    * style : str
        String with the color and line style (as in matplotlib.pyplot.plot)
    * z0 : float
        The depth of the top of the first layer
    * linewidth : float
        Line width
    * label : str
        label associated with the square.

    Returns:

    * axes : ``matplitlib.axes``
        The axes element of the plot

    """
    if len(thickness) != len(values):
        raise ValueError("thickness and values must have same length")
    nlayers = len(thickness)
    interfaces = [z0 + sum(thickness[:i]) for i in range(nlayers + 1)]
    ys = [interfaces[0]]
    for y in interfaces[1:-1]:
        ys.append(y)
        ys.append(y)
    ys.append(interfaces[-1])
    xs = []
    for x in values:
        xs.append(x)
        xs.append(x)
    kwargs['linewidth'] = linewidth
    if label is not None:
        kwargs['label'] = label
    plot, = pyplot.plot(xs, ys, style, **kwargs)
    return plot


def contour(x, y, v, shape, levels, interp=False, extrapolate=False, color='k',
            label=None, clabel=True, style='solid', linewidth=1.0,
            basemap=None):
    """
    Make a contour plot of the data.

    Parameters:

    * x, y : array
        Arrays with the x and y coordinates of the grid points. If the data is
        on a regular grid, then assume x varies first (ie, inner loop), then y.
    * v : array
        The scalar value assigned to the grid points.
    * shape : tuple = (ny, nx)
        Shape of the regular grid.
        If interpolation is not False, then will use *shape* to grid the data.
    * levels : int or list
        Number of contours to use or a list with the contour values.
    * interp : True or False
        Wether or not to interpolate before trying to plot. If data is not on
        regular grid, set to True!
    * extrapolate : True or False
        Wether or not to extrapolate the data when interp=True
    * color : str
        Color of the contour lines.
    * label : str
        String with the label of the contour that would show in a legend.
    * clabel : True or False
        Wether or not to print the numerical value of the contour lines
    * style : str
        The style of the contour lines. Can be ``'dashed'``, ``'solid'`` or
        ``'mixed'`` (solid lines for positive contours and dashed for negative)
    * linewidth : float
        Width of the contour lines
    * basemap : mpl_toolkits.basemap.Basemap
        If not None, will use this basemap for plotting with a map projection
        (see :func:`vis.mpl.basemap` for creating basemaps)

    Returns:

    * levels : list
        List with the values of the contour levels

    """
    if style not in ['solid', 'dashed', 'mixed']:
        raise ValueError("Invalid contour style %s" % (style))
    if x.shape != y.shape != v.shape:
        raise ValueError("Input arrays x, y, and v must have same shape!")
    if interp:
        x, y, v = utils.interp(x, y, v, shape, extrapolate=extrapolate)
    X = numpy.reshape(x, shape)
    Y = numpy.reshape(y, shape)
    V = numpy.reshape(v, shape)
    kwargs = dict(colors=color, picker=True)
    if basemap is None:
        ct_data = pyplot.contour(X, Y, V, levels, **kwargs)
        pyplot.xlim(X.min(), X.max())
        pyplot.ylim(Y.min(), Y.max())
    else:
        lon, lat = basemap(X, Y)
        ct_data = basemap.contour(lon, lat, V, levels, **kwargs)
    if clabel:
        ct_data.clabel(fmt='%g')
    if label is not None:
        ct_data.collections[0].set_label(label)
    if style != 'mixed':
        for c in ct_data.collections:
            c.set_linestyle(style)
    for c in ct_data.collections:
        c.set_linewidth(linewidth)
    return ct_data.levels


def contourf(x, y, v, shape, levels, interp=False, extrapolate=False,
             vmin=None, vmax=None, cmap=pyplot.cm.jet, basemap=None):
    """
    Make a filled contour plot of the data.
    x先变
    Parameters:

    * x, y : array
        Arrays with the x and y coordinates of the grid points. If the data is
        on a regular grid, then assume x varies first (ie, inner loop), then y.
    * v : array
        The scalar value assigned to the grid points.
    * shape : tuple = (ny, nx)
        Shape of the regular grid.
        If interpolation is not False, then will use *shape* to grid the data.
    * levels : int or list
        Number of contours to use or a list with the contour values.
    * interp : True or False
        Wether or not to interpolate before trying to plot. If data is not on
        regular grid, set to True!
    * extrapolate : True or False
        Wether or not to extrapolate the data when interp=True
    * vmin, vmax
        Saturation values of the colorbar. If provided, will overwrite what is
        set by *levels*.
    * cmap : colormap
        Color map to be used. (see pyplot.cm module)
    * basemap : mpl_toolkits.basemap.Basemap
        If not None, will use this basemap for plotting with a map projection
        (see :func:`vis.mpl.basemap` for creating basemaps)

    Returns:

    * levels : list
        List with the values of the contour levels

    """
    if x.shape != y.shape != v.shape:
        raise ValueError("Input arrays x, y, and v must have same shape!")
    if interp:
        x, y, v = utils.interp(x, y, v, shape, extrapolate=extrapolate)
    X = numpy.reshape(x, shape)
    Y = numpy.reshape(y, shape)
    V = numpy.reshape(v, shape)
    kwargs = dict(vmin=vmin, vmax=vmax, cmap=cmap, picker=True)
    if basemap is None:
        ct_data = pyplot.contourf(X, Y, V, levels, **kwargs)
        pyplot.xlim(X.min(), X.max())
        pyplot.ylim(Y.min(), Y.max())
    else:
        lon, lat = basemap(X, Y)
        ct_data = basemap.contourf(lon, lat, V, levels, **kwargs)
    return ct_data.levels


def pcolor(x, y, v, shape, interp=False, extrapolate=False, cmap=pyplot.cm.jet,
           vmin=None, vmax=None, basemap=None):
    """
    Make a pseudo-color plot of the data.

    Parameters:

    * x, y : array
        Arrays with the x and y coordinates of the grid points. If the data is
        on a regular grid, then assume x varies first (ie, inner loop), then y.
    * v : array
        The scalar value assigned to the grid points.
    * shape : tuple = (ny, nx)
        Shape of the regular grid.
        If interpolation is not False, then will use *shape* to grid the data.
    * interp : True or False
        Wether or not to interpolate before trying to plot. If data is not on
        regular grid, set to True!
    * extrapolate : True or False
        Wether or not to extrapolate the data when interp=True
    * cmap : colormap
        Color map to be used. (see pyplot.cm module)
    * vmin, vmax
        Saturation values of the colorbar.
    * basemap : mpl_toolkits.basemap.Basemap
        If not None, will use this basemap for plotting with a map projection
        (see :func:`vis.mpl.basemap` for creating basemaps)

    Returns:

    * axes : ``matplitlib.axes``
        The axes element of the plot

    """
    if x.shape != y.shape != v.shape:
        raise ValueError("Input arrays x, y, and v must have same shape!")
    if vmin is None:
        vmin = v.min()
    if vmax is None:
        vmax = v.max()
    if interp:
        x, y, v = utils.interp(x, y, v, shape, extrapolate=extrapolate)
    X = numpy.reshape(x, shape)
    Y = numpy.reshape(y, shape)
    V = numpy.reshape(v, shape)
    if basemap is None:
        plot = pyplot.pcolor(X, Y, V, cmap=cmap, vmin=vmin, vmax=vmax,
                             picker=True)
        pyplot.xlim(X.min(), X.max())
        pyplot.ylim(Y.min(), Y.max())
    else:
        lon, lat = basemap(X, Y)
        plot = basemap.pcolor(lon, lat, V, cmap=cmap, vmin=vmin, vmax=vmax,
                              picker=True)
    return plot



