"""
Holds all physical constants and unit conversions

All modules should import the constants from here!

All constants should be in SI, unless otherwise stated!

References
++++++++++

Uieda, L., V. Barbosa, and C. Braitenberg (2016), Tesseroids: Forward-modeling
gravitational fields in spherical coordinates, Geophysics, F41-F48,
doi:10.1190/geo2015-0204.1

----
"""


#: The default thermal diffusivity in :math:`m^2/s`
THERMAL_DIFFUSIVITY = 0.000001

#: The default thermal diffusivity but in :math:`m^2/year`
THERMAL_DIFFUSIVITY_YEAR = 31.5576

#: Conversion factor from SI units to Eotvos: :math:`1/s^2 = 10^9\ Eotvos`
SI2EOTVOS = 1000000000.0

#: Conversion factor from SI units to mGal: :math:`1\ m/s^2 = 10^5\ mGal`
SI2MGAL = 100000.0

#: The gravitational constant in :math:`m^3 kg^{-1} s^{-1}`
Gs = 0.00000000006673 # G for spherical
#: The gravitational constant in :math:`cm^3 g^{-1} s^{-1}`
G = 0.00000006673  # G for prism

#: Proportionality constant used in the magnetic method in henry/m (SI)
CM = 10. ** (-7)

#: Conversion factor from tesla to nanotesla
# T2NT = 10. ** (9)  # 原始；nT
T2NT = 10. ** (6)  # 修改MuT
T2MuT = 10. ** (6)
#: The mean earth radius in meters
MEAN_EARTH_RADIUS = 6378137.0
MEAN_MOON_RADIUS = 1738000.0
#: Permeability of free space in :math:`N A^{-2}`
PERM_FREE_SPACE = 4 * 3.141592653589793115997963468544185161590576171875 * (10 ** -7)

# Gravitational acceleration g0 m/s^2
g0 = 9.80

