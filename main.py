#!/usr/bin/env python3
"""
This code is used to calculate of the Urban Heat Island effect as described in:

Valmassoi and Keller (2021), Representing the Urban Heat Island in Gridded Datasets, Advances in Science and Research.
DOI:

This program is a free software distributed under the terms of the GNU General Public License as published by
the Free Software Foundation, version 3 (GNU-GPLv3).

You can redistribute and/or modify by citing the mentioned publication, but WITHOUT ANY WARRANTY; without even the
implied warranty of  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

For a description of the UHI methods, refer to Valmassoi and Keller (2021).
"""


from uhi_calculation import calc_uhi, quick_uhi_plot
from netCDF4 import Dataset
import numpy as np

datafile = Dataset('') # file containing the temperature field
extfile = Dataset('') # file containing the invariant data
gridfile = Dataset('') # file containing the grid field

# surface temperature (time, cell) - unstructured grid
temperature_time_series = datafile.variables['T_2M'][:, :] - 273.15
lons = gridfile.variables["clon"][:]  # longitude (degrees) 1D
lats = gridfile.variables["clat"][:]  # latitude (degrees) 1D
topography = extfile.variables['topography_c'][:]  # topography: 1D
rural_land_use = extfile.variables['LU_CLASS_FRACTION'][1, :]  # rural land use: 1D
urban_land_use = extfile.variables['LU_CLASS_FRACTION'][18, :]  # urban use: 1D

# first time-step
temperature = temperature_time_series[0,:]

# coordinates for the methods
min_urb_coord_box = [51.5, 5.8]
max_urb_coord_box = [52, 6.2]
min_rur_coord_box = [51, 5]
max_rur_coord_box = [53, 7]

# example 1: UHI field with M7
method = 7 # method number
uhi_m7 = calc_uhi(method, temperature, lons, lats, topography, rural_land_use, urban_land_use, min_urb_coord_box,
                  max_urb_coord_box, min_rur_coord_box, max_rur_coord_box)

# example 2: UHI time series M1
uhi = np.zeros(temperature_time_series.shape)
method = 1 # method number
for i in np.arange(0, uhi.shape[0]):
    uhi[i] = calc_uhi(method, temperature_time_series[i, :], lons, lats, topography, rural_land_use, urban_land_use,
                      min_urb_coord_box,
                      max_urb_coord_box, min_rur_coord_box, max_rur_coord_box, rur_obs=28)

# quick plotting routine to check the results
# coordinate limits
map_limits_min = [50.5, 5]
map_limits_max = [54, 7.8]
quick_uhi_plot(uhi_m7, lons, lats, map_limits_min, map_limits_max)
