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
To test the script with the sample data use test_main.py
"""


from uhi_calculation import calc_uhi, quick_uhi_plot
import numpy as np
import pandas as pd
import geopandas as gpd



lons =   # longitude (degrees) 1D
lats =   # latitude (degrees) 1D
topography =  # topography: 1D
rural_land_use =  # rural land use: 1D
urban_land_use =   # urban use: 1D
temperature =  # temperature: 1D

# coordinates for the methods
min_urb_coord_box = [, ] #lat, lon
max_urb_coord_box = [, ]#lat, lon
min_rur_coord_box = [, ]#lat, lon
max_rur_coord_box = [, ]#lat, lon


# example 1: UHI field with M7
method = 7 # method number
uhi_m7, baseline_temperature = calc_uhi(method, temperature, lons, lats, topography, rural_land_use, urban_land_use, min_urb_coord_box, max_urb_coord_box, min_rur_coord_box, max_rur_coord_box,seed)


# quick plotting routine to check the results
# coordinate limits
map_limits_min = [min(lats), min(lons)]
map_limits_max = [max(lats), max(lons)]
quick_uhi_plot(uhi_m7, lons, lats, map_limits_min, map_limits_max)


