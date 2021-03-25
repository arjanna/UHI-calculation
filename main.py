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
import numpy as np
import pandas as pd
import geopandas as gpd

df = pd.read_csv('dataset2.csv')

lons = df.lons.values  # longitude (degrees) 1D
lats = df.lats.values  # latitude (degrees) 1D
topography = df.topo.values # topography: 1D
rural_land_use = df.rural.values  # rural land use: 1D
urban_land_use = df.urban.values  # urban use: 1D
temperature = df.temperature.values # temperature: 1D

# coordinates for the methods
min_urb_coord_box = [51.5, 5.8]
max_urb_coord_box = [52, 6.2]
min_rur_coord_box = [51, 5]
max_rur_coord_box = [53, 7]


## find the seed from the city center coordinates
from sklearn.neighbors import BallTree
ulolarad=np.deg2rad((lons,lats))
cities = [50.936860, 6.951152]
ball_tree = BallTree(ulolarad.T, leaf_size=2)
city='Cologne'
seed = []

dist, ind = ball_tree.query(np.asarray([np.deg2rad(cities[1]),np.deg2rad(cities[0])]).reshape(1, -1),k=1) # nearest neighbor for the city center
if dist[0,0]*6371<2:
   seed.append(ind[0][0])


# example 1: UHI field with M7
method = 7 # method number
uhi_m7, baseline_temperature = calc_uhi(method, temperature, lons, lats, topography, rural_land_use, urban_land_use, min_urb_coord_box, max_urb_coord_box, min_rur_coord_box, max_rur_coord_box,seed)


# quick plotting routine to check the results
# coordinate limits
map_limits_min = [min(lats), min(lons)]
map_limits_max = [max(lats), max(lons)]
quick_uhi_plot(uhi_m7, lons, lats, map_limits_min, map_limits_max)


