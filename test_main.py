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
import sys
inp_file='dataset_test.csv'

df = pd.read_csv(inp_file)

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


expected_results=[28,17.732819,2]

add_input = None
for method in np.arange(1,4):
    print(method)
    if method==1:
       add_input = 28.
    if method==7:
       add_input = seed
    uhi,base = calc_uhi(method, temperature, lons, lats, topography, rural_land_use, urban_land_use, min_urb_coord_box, max_urb_coord_box, min_rur_coord_box, max_rur_coord_box,add_input)
    if base!=expected_results[method-1] :
         raise ValueError("Wrong value obtained")




