#!/usr/bin/env python3
"""
This code is used to calculate of the Urban Heat Island effect as described in:

Valmassoi and Keller (2021), Representing the Urban Heat Island in Gridded Datasets, Advances in Science and Research.
DOI:

This program is a free software distributed under the terms of the GNU General Public License as published by
the Free Software Foundation, version 3 (GNU-GPLv3).

You can redistribute and/or modify by citing the mentioned publication, but WITHOUT ANY WARRANTY; without even the
implied warranty of  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

For a description of the functions uhi1 to uhi7, refer to Valmassoi and Keller (2021).
"""
import numpy as np
from sklearn.neighbors import BallTree
from typing import Union, Tuple
import sys
from enum import Enum


# TODO Unused for the time being

class Coords:
    """
    Simple wrapper class that holds a latitude and a longitude coordinate. Use as follows:

        coords = Coords(5.8, 12.3)
        coords = Coords(lat=5.8, lon=12.3)
    """

    def __init__(self, lat: float, lon: float):
        self.lat = lat
        self.lon = lon


class Bounds:
    def __init__(self, min_coords: Union[Coords, Tuple[float, float]], max_coords: Union[Coords, Tuple[float, float]]):
        """
        Construct a new bounds object
        :param min_coords: Minimum latitude and longitude. Accepts both a Coords object or a 2-tuple (lat, lon).
        :param max_coords: Minimum latitude and longitude. Accepts both a Coords object or a 2-tuple (lat, lon).

            bounds = Bounds(min_coords=(5.8, 12.3), max_coords=(8.5, 15.6))
            bounds = Bounds(Coords(5.8, 12.3), Coords(8.5, 15.6))
        """
        self.min = min_coords if isinstance(min_coords, Coords) else Coords(*min_coords)
        self.max = max_coords if isinstance(max_coords, Coords) else Coords(*min_coords)

    def get_index_mask(self, latitudes, longitudes):
        """
        Returns a index mask (1 where satisfied, 0 where not) marking all latitudes and logitudes that are in range for
        this Bounds object. Extrema are excluded, i.e. it returns a index mask where

            self.min.lon < longitudes < self.max.lon and self.min.lat < latitudes < self.max.lat
        :param latitudes: List of latitude coordinates (matching the shape of longitudes)
        :param longitudes: List of longitude coordinates (matching the shape of latitudes)
        :return:
        """
        return np.logical_and(np.logical_and(longitudes > self.min.lon, longitudes < self.max.lon),
                              np.logical_and(latitudes > self.min.lat, latitudes < self.max.lat))


class UHI:
    GAMMA0 = - 6.5 * 10 ** (-3)  # lapse rate
    URBAN_USE_THRESHOLD_DEFAULT = 0.2
    URBAN_CORE_THRESHOLD_DEFAULT = 0.5
    TOPO_URBAN_MEAN_SCALING_DEFAULT = (0.8, 1.2)
    URBAN_SEARCH_RADIUS_DEFAULT = 8  # km
    RURAL_SEARCH_RADIUS_DEFAULT = 10  # km
    NEAR_NEIGHBOUR_DEFAULT = 5
    #URBAN_SEEDS_DEFAULT = [0,1,2]  # grid point number
    topography = None

    """
    Area extent: it includes the urban one.
    """
    area_extent_index = None

    """
    Urban area.
    """
    urban_area_index = None

    """
    Rural only area.
    """
    rural_area_index = None

    """
    Average topography height of the urban core.
    """
    topo_urban_mean = None

    def __init__(self, lons, lats, topography, rural_land_use, urban_use, area_extent_index_mask,
                 urban_area_index_mask):
        """
        :param lons: 1D
        :param lats: 1D
        :param topography: 1D
        :param rural_land_use: 1D
        :param urban_use: 1D
        :param area_extent_index_mask: ones in the considered areas, zero otherwise, indexing referred to ''topography''
        :param urban_area_index_mask: ones in the urban area, zero otherwise, indexing referred to ''topography''
        """
        self.lons = lons
        self.lats = lats
        self.topography = topography
        self.rural_land_use = rural_land_use
        self.urban_use = urban_use
        self.area_extent_index = np.where(area_extent_index_mask)[0]
        self.urban_area_index = np.where(urban_area_index_mask)[0]
        self.rural_area_index = np.where(np.logical_and(area_extent_index_mask,
                                                        np.logical_not(urban_area_index_mask)))[0]
        self.topo_urban_mean = np.nanmean(self.topography[np.where(self.topography[self.urban_area_index])[0]])

    def _choose_uhi(self, method):
        fn = getattr(self, f'uhi{method}', None)
        if fn is None or not callable(fn):
            raise NotImplementedError(f'The method {method} is not known')
        return fn

    def __call__(self, method, *args, **kwargs):
        return self._choose_uhi(method)(*args, **kwargs)

    def uhi1(self, t, baseline):
        print(f'base uhi1 {baseline}')
        return t - baseline

    def uhi2(self, t, urban_use_threshold=URBAN_USE_THRESHOLD_DEFAULT):
        ind = np.where(np.logical_and(self.urban_use[self.rural_area_index] < urban_use_threshold,
                                      self.topography[self.rural_area_index] < self.topo_urban_mean))[0]
        rural = self.rural_land_use[self.rural_area_index]
        print(self.rural_area_index)
        index_max_rural = np.nanmax(rural[ind])
        rur_grid_point = rural[ind].tolist().index(index_max_rural)
        baseline = t[self.rural_area_index[rur_grid_point]]
        print(f'base uhi2 {baseline}')
        return t - baseline

    def uhi3(self, t, urban_use_threshold=URBAN_USE_THRESHOLD_DEFAULT):
        ind = np.where(self.urban_use[self.area_extent_index] < urban_use_threshold)[0]
        baseline = np.nanmean(t[self.area_extent_index[ind]])
        print(f'base uhi3 {baseline}')
        return t - baseline

    def uhi4(self, t, urban_use_threshold=URBAN_USE_THRESHOLD_DEFAULT):
        ind = np.where(self.urban_use[self.rural_area_index] < urban_use_threshold)[0]
        baseline = np.nanmean(t[self.rural_area_index[ind]])
        print(f'base uhi4 {baseline}')
        return t - baseline

    def uhi5(self, t, urban_use_threshold=URBAN_USE_THRESHOLD_DEFAULT,
             topo_urban_mean_scaling=TOPO_URBAN_MEAN_SCALING_DEFAULT):

        topo_urban_mean_lscale, topo_urban_mean_uscale = topo_urban_mean_scaling
        ind = np.where(np.logical_and(
            self.urban_use[self.rural_area_index] < urban_use_threshold,
            np.logical_and(self.topography[self.rural_area_index] > topo_urban_mean_lscale * self.topo_urban_mean,
                           self.topography[self.rural_area_index] < topo_urban_mean_uscale * self.topo_urban_mean)
        ))[0]
        baseline = np.nanmean(t[self.rural_area_index[ind]])
        print(f'base uhi5 {baseline}')
        return t - baseline

    def uhi6(self, t, urban_use_threshold=URBAN_USE_THRESHOLD_DEFAULT):
        ind = np.where(self.urban_use[self.rural_area_index] < urban_use_threshold)[0]
        t_correct = np.asarray([t[i] + self.GAMMA0 * (self.topo_urban_mean - self.topography[i])
                                for i in np.arange(0, self.topography.shape[0])])
        baseline = np.nanmean(t_correct[self.rural_area_index[ind]])
        print(f'base uhi6 {baseline}')
        return t_correct - baseline

    def uhi7(self, t, seeds, urban_use_threshold=URBAN_USE_THRESHOLD_DEFAULT,
             urban_core_threshold=URBAN_CORE_THRESHOLD_DEFAULT, urban_search_radius=URBAN_SEARCH_RADIUS_DEFAULT,
             rural_search_radius=RURAL_SEARCH_RADIUS_DEFAULT, nn=NEAR_NEIGHBOUR_DEFAULT):
        lolarad = np.deg2rad((self.lons, self.lats))
        urb = self.urban_use > urban_core_threshold
        urblo = self.lons[urb]
        urbla = self.lats[urb]
        ulolarad = np.deg2rad((urblo, urbla))
        indices = np.zeros((30, ulolarad.shape[1])).astype(int)
        indices[:] = -1
        maxdist = 8  # search radius
        new_seeds = np.where([x in seeds for x in np.where(urb)[0]])
        print(new_seeds)
        indices[0, 0:len(new_seeds[0].tolist())] = new_seeds[0]
        for i in np.arange(0, 25):
            ulolarad_core = ulolarad[:, indices[i, 0:np.sum(indices[i, :] > -1)]]
            bt = BallTree(ulolarad_core.transpose())
            dist, idx = bt.query(ulolarad.transpose())
            dist = dist[:, 0] * 6371
            indices[i + 1, 0:np.sum(dist < maxdist)] = np.array(np.where(dist < maxdist)[0])
            dist, idx = bt.query(lolarad.transpose())

        dist = dist[:, 0] * 6371
        t_rural = t[:][np.logical_and(self.urban_use[:] < urban_use_threshold, dist > rural_search_radius)]
        lolarad_rural = lolarad[:, np.logical_and(self.urban_use[:] < urban_use_threshold, dist > rural_search_radius)]
        bt = BallTree(lolarad_rural.transpose())
        rurdist, idx = bt.query(lolarad.transpose(), k=nn)
        rurdist = rurdist * 6371
        if (nn > 1):
            weights = 1 / np.exp(rurdist / np.reshape(np.repeat(np.sum(rurdist, axis=1), nn), rurdist.shape))
            weights = weights / np.reshape(np.repeat(np.sum(weights, axis=1), nn), rurdist.shape)
            tbase = np.sum(t_rural[idx] * weights, axis=1)
        else:
            tbase = t_rural[idx[:, 0]]
        
        map_limits_min = [50.5, 5]
        map_limits_max = [54, 7.8]
        quick_uhi_plot(tbase, self.lons, self.lats, map_limits_min, map_limits_max,np.arange(5,30))
        baseline = np.mean(tbase[self.rural_area_index])
        print(f'base uhi7 {baseline}')
        uhi = t - baseline
        return uhi


class Parameters:
    lons = None  # longitude 1D
    lats = None  # latitude 1D
    topography = None  # topography: 1D
    rural_land_use = None  # rural land use: 1D
    urban_land_use = None  # urban use: 1D
    temperature = None
    def assert_not_none(self):
        all_defined = True
        for member_name in filter(lambda name: not name.startswith('_'), self.__dict__.keys()):
            if getattr(self, member_name) is None:
                print(f'Please edit the script and set {member_name}.', file=sys.stderr)
                all_defined = False
        if not all_defined:
            sys.exit(1)

    def check_if_1D(self):
        dimension = 1
        for member_name in filter(lambda name: not name.startswith('_'), self.__dict__.keys()):
            if getattr(self, member_name).ndim != 1:
                print(f' All variables should be 1D, here {member_name} has ', getattr(self, member_name).ndim )
                sys.exit(1)


def calc_uhi(method, t, lons, lats, topography, rural_land_use, urban_land_use, min_urb_coord_box,
             max_urb_coord_box, min_rur_coord_box, max_rur_coord_box, additional_info=None):

    parms = Parameters()

    parms.temperature = t  # surface temperature 1D
    parms.lons = lons  # longitude 1D
    parms.lats = lats  # latitude 1D
    parms.topography = topography  # topography: 1D
    parms.rural_land_use = rural_land_use  # rural land use: 1D
    parms.urban_land_use = urban_land_use  # urban use: 1D
    parms.check_if_1D()

    # urban box coordinates: urban core
    parms.urban_box = Bounds(min_coords=Coords(lat=min_urb_coord_box[0], lon=min_urb_coord_box[1]),
                             max_coords=Coords(lat=max_urb_coord_box[0], lon=max_urb_coord_box[1]))

    # rural box coordinates
    parms.rural_box = Bounds(min_coords=Coords(lat=min_rur_coord_box[0], lon=min_rur_coord_box[1]),
                             max_coords=Coords(lat=max_rur_coord_box[0], lon=max_rur_coord_box[1]))

    # This will fail the script if they're undefined
    parms.assert_not_none()
    uhi = UHI(lons=parms.lons, lats=parms.lats,
              topography=parms.topography,
              rural_land_use=parms.rural_land_use,
              urban_use=parms.urban_land_use,
              area_extent_index_mask=parms.rural_box.get_index_mask(parms.lats, parms.lons),
              urban_area_index_mask=parms.urban_box.get_index_mask(parms.lats, parms.lons))

    if (method == 1 or method == 7):
        uhi_all = uhi(method, parms.temperature, additional_info)
    else:
        uhi_all = uhi(method, parms.temperature)
    return uhi_all


def quick_uhi_plot(uhi,  lons, lats, map_limits_min, map_limits_max, levs=np.arange(-3.25, 3.75, .5)):
    import matplotlib.pylab as plt
    import os
    # if needed add os.environ["PROJ_LIB"] = "[..]/share/proj"
    from mpl_toolkits.basemap import Basemap
    LIMS = Bounds(min_coords=Coords(lat=map_limits_min[0], lon=map_limits_min[1]),
                  max_coords=Coords(lat=map_limits_max[0], lon=map_limits_max[1]))
    par = np.arange(int(LIMS.min.lat)-1, int(LIMS.max.lat)+1, .5)
    mer = np.arange(int(LIMS.min.lon)-1, int(LIMS.max.lon)+1, .5)
    latc = (int(LIMS.max.lat) + int(LIMS.min.lat))/2
    lonc = (int(LIMS.max.lat) + int(LIMS.min.lat))/2
    fig, ax = plt.subplots(figsize=(10, 10))
    m = Basemap(llcrnrlon=LIMS.min.lon, llcrnrlat=LIMS.min.lat, urcrnrlon=LIMS.max.lon, urcrnrlat=LIMS.max.lat,
                lat_0=latc, lon_0=lonc,
                projection='cyl', resolution='c')
    m.drawparallels(par, labels=[1, 0, 0, 0], linewidth=0.0)
    m.drawmeridians(mer, labels=[0, 0, 0, 1], linewidth=0.0)
    print(np.nanmean(uhi))
    cb = m.contourf(lons, lats, uhi, levs, cmap=plt.cm.bwr, tri=True, extend='both', alpha=.8)
    plt.colorbar(cb)
    plt.show()
