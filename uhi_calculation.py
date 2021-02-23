r"""
This code is used to calculate of the Urban Heat Island effect as described in doi:XXXX

For a proper usage fill the empty case-specific values in the header.

All gridded data read has the spatial coordinate in 1D. If your data comes in a lon-lat grid, flatten it before applying the methods.


"""
import numpy as np
from netCDF4 import Dataset
from sklearn.neighbors import BallTree


def uhi1(t, obs):
    uhi = t - obs
    return uhi


def uhi2(t):
    global topo_urb_mean, urb_lu, topo, rur_lu, rural_area_index
    n = np.where(np.logical_and(urb_lu[rural_area_index] < 0.2, topo[rural_area_index] < topo_urb_mean))[0]
    rural = rur_lu[rural_area_index]
    index_max_rural = np.nanmax(rural[n])
    rur_grid_point = rural[n].tolist().index(index_max_rural)
    baseline = t[rural_area_index[rur_grid_point]]
    uhi = t - baseline
    print('base uhi2 ', baseline)
    return uhi


def uhi3(t):
    global area_extent_index, topo_urb_mean, urb_lu, topo
    ind = np.where(urb_lu[area_extent_index] < 0.2)[0]
    baseline = np.nanmean(t[area_extent_index[ind]])
    uhi = t - baseline
    print('base uhi3 ', baseline)

    print(uhi.shape)
    return uhi


def uhi4(t):
    global rural_area_index, topo_urb_mean, urb_lu, topo
    ind = np.where(urb_lu[rural_area_index] < 0.2)[0]
    baseline = np.nanmean(t[rural_area_index[ind]])
    uhi = t - baseline
    print('base uhi4 ', baseline)
    return uhi


def uhi5(t):
    global rural_area_index, topo_urb_mean, urb_lu, topo
    ind = np.where(np.logical_and(urb_lu[rural_area_index] < 0.2,
                                  np.logical_and(topo[rural_area_index] > 0.8 * topo_urb_mean,
                                                 topo[rural_area_index] < 1.2 * topo_urb_mean)
                                  )
                   )[0]
    baseline = np.nanmean(t[rural_area_index[ind]])
    print('base uhi5 ', baseline)
    uhi = t - baseline
    return uhi


def uhi6(t):
    global rural_area_index, topo_urb_mean, urb_lu, topo
    ind = np.where(urb_lu[rural_area_index] < 0.2)[0]
    gamma0 = - 6.5 * 10 ** (-3)
    t_correct = np.asarray([t[i] + gamma0 * (topo_urb_mean - topo[i]) for i in np.arange(0,topo.shape[0])])
    baseline = np.nanmean(t_correct[rural_area_index[ind]])
    uhi = t_correct - baseline
    print('base uhi6 ', baseline)
    return uhi


def uhi7(t):
    global rural_area_index, seeds
    lolarad = np.deg2rad((lons, lats))
    urb = urb_lu > 0.5
    urblo = lons[urb]
    urbla = lats[urb]
    ulolarad = np.deg2rad((urblo, urbla))
    indices = np.zeros((30, ulolarad.shape[1])).astype(int)
    indices[:] = -1
    maxdist = 8  # search radius
    new_seeds = np.where([x in seeds for x in np.where(urb)[0]])
    indices[0, 0:len(new_seeds[0].tolist())] = new_seeds[0]
    for i in np.arange(0, 25):
        ulolarad_core = ulolarad[:, indices[i, 0:np.sum(indices[i, :] > -1)]]
        bt = BallTree(ulolarad_core.transpose())
        dist, idx = bt.query(ulolarad.transpose())
        dist = dist[:, 0] * 6371
        indices[i + 1, 0:np.sum(dist < maxdist)] = np.array(np.where(dist < maxdist)[0])
        dist, idx = bt.query(lolarad.transpose())

    dist = dist[:, 0] * 6371
    nn = 5
    minrural = 10  # in km
    rural_value = .2
    t_rural = t[:][np.logical_and(urb_lu[:] < rural_value, dist > minrural)]
    lolarad_rural = lolarad[:, np.logical_and(urb_lu[:] < rural_value, dist > minrural)]
    bt = BallTree(lolarad_rural.transpose())
    rurdist, idx = bt.query(lolarad.transpose(), k=nn)
    rurdist = rurdist * 6371
    if (nn > 1):
        weights = 1 / np.exp(rurdist / np.reshape(np.repeat(np.sum(rurdist, axis=1), nn), rurdist.shape))
        weights = weights / np.reshape(np.repeat(np.sum(weights, axis=1), nn), rurdist.shape)
        tbase = np.sum(t_rural[idx] * weights, axis=1)
    else:
        tbase = t_rural[idx[:, 0]]
    baseline = np.mean(tbase[rural_area_index])
    print('base uhi7 ', baseline)
    uhi = t - baseline
    return uhi


# plt.scatter(lons[rural_area_index[ind]], lats[rural_area_index[ind]])
# plt.show()
# urban box coordinates: urban core
lo1 = 5.8
la1 = 51.5
la2 = 52
lo2 = 6.2

# rural box coordinates
lor1 = 5
lar1 = 51
lar2 = 53
lor2 = 7

# map limits
map_lim_llon = 5
map_lim_llat = 50.5
map_lim_ulon = 7.8
map_lim_ulat = 54

# define the seeds as a list here
seeds=[]

# read here the station data:
rur_obs = # rural observation from the weather station

# read here the fields:
inp = '' # netcdf input file with the temperature fields
datafile = Dataset(inp)
temperature = # surface temperature 1D

invarfile = Dataset("") # input file with the invariant data
topo = # topography: 1D
rur_lu = # rural land use: 1D
urb_lu =  # urban use: 1D

#coordinates
lons = # longitude 1D
lats = # latitude 1D

# select the area within the map limits
select_area_index = np.where(np.logical_and(np.logical_and(lons > map_lim_llon, lons < map_lim_ulon),
                                            np.logical_and(lats > map_lim_llat, lats < map_lim_ulat)))[0]
# urban area
urban_area_index_mask = np.logical_and(np.logical_and(lons > lo1, lons < lo2),
                                       np.logical_and(lats > la1, lats < la2))
urban_area_index = np.where(urban_area_index_mask)[0]

# rural area: it includes the urban one
area_extent_index_mask = np.logical_and(np.logical_and(lons > lor1, lons < lor2),
                                        np.logical_and(lats > lar1, lats < lar2))
area_extent_index = np.where(area_extent_index_mask)[0]

# rural only area:
rural_area_index = np.where(np.logical_and(area_extent_index_mask, np.logical_not(urban_area_index_mask)))[0]

# average topography height of the urban core
topo_urb_mean = np.nanmean(topo[np.where(topo[urban_area_index])[0]])


# e.g UHI1
uhi_m1 = uhi1(temperature, rur_obs)

#e.g. UHI5
uhi_m5 = uhi5(temperature)