from osgeo import ogr
from osgeo import osr
from netCDF4 import Dataset
from math import pi, hypot, atan2, pi
from numpy import cos, sin
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import cKDTree
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

def reproject_ft(ft, proj):
    source = osr.SpatialReference()
    source.ImportFromWkt(proj)

    target = osr.SpatialReference()
    target.ImportFromEPSG(4326)

    transform = osr.CoordinateTransformation(source, target)

    result = ogr.CreateGeometryFromWkt(ft)
    result.Transform(transform)

    return result

def tunnel_fast(latvar, lonvar, lat0, lon0):
    rad_factor = pi / 180.0  # for trignometry, need angles in radians
    # Read latitude and longitude from file into numpy arrays
    latvals = latvar[:] * rad_factor
    lonvals = lonvar[:] * rad_factor
    ny, nx = latvals.shape
    lat0_rad = lat0 * rad_factor
    lon0_rad = lon0 * rad_factor

    # Compute numpy arrays for all values, no loops
    clat, clon = cos(latvals), cos(lonvals)
    slat, slon = sin(latvals), sin(lonvals)

    delX = cos(lat0_rad) * cos(lon0_rad) - clat * clon
    delY = cos(lat0_rad) * sin(lon0_rad) - clat * slon
    delZ = sin(lat0_rad) - slat;
    dist_sq = delX ** 2 + delY ** 2 + delZ ** 2

    minindex_1d = dist_sq.argmin()  # 1D index of minimum element

    iy_min, ix_min = np.unravel_index(minindex_1d, latvals.shape)

    print(iy_min, ix_min)

    return iy_min, ix_min

def kdtree_fast(latvar, lonvar, lat0, lon0):
    rad_factor = pi / 180.0  # for trignometry, need angles in radians
    # Read latitude and longitude from file into numpy arrays
    latvals = latvar[:] * rad_factor
    lonvals = lonvar[:] * rad_factor
    ny, nx = latvals.shape
    clat, clon = cos(latvals), cos(lonvals)
    slat, slon = sin(latvals), sin(lonvals)
    # Build kd-tree from big arrays of 3D coordinates
    triples = zip(np.ravel(clat * clon), np.ravel(clat * slon), np.ravel(slat))

    # kdt = cKDTree(triples)
    print('\nBuilding KDE tree...')
    kdt = cKDTree(np.c_[np.ravel(clat * clon), np.ravel(clat * slon), np.ravel(slat)])
    print('Done.\n')

    lat0_rad = lat0 * rad_factor
    lon0_rad = lon0 * rad_factor
    clat0, clon0 = cos(lat0_rad), cos(lon0_rad)
    slat0, slon0 = sin(lat0_rad), sin(lon0_rad)
    dist_sq_min, minindex_1d = kdt.query([clat0 * clon0, clat0 * slon0, slat0])
    iy_min, ix_min = np.unravel_index(minindex_1d, latvals.shape)
    print(iy_min, ix_min)
    return iy_min, ix_min


def get_deformation_shp(shp_fname, nc_fname):
    # shapefile = r'/mnt/sverdrup-2/sat_auxdata/MOIRA/ridging_case_VSM_12_2020/shp/flat_ice/2212_flat.shp'
    drv = ogr.GetDriverByName('ESRI Shapefile')
    dataSet = drv.Open(shp_fname)

    # Open netCDF with deformation
    nc_defo = Dataset(nc_fname, 'r')
    lats_nc = nc_defo['lat'][:]
    lons_nc = nc_defo['lon'][:]
    shear = nc_defo['ice_shear'][:]
    div = nc_defo['ice_divergence'][:]
    total_deformation = nc_defo['total_deformation'][:]

    r_div = []
    r_shear = []
    r_total_deformation = []

    layer = dataSet.GetLayer(0)
    layerDefinition = layer.GetLayerDefn()
    for i in range(layerDefinition.GetFieldCount()):
        name = layerDefinition.GetFieldDefn(i).GetName()

    for feature in layer:
        # print('\n%s\n' % feature.GetGeometryRef())
        geom = feature.GetGeometryRef()
        spatialRef = geom.GetSpatialReference().ExportToWkt()
        # print(spatialRef)
        rep_ft = reproject_ft(feature.GetGeometryRef().ExportToWkt(), spatialRef)
        # print('\n%s\n' % rep_ft)

        for icoord in rep_ft.GetPoints():
            # print(icoord)
            line, row = tunnel_fast(lats_nc, lons_nc, icoord[1], icoord[0])
            # print('Divergence: %s' % div[line,row])
            # print('Shear: %s' % shear[line,row])
            r_div.append(div[line, row])
            r_shear.append(shear[line, row])
            r_total_deformation.append(total_deformation[line, row])

    layer.ResetReading()
    return r_div, r_shear, r_total_deformation


def get_pca_shp(shp_fname, nc_fname, vars):
    # shapefile = r'/mnt/sverdrup-2/sat_auxdata/MOIRA/ridging_case_VSM_12_2020/shp/flat_ice/2212_flat.shp'
    drv = ogr.GetDriverByName('ESRI Shapefile')
    dataSet = drv.Open(shp_fname)

    # Open netCDF with PC components
    nc_file = Dataset(nc_fname, 'r')
    lats_nc = nc_file['lat'][:]
    lons_nc = nc_file['lon'][:]

    # vars = [var for var in nc_file.variables.keys() if (not var == 'time'  and not var == 'lat' and not var == 'lon')]

    d_res = {}

    for ivar in vars:
        d_res[ivar] = []

        # Open netCDF values
        vals = nc_file[ivar][:, :]

        layer = dataSet.GetLayer(0)
        layerDefinition = layer.GetLayerDefn()

        for i in range(layerDefinition.GetFieldCount()):
            name = layerDefinition.GetFieldDefn(i).GetName()

        for i, feature in enumerate(layer):
            print('%s %s of %s' % (ivar, i, len(layer)))
            # print('\n%s\n' % feature.GetGeometryRef())
            geom = feature.GetGeometryRef()
            spatialRef = geom.GetSpatialReference().ExportToWkt()
            # print(spatialRef)
            rep_ft = reproject_ft(feature.GetGeometryRef().ExportToWkt(), spatialRef)
            # print('\n%s\n' % rep_ft)

            for icoord in rep_ft.GetPoints():
                print('\n Init: %s' % str(icoord))
                line, col = kdtree_fast(lats_nc, lons_nc, icoord[1], icoord[0])
                d_res[ivar].append(vals[line, col])
                print('found coords: lon:%s lat:%s\n' % (lons_nc[line, col], lats_nc[line, col]))
        layer.ResetReading()

    return d_res


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


def get_s0_shp(shp_fname, nc_fname, ws):
    res = {}

    # dataSet = drv.Open(shp_fname)

    # Open netCDF
    nc = Dataset(nc_fname, 'r')
    lats_nc = nc['lat'][:]
    lons_nc = nc['lon'][:]

    vvars = []

    for ivar in nc.variables:
        if 'sigma' in ivar:
            vvars.append(ivar)

    for ivar in vvars:
        print('\n######## %s ##########\n' % ivar)
        s0 = nc[ivar][:]

        r_s0 = []

        # Read shapefile
        drv = ogr.GetDriverByName('ESRI Shapefile')
        dataSet = drv.Open(shp_fname)
        layer = dataSet.GetLayer(0)
        num_of_ft = layer.GetFeatureCount()
        layerDefinition = layer.GetLayerDefn()

        for i in range(layerDefinition.GetFieldCount()):
            name = layerDefinition.GetFieldDefn(i).GetName()

        for i, feature in enumerate(layer):
            print('\nFeature # %s of %s\n' % (i + 1, num_of_ft))
            # print('\n%s\n' % feature.GetGeometryRef())
            geom = feature.GetGeometryRef()
            spatialRef = geom.GetSpatialReference().ExportToWkt()
            # print(spatialRef)
            rep_ft = reproject_ft(feature.GetGeometryRef().ExportToWkt(), spatialRef)
            # print('\n%s\n' % rep_ft)

            print('\nNumber of points to analyze: %s' % len(rep_ft.GetPoints()))

            for icoord in rep_ft.GetPoints():
                # print(icoord)
                r, c = tunnel_fast(lats_nc, lons_nc, icoord[1], icoord[0])
                r_s0.append(np.nanmean(s0[r - ws:r + ws, c - ws:c + ws]))
                print(s0[r, c])

        res[ivar] = r_s0
        layer.ResetReading()

    return res

def plot_hist_2d(fname, data, title, x_label, c, xlim, bins, v_line):
    plt.clf()

    df = pd.DataFrame(data=data, columns=['data'])
    print(df['data'])

    fig, ax = plt.subplots()
    # sns.axes_style("white")
    sns.set(color_codes=True)
    h = sns.distplot(df['data'], color=c, ax=ax)

    sns.set_style("ticks")
    title = '%s\nMean: %.3f, STD: %.3f' % (title, df.data.mean(), df.data.std())

    # ax.set(xlabel=x_label, xlim=xlim, ylabel="Density", title=title)
    h.axes.set_title(title, fontsize=25)
    h.set_xlabel(x_label, fontsize=25)
    h.set_ylabel('Probability density', fontsize=25)
    h.tick_params(labelsize=20)
    h.set_xlim(xlim)
    # ax.legend()

    if v_line is not None:
        ax.axvline(x=v_line, linestyle='--')

    plt.savefig(out_fname, bbox_inches='tight', dpi=300)

######################################
# PC components over ridges
######################################

axes = plt.gca()

pca_fname = '/mnt/sverdrup-2/sat_auxdata/MOIRA/ridging_case_VSM_12_2020/geotiff/experiments/crop1_ps_HH_db_S1B_IW_GRDH_1SDV_20201222T203955_20201222T204024_024820_02F3F4_625F.tif/nc/11_crop1_ps_HH_db_S1B_IW_GRDH_1SDV_20201222T203955_20201222T204024_024820_02F3F4_625F.nc'
shp_fname = '/mnt/sverdrup-2/sat_auxdata/MOIRA/ridging_case_VSM_12_2020/shp/newedition/DD_cln_20201222_hummock_strong.shp'
#shp_fname = '/mnt/sverdrup-2/sat_auxdata/MOIRA/ridging_case_VSM_12_2020/shp/newedition/single_ridge.shp'
out_folder = '/mnt/sverdrup-2/sat_auxdata/MOIRA/ridging_case_VSM_12_2020/geotiff/experiments/crop1_ps_HH_db_S1B_IW_GRDH_1SDV_20201222T203955_20201222T204024_024820_02F3F4_625F.tif/PCA_OVER_RIDGES'

try:
    os.makedirs(out_folder)
except:
    pass

nc_file = Dataset(pca_fname, 'r')

nc_file.variables.keys()

#vars = [var for var in nc_file.variables.keys() if (not var == 'time'  and not var == 'lat' and not var == 'lon')]
vars = ['PC1', 'PC2','Contrast', 'Entropy'] #
d_res_ridges = get_pca_shp(shp_fname, pca_fname, vars)

######################################
# PC components over flat ice
######################################

axes = plt.gca()

pca_fname = '/mnt/sverdrup-2/sat_auxdata/MOIRA/ridging_case_VSM_12_2020/geotiff/experiments/crop1_ps_HH_db_S1B_IW_GRDH_1SDV_20201222T203955_20201222T204024_024820_02F3F4_625F.tif/nc/11_crop1_ps_HH_db_S1B_IW_GRDH_1SDV_20201222T203955_20201222T204024_024820_02F3F4_625F.nc'
shp_fname = '/mnt/sverdrup-2/sat_auxdata/MOIRA/ridging_case_VSM_12_2020/shp/newedition/22122020_level_ice_new.shp'
out_folder = '/mnt/sverdrup-2/sat_auxdata/MOIRA/ridging_case_VSM_12_2020/geotiff/experiments/crop1_ps_HH_db_S1B_IW_GRDH_1SDV_20201222T203955_20201222T204024_024820_02F3F4_625F.tif/PCA_OVER_FLAT_ICE'

try:
    os.makedirs(out_folder)
except:
    pass

nc_file = Dataset(pca_fname, 'r')

nc_file.variables.keys()
#vars = [var for var in nc_file.variables.keys() if (not var == 'time'  and not var == 'lat' and not var == 'lon')]

d_res_flat = get_pca_shp(shp_fname, pca_fname, vars)

#######################################
# Data frame with features
#######################################
d_df = {}

# Variables
for ivar in vars:
    d_df[ivar] = d_res_ridges[ivar] + d_res_flat[ivar]

# Classes of ice
d_df['ice_class'] = [1 for i in range(len(d_res_ridges[ivar]))] + [0 for i in range(len(d_res_flat[ivar]))]

data_train_pca_ridges = pd.DataFrame(d_df)

#########################################
# Ridge classification
#########################################

# list_features = ['PC1', 'PC2']
list_features = vars

X = data_train_pca_ridges[list_features]  # Features
y = data_train_pca_ridges['ice_class']  # Labels

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)  # 70% training and 30% test

# Create a Gaussian Classifier
clf = RandomForestClassifier(n_estimators=100)

# Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# print(data_train_pca_ridges[0:5])
# print(data_train_pca_ridges[-5:])

# Dictonary with features
d_ft_data = {}
for i_ft in list_features:
    d_ft_data[i_ft] = nc_file[i_ft][:].data

classified_data = np.zeros((d_ft_data[i_ft].shape[0], d_ft_data[i_ft].shape[1]))
classified_data[:] = np.nan

r_min, r_max = 0, classified_data.shape[0]
c_min, c_max = 0, classified_data.shape[1]

for row in range(r_min, r_max):
    for column in range(c_min, c_max):
        # for row in range(pc1_data.shape[0]):
        #    for column in range(pc1_data.shape[1]):
        test_sample = []
        for i_ft in list_features:
            test_sample.append(d_ft_data[i_ft][row, column])
        y_pred = clf.predict([test_sample])
        classified_data[row, column] = y_pred

plt.clf()
plt.title(list_features)
plt.imshow(classified_data[r_min:r_max, c_min:c_max])
plt.savefig('classified.png', bbox_inches='tight', dpi=300)

