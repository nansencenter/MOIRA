# MOIRA

The package functionality includes deformation state detection from SAR Sentinel-1 GRD Level-1 images based on
combined use of texture characteristics and ice deformation data.
The use of ice deformation data is optional but efficiently complement SAR textures 
for unambiguous distinguishing between ridged and level ice.

The package also includes tools to prepare the input data for the training and detection. 

The test trained models for Sentinel-1 EW and IW data are also included in the package.

The following Quickstart guide covers ridged ice detection example step-by-step.

## Quick start

For quick start tutorial you can download a test Sentinel-1 image:

https://drive.google.com/file/d/164js5XFyExW5tt9Ud7PgvOG5upKshK_U/view?usp=sharing

The image provided by European Space Agency (ESA) © and has been downloaded from 
Copernicus Open Access Hub (https://scihub.copernicus.eu/)

To start using the package you need to change working directory or just copy ```ridge_classification.py``` 
to your local directory or change directory to the package path.

```python
import matplotlib.pyplot as plt
import os
plt.rcParams['figure.dpi'] = 150
from ridge_classification import *

# 1. First, we calculate multiscale texture features from Sentinel-1 image

s1_file = 'S1B_IW_GRDH_1SDV_20201222T203955_20201222T204024_024820_02F3F4_625F.zip'

# Define lat/lon bounding box
lon1, lat1 = 149.7342, 75.3338
lon2, lat2 = 153.0878, 75.4001

# Define projection EPSG number and pixel size in meter
proj_epsg = 5940
res = 20

# Initialize SAR texture object
t = SarTextures(s1_file, bbox=[lon1, lat1, lon2, lat2])
# Calibrate and project data
t.calibrate_project(proj_epsg, res, mask=False, write_file=True, out_path=out_path, backscatter_coeff='sigmaNought')
# Calculate texture features
t.calcTexFt()
# Save texture features in netCDF4 format
t.export_netcdf('textures_%s.nc' % 
                (os.path.basename(t.name).split('.')[0]))

# 2. Initialize a detection object based on only SAR textures and on SAR textures + ice defromation
clf_textures = deformedIceClassifier()
clf_textures_defo = deformedIceClassifier()

# 3. Load the detection models
# Load the detection model for Sentinel-1 IW VV data 
clf_textures.load_model('/mnt/sverdrup-2/sat_auxdata/MOIRA/models/mod_text_IW_VV.sav')

# Load the detection model for Sentinel-1 IW VV data 
clf_textures_defo.load_model('/mnt/sverdrup-2/sat_auxdata/MOIRA/models/mod_text_defo_IW_VV.sav')

# 4. Open file with texture features created at Step 1
file_path = 'textures_norm_s0_vv_S1B_IW_GRDH_1SDV_20201222T203955_20201222T204024_024820_02F3F4_625F_out.nc'
fts = clf_textures.read_features_file(file_path)

# 5. Perform detection from texture fetaures
res_text = clf_textures.detect_ice_state(fts)

# 6. Plot result
# Bbox for visualization
r1, r2, c1, c2  = 830, 1200, 1480, 1900
plt.imshow(res_text[r1:r2, c1:c2], interpolation='nearest')
plt.axis('off')





# Plot obtained result (zoomed area)
# Bbox for visualization
r1, r2, c1, c2  = 810, 1200, 1410, 1900

plt.imshow(res_no_defo[r1:r2,c1:c2], interpolation='nearest')

# Next we perform ridge detection with deformation data
clf_defo = deformedIceClassifier(glcm_filelist, ridges_filelist, level_filelist,
                                  defo_filelist, s0_filelist, defo_training=True)


# Open deformation data, interpolate onto texture features grid, add to the feature matrix
f_defo = 'MOIRA_test_data/defromation/100px_ICEDEF_20201221t204819_20201222t203955.nc'
r = Resampler(f_defo, 'MOIRA_test_data/textures/norm_s0_vv_S1B_IW_GRDH_1SDV_20201222T203955_20201222T204024_024820_02F3F4_625F_out.nc')

data_int_div = r.resample(r.f_source['lons'], r.f_source['lats'], r.f_target['lons'],
                          r.f_target['lats'], r.f_source['data']['ice_divergence'],
                          method='nearest', radius_of_influence=500000)

data_int_shear = r.resample(r.f_source['lons'], r.f_source['lats'], r.f_target['lons'], 
                            r.f_target['lats'], r.f_source['data']['ice_shear'],
                            method='nearest', radius_of_influence=50000)

# Add shear and divergence
fts_text_defo = np.dstack((fts, data_int_shear))
fts_text_defo = np.dstack((fts_text_defo, data_int_div))

# Replace NaN values with 0
fts_text_defo[np.isnan(fts_text_defo)] = 0

# Then we mask deformation values over those pixels classified as level (to harmonize deformation and texture data)
fts_text_defo[:,:,-1][res_no_defo<2] = 0
fts_text_defo[:,:,-2][res_no_defo<2] = 0

# Detect the deformation state of sea ice from texture and deformation features
res_defo = clf_defo.detect_ice_state(fts_text_defo)

# Plot obtained result (zoomed area)
# Bbox for visualization
plt.imshow(res_defo[r1:r2,c1:c2], interpolation='nearest')
```

![alt text](test/clf_no_defo.png)
![alt text](test/clf_defo.png)

Below you will find how prepare and calculate your data for the processing.

## 1. Open Sentinel-1 image

First you should import all classes from the package

```python
from ridge_classification import *
```

Then a Sentinel-1 image can be initialized by `SarTextures` class

```python
t = SarTextures(PATH/TO/S1/FILE, ws=11, stp=15, threads=10)
```

where `PATH/TO/S1/FILE` is a path to Sentinel-1 GRD (EW/IW) Level-1 file; `ws` - windows size for texture features computation; `stp` - computational grid step size; `threads` - number of threads.

## 2. Calibrate and project Sentinel-1 image

Peform data calibration and projection onto Polar Stereographic projection (EPSG:5041) with a spatial resolution of `res` [meters].

```python
t.calibrate_project(5041, res, mask=False, write_file=False, out_path='/OUTPUT/DIRECTORY')	
```

other parameters includes: `write_file` - set to True if you want to export the calibrated data as a geotiff file; `out_path` - ouput directory to store a geotiff file. 

## 3. Vector data preparation

The classification require reference data for the training. The common way to produce that is manual mapping of ridged and flat ice using GIS software. Once it is done, a vector file should be initialized by a class for vector data processing called `VectorData`:

```python

v = VectorData('/PATH/TO/VECTOR/FILE', t.ds[list(t.ds.keys())[0]], downsample=True)
```

where `t.ds[list(t.ds.keys())[0]]` is a gdal object with a projected geotiff from a previous step and we also set a `downsample` parameter to True to make further computations more fast. 


```python
v.rasterize('PATH/TO/RASTERIZED/GEOTIFF/TIFE')

```

## 4. SAR texture calculation

calculate texture characteristics and edges from SAR data:

```python
t.getMultiscaleTextureFeatures()
```

Then the obtained result can be stored in NetCDF4 file:

```python
t.export_netcdf('PATH/TO/OUTPUT/NETCDF')
```

The package was created as a part of MOdel for Sea Ice RAdar Image Microstructure (MOIRA)
project for Europian Space Agency (ESA).