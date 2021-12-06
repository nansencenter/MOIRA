# MOIRA

The package allows performing detection of ridged and level ice from SAR Sentinel-1 GRD Level-1 image based on texture characteristics and ice deformation data.
There are two ways to detect ice ridges and ridge clusters by the algorithm: using ice deformation data and SAR textures and only SAR texture information. We recommend using both data sources as a more robust way to overcome ambiguities caused by the similar signature of ridged ice and young ice in leads. But the deformation data is optional for the algorithm.

The package also includes tools that help to prepare the input data for the training and detection. 

The following Quickstart guide covers all the essential steps to perform sea ice ridge detection step-by-step.

## Quick start

For quick start tutorial you can download zipped test dataset containing data for ridged ice detection:

https://drive.google.com/file/d/1_tPzNTzbGGKzytzMXWQuXcfAoAzaxOpc/view?usp=sharing

After you unpack the archive you can use them in the following script. Here is the folder structure:

```textures``` - SAR textures and edges calculated from Sentinel-1 image<br/>
```ice_deformaition``` - contains ice divergence and shear in NetCDF4 format<br/>
```level_ice``` - GeoTIFF file containing pixels with level ice flag<br/>
```ridged_ice``` - GeoTIFF file containing pixels with ridged ice flag<br/>

To start using the package you need to change working directory or just copy ```ridge_classification.py``` to your local directory.

```python
import matplotlib.pyplot as plt
import glob
plt.rcParams['figure.dpi'] = 150
from ridge_classification import *

# Define paths to input data
s0_filelist = []
glcm_filelist = glob.glob('MOIRA_test_data/textures/norm_s0_vv_S1B_IW_GRDH_1SDV_20201222T203955_20201222T204024_024820_02F3F4_625F_*.nc')
level_filelist = glob.glob('MOIRA_test_data/level_ice/20201222T203955_level_*.tiff')
ridges_filelist = glob.glob('MOIRA_test_data/ridged_ice/20201222T203955_ridges_*.tiff')
defo_filelist = glob.glob('MOIRA_test_data/defromation/100px_ICEDEF_20201221t204819_20201222t203955*.nc')

# Initialize detection object (from textures)
clf_no_defo = deformedIceClassifier(glcm_filelist, ridges_filelist, level_filelist,
                                  defo_filelist, s0_filelist, defo_training=False)

# Train classifier
clf_no_defo.train_rf_classifier(n_estimators=200, n_jobs=10, max_depth=100, max_samples=0.2)

# Open file with texture features for detection
reader = dataReader()
data_textures = reader.read_nc('MOIRA_test_data/textures/norm_s0_vv_S1B_IW_GRDH_1SDV_20201222T203955_20201222T204024_024820_02F3F4_625F_*.nc')
ft_names = [str(ft_name) for ft_name in range(len(data_textures['data'].keys()))]
z_dim = len(ft_names)
fts = np.zeros((z_dim, data_textures['data'][ft_names[0]].shape[0],
                                           data_textures['data'][ft_names[0]].shape[1]), dtype=np.float)
for iz in range(z_dim):
    fts[iz, :, :] = data_textures['data'][ft_names[iz]]

# Move axis with z-dimension 
fts = np.moveaxis(fts, 0, 2)

# Detect the deformation state of sea ice
res_no_defo = clf_no_defo.detect_ice_state(fts)

# Plot obtained result (zoomed area)
# Bbox for visualization
r1, r2, c1, c2  = 810, 1200, 1410, 1900

plt.imshow(res_no_defo[r1:r2,c1:c2], interpolation='nearest')

# Next we perform ridge detection with deformation data
clf_defo = deformedIceClassifier(glcm_filelist, ridges_filelist, level_filelist,
                                  defo_filelist, s0_filelist, defo_training=True)

# Add files with texture features and deformation
reader = dataReader()
data_textures = reader.read_nc('MOIRA_test_data/textures/norm_s0_vv_S1B_IW_GRDH_1SDV_20201222T203955_20201222T204024_024820_02F3F4_625F_*.nc')
ft_names = [str(ft_name) for ft_name in range(len(data_textures['data'].keys()))]
z_dim = len(ft_names)
fts = np.zeros((z_dim, data_textures['data'][ft_names[0]].shape[0],
                                           data_textures['data'][ft_names[0]].shape[1]), dtype=np.float)
for iz in range(z_dim):
    fts[iz, :, :] = data_textures['data'][ft_names[iz]]

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

To calculate SAR texture characteristics which are GLCM features a method called `calcTexFt()` from `SarTextures` is used:

```python
t.calcTexFt()
```

Then the obtained result can be stored in NetCDF4 file:

```python
t.export_netcdf('PATH/TO/OUTPUT/NETCDF')
```

The package was created as a part of MOdel for Sea Ice RAdar Image Microstructure (MOIRA)
project for Europian Space Agency (ESA).