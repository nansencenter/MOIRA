# MOIRA

The package allows to perform classification of ridged and flat ice from SAR image based on texture characteristics and ice deformation data.
There are to way for the classification - using ice deformation data along with SAR textures and using only SAR texture information. We reccomend to use the both data sources as a more robust way to overcome obstacles caused by the similar signature of ridged ice and young ice in leads but the deformation data is optional for the algorithm.

The package also includes some tools that helps to prepare the data for classification. 

The following Quickstart guide covers all the essentials steps to perform sea ice ridge classification in step by step manner.

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

Then the obtaned results can be stored in NetCDF4 file:

```python
	t.export_netcdf('PATH/TO/OUTPUT/NETCDF')
```

## 5. Ridged ice detection

To classify ice into two classes: ridged/non-ridged sea ice a class first you need to train a classifier. To do that data for trainig should be prepared and passed into class `ridgedIceClassifier`

```python
	ridgedIceClassifier(glcm_filelist, ridges_filelist, flat_filelist, defo_filelist, defo_training=False)
```

where `glcm_filelist` - file list containing SAR texture features calculated during Step 4; `ridges_filelist` - file list containing rasterized ridges in geotiff format; `flat_filelist` - file list containing rasterized flat ice regions in geotiff format; `defo_filelist` (optional) - file list containing ice deformation data; `defo_training` - should be set to True if you want to use ice deformation data for training.