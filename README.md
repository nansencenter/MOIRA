# MOIRA

The package allows to perform classification of ridged and flat ice from SAR image based on texture characteristics and ice deformation data.
There are to way for the classification - using ice dformation data along with SAR textures and using only SAR texture information. We reccomend to use the both data sources as a more robust way to overcome obstacles caused by the similar signature of ridged ice and young ice in leads but the deformation data is optional for the algorithm.

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

where `PATH/TO/S1/FILE` is a path to Sentinel-1 GRD (EW/IW) Level-1 file; `ws` - windows size for texture features computation; `stp` - computational grid step size; `threads` - numner of threads.

## 2. Calibrate and project Sentinel-1 image

Peform data calibration and projection into Polar Stereographic projection (EPSG:5041) with a spatial resolution of `res` [meters].

```python
	t.calibrate_project(5041, res, mask=False, write_file=False, out_path='/OUTPUT/DIRECTORY')	
```

other parameters includes: `write_file` - allows to export the calibrated data as a geotiff file; `out_path` - ouput directory to store a file. 

## 3. Vector data preparation

The classification require reference data for the training. The common way to produce that is manual mapping of ridged and flat ice using one of GIS software. Once it is done, a vector file should be initialized by class for vector data processing called `python VectorData`:

```python

v = VectorData('/PATH/TO/VECTOR/FILE', t.ds[list(t.ds.keys())[0]], downsample=True)
```

where `python t.ds[list(t.ds.keys())[0]]` is a gdal object with a projected geotiff from a previous step and we also set a `python downsample` parameter to True to make further computations more fast. 


```python
v.rasterize('PATH/TO/RASTERIZED/GEOTIFF/TIFE')

```