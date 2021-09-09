# MOIRA

The package allows to perform classification of ridged and flat ice from SAR image based on texture characteristics and ice deformation data.
There are to way for the classification - using ice dformation data along with SAR textures and using only SAR texture information. We reccomend to use the both data sources as it is more robust to discriminate between ridged ice and young ice in leads.

The package also includes some tools that helps to prepare the data for classification. 

The following Quickstart guide covers all the essentials steps to perform sea ice ridge classification in step by step manner.

1. Open Sentinel-1 image

First you should import all classes from the package

```python
	from ridge_classification import *
```

Then a Sentinel-1 image can be initialized by `SarTextures` class

```python
	t = SarTextures(PATH/TO/S1/FILE, ws=11, stp=15, threads=10)
```

where `PATH/TO/S1/FILE` is a path to Sentinel-1 GRD (EW/IW) Level-1 file; `ws` - windows size for texture features computation; `stp` - computational grid step size; `threads` - numner of threads.

2. Calibrate and project Sentinel-1 image

Peform data calibration and projection into Polar Stereographic projection (EPSG:5041) with a spatial resolution of `res` [meters].

```python
	t.calibrate_project(5041, res, mask=False, write_file=False, out_path='/OUTPUT/DIRECTORY')	
```

other parameters includes: `write_file` - allows to export the calibrated data as a geotiff file; `out_path` - ouput directory to store a file. 

Vector data preparation
The classification require reference data for the training. The common way to do that is manual ridged and flat ice mapping. Once it is done, a vector file should be rasterized: