from osgeo import gdal
from osgeo import ogr
from osgeo import osr

# inputs
base_raster = sys.argv[1]
line_file = sys.argv[2]

# ouput
output_file = sys.argv[3]

# processing
raster_ds = gdal.Open(base_raster, 0)
geo_transform = raster_ds.GetGeoTransform()
projection = raster_ds.GetProjection()
x_min = geo_transform[0]
y_max = geo_transform[3]
x_max = x_min + geo_transform[1] * raster_ds.RasterXSize
y_min = y_max + geo_transform[5] * raster_ds.RasterYSize
width = raster_ds.RasterXSize
height = raster_ds.RasterYSize

x_res = geo_transform[1]
y_res = geo_transform[5]

# reproject vector
line_ds = ogr.Open(line_file)
line_layer = line_ds.GetLayer()
sourceprj = line_layer.GetSpatialRef()
targetprj = osr.SpatialReference(wkt = raster_ds.GetProjection())
transform = osr.CoordinateTransformation(sourceprj, targetprj)

mem = ogr.GetDriverByName("ESRI Shapefile")
line_layer_reprojected = mem.CreateDataSource("temp.shp")
outlayer = line_layer_reprojected.CreateLayer('', targetprj, ogr.wkbLineString)

for feature in line_layer:
    transformed = feature.GetGeometryRef()
    transformed.Transform(transform)
    geom = ogr.CreateGeometryFromWkb(transformed.ExportToWkb())
    defn = outlayer.GetLayerDefn()
    feat = ogr.Feature(defn)
    feat.SetGeometry(geom)
    outlayer.CreateFeature(feat)
    feat = None

del line_layer_reprojected

rasterized = gdal.Rasterize('MEM', 'temp.shp', format='MEM',
                   outputBounds=[x_min, y_min, x_max, y_max],
                   xRes=x_res, yRes=y_res, burnValues=[1])

rasterized_projected = gdal.Warp(output_file, rasterized, format='GTiff', dstSRS=projection)

del rasterized
del rasterized_projected