from osgeo import gdal
from osgeo import ogr
from osgeo import osr
from netCDF4 import Dataset
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from skimage import filters
import time
from multiprocessing import Pool
import sys
from skimage.feature import greycomatrix
import zipfile
import shutil
import xml.etree.ElementTree
from scipy import interpolate, ndimage
import glob
import pyresample
from datetime import datetime
import re

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from skimage import feature, future
from functools import partial

from itertools import combinations_with_replacement
import itertools
import numpy as np
from skimage import filters, feature
from skimage.util.dtype import img_as_float32
from skimage._shared import utils
from concurrent.futures import ThreadPoolExecutor

class SarImage:
    '''
    Basic S1 L1 GRD image processing
    '''

    def __init__(self, zipPath, bbox=None):
        self.name = zipPath
        self.bbox = bbox
        self.data = {}

        # s0 range in dB
        self.db_min = -35
        self.db_max = -5

        # Grayscale range
        self.gl_min = 1
        self.gl_max = 255

        self.meta = {}
        self.ds = {}

    def transform_gcps(self, gcp_list, ct):
        new_gcp_list = []
        for gcp in gcp_list:
            # point = ogr.CreateGeometryFromWkt("POINT (%s %s)" % (gcp.GCPX, gcp.GCPY))

            # Check gdal version first
            ss = gdal.__version__
            if int(ss[0]) >= 3:
                xy_target = ct.TransformPoint(gcp.GCPY, gcp.GCPX)
            else:
                xy_target = ct.TransformPoint(gcp.GCPX, gcp.GCPY)

            new_gcp_list.append(
                gdal.GCP(xy_target[0], xy_target[1], 0, gcp.GCPPixel, gcp.GCPLine))  # 0 stands for point elevation
        return new_gcp_list

    def get_transformation(self, target):
        source = osr.SpatialReference()
        source.ImportFromEPSG(4326)
        ct = osr.CoordinateTransformation(source, target)
        return ct

    def getLatsLons(self):
        '''
        Get matrices with lats and lons from GeoTIFF file for pixels with GLCM values
        '''

        rasterArray = self.ds[self.meta['ds_fname']].ReadAsArray()

        rows, columns = rasterArray.shape[0], rasterArray.shape[1]

        lon_2d = np.empty((rows, columns))
        lon_2d[:] = np.nan

        lat_2d = np.empty((rows, columns))
        lat_2d[:] = np.nan

        geotransform = self.ds[self.meta['ds_fname']].GetGeoTransform()
        old_cs = osr.SpatialReference()
        old_cs.ImportFromWkt(self.ds[self.meta['ds_fname']].GetProjection())
        new_cs = osr.SpatialReference()
        new_cs.ImportFromProj4('+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs')
        transform = osr.CoordinateTransformation(old_cs, new_cs)
        pixelWidth = geotransform[1]
        pixelHeight = geotransform[-1]

        # 2D indexes for all elements
        idxs_2d = []

        #! TODO: replace with this
        # Generate Longitude and Latitude array
        #xx1 = geotransform[0] + np.arange(0, rows) * pixelWidth
        #yy1 = geotransform[3] + np.arange(0, columns) * pixelHeight

        #lon_2d, lat_2d = transform.TransformPoint(geotransform[0] + np.arange(0, rows) * pixelWidth,
        #                                  geotransform[3] + np.arange(0, columns) * pixelHeight)

        # Compute 2d arrays of lat lon
        for i, idx_line in enumerate(range(0, rows)):
            sys.stdout.write('\r%s of %s' % (i, rows))
            sys.stdout.flush()
            for j, idx_col in enumerate(range(0, columns)):
                # Convert x and y to lon lat
                xx1 = geotransform[0] + idx_col * pixelWidth
                yy1 = geotransform[3] + idx_line * pixelHeight

                # Check gdal version first
                ss = gdal.__version__
                if int(ss[0]) >= 3:
                    latlon = transform.TransformPoint(float(xx1), float(yy1))
                else:
                    latlon = transform.TransformPoint(float(yy1), float(xx1))

                ilon = latlon[0]
                ilat = latlon[1]

                if np.isnan(ilon):
                    print('Row, Column: %s, %s' % (idx_line, idx_col))

                lon_2d[i, j] = ilon
                lat_2d[i, j] = ilat

        self.lats = lat_2d
        self.lons = lon_2d

        del rasterArray

    def get_coefficients_array(self, xml_path, xml_element_name, xml_attribute_name, cols, rows):
        '''
        Get calibration coefficients
        '''
        coefficients_rows = []
        e = xml.etree.ElementTree.parse(xml_path).getroot()
        print('reading data...')
        for noiseVectorList in e.findall(xml_element_name):
            for child in noiseVectorList:
                for param in child:
                    if param.tag == 'pixel':
                        currentPixels = str(param.text).split()
                    if param.tag == xml_attribute_name:
                        currentValues = str(param.text).split()

                i = 0
                currentRow = np.empty([1, cols])
                currentRow[:] = np.nan
                while i < len(currentPixels):
                    currentRow[0, int(currentPixels[i])] = float(currentValues[i])
                    i += 1

                currentRow = self.fill_nan(currentRow)

                coefficients_rows.append(currentRow[0])

        print('interpolating data...')

        zoom_x = float(cols) / len(coefficients_rows[0])
        zoom_y = float(rows) / len(coefficients_rows)

        return ndimage.zoom(coefficients_rows, [zoom_y, zoom_x])

    def fill_nan(self, A):
        '''
        Fill NaN values by linear interpolation
        '''
        B = A
        ok = ~np.isnan(B)
        xp = ok.ravel().nonzero()[0]
        fp = B[~np.isnan(B)]
        x = np.isnan(B).ravel().nonzero()[0]
        B[np.isnan(B)] = np.interp(x, xp, fp)
        return B

    def radiometric_calibration(self, input_tiff_path, calibration_xml_path, backscatter_coeff='sigmaNought'):
        '''
        Apply calibration values to DN
        '''

        measurement_file = gdal.Open(input_tiff_path)
        measurement_file_array = np.array(measurement_file.GetRasterBand(1).ReadAsArray().astype(np.float32))

        radiometric_coefficients_array = self.get_coefficients_array(calibration_xml_path, 'calibrationVectorList',
                                                                     backscatter_coeff,
                                                                     measurement_file.RasterXSize,
                                                                     measurement_file.RasterYSize)
        print('Radiometric calibration...')
        tiff_name = os.path.basename(input_tiff_path)
        self.data[tiff_name] = (measurement_file_array * measurement_file_array) / \
                               (radiometric_coefficients_array * radiometric_coefficients_array)
        print('Done.\n')
        # save_array_as_geotiff_gcp_mode(calibrated_array, output_tiff_path, measurement_file)

    def noise_calibration(self, input_tiff_path, calibration_xml_path,
                          noise_xml_path, backscatter_coeff='sigmaNought'):
        measurement_file = gdal.Open(input_tiff_path)
        measurement_file_array = np.array(measurement_file.GetRasterBand(1).ReadAsArray().astype(np.float32))

        print('Radiometric calibration and thermal noise removal...')
        radiometric_coefficients_array = self.get_coefficients_array(calibration_xml_path, 'calibrationVectorList',
                                                                     backscatter_coeff,
                                                                     measurement_file.RasterXSize,
                                                                     measurement_file.RasterYSize)

        noise_coefficients_array = self.get_coefficients_array(noise_xml_path, 'noiseRangeVectorList',
                                                               'noiseRangeLut',
                                                               measurement_file.RasterXSize,
                                                               measurement_file.RasterYSize)

        tiff_name = os.path.basename(input_tiff_path)
        tmp_data = (measurement_file_array * measurement_file_array - noise_coefficients_array) / \
                   (radiometric_coefficients_array * radiometric_coefficients_array)
        tmp_data[tmp_data < 0] = 0.
        self.data[tiff_name] = tmp_data

    def calibrate_project(self, t_srs, res, mask=False, write_file=True, out_path='.', backscatter_coeff='sigmaNought'):
        '''
        Project raw S1 GeoTIFF file
        '''

        tiffPaths = []

        # unzip file
        unzipFolderName = os.path.basename(self.name)
        with zipfile.ZipFile(self.name, 'r') as zip_ref:
            zip_ref.extractall('')

        for ifile in zip_ref.filelist:
            if ifile.filename.endswith('tiff'):
                tiffPaths.append(ifile.filename)

        # Creating coordinate transformation
        target = osr.SpatialReference()
        target.ImportFromEPSG(t_srs)
        ct = self.get_transformation(target)

        # TODO
        print('\nPaths to tiff files: %s\n' % tiffPaths)

        for tiffPath in tiffPaths:
            print('Prcessing %s ...' % os.path.basename(tiffPath))

            # Find calibration and noise annotation files
            calib_f = glob.glob(
                os.path.dirname(tiffPaths[0]).replace('measurement', 'annotation/calibration/*calibration*%s*.xml' %
                                                      os.path.basename(tiffPath).split('.')[0]))[0]
            noise_f = \
            glob.glob(os.path.dirname(tiffPaths[0]).replace('measurement', 'annotation/calibration/*noise*%s*.xml' %
                                                            os.path.basename(tiffPath).split('.')[0]))[0]

            if calib_f:
                print('\nStart calibration...')
                print('Calibration file: %s' % calib_f)
                print('Noise file: %s' % noise_f)

                # Radiometric calibration + denoise
                #self.noise_calibration(tiffPath, calib_f, noise_f, backscatter_coeff)

                # Radiometric calibration
                self.radiometric_calibration(tiffPath, calib_f, backscatter_coeff)

                print('Done.\n')

                print('\nOpening raw Geotiff {}'.format(tiffPath))
                ds = gdal.Open(tiffPath)
                dt = ds.GetGeoTransform()

                gcpList = ds.GetGCPs()
                new_gcp_list = self.transform_gcps(gcpList, ct)

                driver = gdal.GetDriverByName("VRT")
                copy_ds = driver.CreateCopy("", ds)
                copy_ds.SetGCPs(new_gcp_list, target.ExportToWkt())

                clb = gdal.TermProgress
                ds_warp = gdal.Warp('', copy_ds, format="MEM", dstSRS="EPSG:%s" % t_srs,
                                    xRes=res, yRes=res, multithread=True, callback=clb)

                driver = gdal.GetDriverByName('GTiff')

                [rows, cols] = self.data[os.path.basename(tiffPath)].shape

                try:
                    os.remove('temp.tif')
                except:
                    pass

                outdata = driver.Create('temp.tif', cols, rows, 1, gdal.GDT_Float32)
                outdata.SetGCPs(new_gcp_list, target.ExportToWkt())

                data = self.data[os.path.basename(tiffPath)]

                arr_out = 10 * np.log10(data)
                arr_out[np.isinf(arr_out)] = np.nan

                arr_mask = np.copy(arr_out)

                # arr_mask[~np.isnan(arr)] = 255
                arr_mask[arr_mask == 0] = np.nan

                outdata.GetRasterBand(1).WriteArray(arr_mask)
                outdata.GetRasterBand(1).SetNoDataValue(np.nan)
                outdata.FlushCache()
                outdata = None

                tmp_ds = gdal.Open('temp.tif')
                var_name = os.path.basename(tiffPath).split('.')[0]
                pol = var_name.split('-')[3]

                if write_file:
                    out_fname = '%s/%s' % (out_path, os.path.basename(tiffPath))
                    os.makedirs(os.path.dirname(out_fname), exist_ok=True)

                    if self.bbox:
                        print(f'\nCropping data {self.bbox}')

                        ds_warp2 = gdal.Warp(out_fname, tmp_ds, format="GTiff", dstSRS="EPSG:%s" % t_srs,
                                             xRes=res, yRes=res, multithread=True, callback=clb,
                                             outputBoundsSRS='EPSG:4326',
                                             outputBounds=self.bbox)
                        print('Done\n')
                    else:
                        ds_warp2 = gdal.Warp(out_fname, tmp_ds, format="GTiff", dstSRS="EPSG:%s" % t_srs,
                                             xRes=res, yRes=res, multithread=True, callback=clb)

                    self.data['s0_%s' % pol] = {}
                    self.data['s0_%s' % pol]['data'] = ds_warp2.ReadAsArray()
                    self.data['s0_%s' % pol]['units'] = 'dB'
                    self.data['s0_%s' % pol]['scale_factor'] = 1.

                    ds_fname = 'ds_s0_%s' % var_name
                    self.ds[ds_fname] = ds_warp2

                    self.meta['ds_fname'] = ds_fname
                else:
                    # ds_warp2 = gdal.Warp('', tmp_ds, format="MEM", dstSRS="EPSG:%s" % t_srs,
                    #                    xRes=res, yRes=res, multithread=True, callback=clb)

                    if self.bbox:
                        print(f'\nCropping data {self.bbox}')
                        ds_warp2 = gdal.Warp('', tmp_ds, format="GTiff", dstSRS="EPSG:%s" % t_srs,
                                             xRes=res, yRes=res, multithread=True, callback=clb,
                                             outputBoundsSRS='EPSG:4326',
                                             outputBounds=self.bbox)
                        print('Done\n')
                    else:
                        ds_warp2 = gdal.Warp('', tmp_ds, format="GTiff", dstSRS="EPSG:%s" % t_srs,
                                             xRes=res, yRes=res, multithread=True, callback=clb)

                    self.data['s0_%s' % pol] = {}
                    self.data['s0_%s' % pol]['data'] = ds_warp2.ReadAsArray()
                    self.data['s0_%s' % pol]['units'] = 'dB'
                    self.data['s0_%s' % pol]['scale_factor'] = 1.

                    ds_fname = 'ds_s0_%s' % var_name
                    self.ds[ds_fname] = ds_warp2

                    self.meta['ds_fname'] = ds_fname

                del self.data[os.path.basename(tiffPath)]

                print('Done.\n')

                ds = None
                band = None
                ds_warp = None
                copy_ds = None
                temp_ds = None
                ds_warp2 = None
                tmp_ds = None

                os.remove('temp.tif')
            else:
                print('Error: Calibration file did not found!\n')

        # Remove zip folder
        shutil.rmtree(zip_ref.filelist[0].filename, ignore_errors=True)

    def func(self, name):
        return getattr(self, name)

    def normalizeInt(self, r):
        '''
        Rescale data into defined range of integer values
        '''
        # r - range of values

        self.norm_data = {}

        for iband in self.data.keys():
            arr = self.data[iband]['data']
            arr[np.isnan(arr)] = 0
            self.norm_data['norm_%s' % iband] = arr #(r * (arr - np.min(arr)) / np.ptp(arr)).astype(int)

    def export_netcdf(self, lats, lons, data, out_fname):
        '''
        Generic method to make NetCDF4 file from dictonary
        of data (data, scale, units) and lat lon matrices

        '''

        try:
            os.remove(out_fname)
        except OSError:
            pass

        ds = Dataset(out_fname, 'w', format='NETCDF4_CLASSIC')

        # Dimensions
        y_dim = ds.createDimension('y', lons.shape[0])
        x_dim = ds.createDimension('x', lons.shape[1])
        time_dim = ds.createDimension('time', None)
        # data_dim = ds.createDimension('data', len([k for k in data.keys()]))

        # Variables
        times = ds.createVariable('time', np.float64, ('time',))
        latitudes = ds.createVariable('lat', np.float32, ('y', 'x',))
        longitudes = ds.createVariable('lon', np.float32, ('y', 'x',))

        for var_name in data.keys():
            globals()[var_name] = ds.createVariable(var_name, np.float32, ('y', 'x',))
            globals()[var_name][:] = data[var_name]['data']
            globals()[var_name].units = data[var_name]['units']
            globals()[var_name].scale_factor = data[var_name]['scale_factor']

        # Global Attributes
        ds.description = ''
        ds.history = 'Created ' + time.ctime(time.time())
        ds.source = 'NERSC'

        # Variable Attributes
        latitudes.units = 'degree'
        longitudes.units = 'degree'
        times.units = 'hours since 0001-01-01 00:00:00'
        times.calendar = 'gregorian'

        # Put variables
        latitudes[:, :] = lats
        longitudes[:, :] = lons

        ds.close()
        print('\n%s has been successefully created.\n' % out_fname)
        self.nc_file = out_fname


class SarTextures(SarImage):
    '''
    Class for SAR textural features computation
    '''

    def __init__(self, zipPath, ws=50, stp=5, threads=10, bbox=None):
        super().__init__(zipPath)
        self.ws = ws
        self.stp = stp
        self.threads = threads
        self.bbox = bbox

        self.names_glcm = {
            1: "Angular Second Moment",
            2: "Contrast",
            3: "Correlation",
            4: "Sum of Squares: Variance",
            5: "Inverse Difference Moment",
            6: "Sum Average",
            7: "Sum Variance",
            8: "Sum Entropy",
            9: "Entropy",
            10: "Difference Variance",
            11: "Difference Entropy",
            12: "Information Measures of Correlation",
            13: "Maximal Correlation Coefficient"
        }

    @classmethod
    def computeHaralickTextureFeatures(cls, subimage):
        '''
            Modified from the implementation of MAHOTAS package, see
            https://github.com/luispedro/mahotas/blob/master/mahotas/features/texture.py
            For GLCM calculation, this uses greycomatrix from SCIKIT-IMAGE rather than following the
            implementation in MAHOTAS.
            To speed up, the averaging process is done in GLCM level rather than texture feature level.

        '''

        if (subimage == 0).all():
            haralick = np.zeros(13) + np.nan
        else:
            def _entropy(p):
                p = p.ravel()
                p1 = p.copy()
                p1 += (p == 0)
                return -np.dot(np.log2(p1), p)

            cooccuranceDistances = range(1, np.min(subimage.shape) // 2)
            directions = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
            glcmDim = int(np.max(subimage) + 1)
            glcm = greycomatrix(subimage, distances=cooccuranceDistances, \
                                angles=directions, levels=glcmDim, \
                                symmetric=True, normed=False)
            glcm[0, :, :, :] = 0  # ignore zeros
            glcm[:, 0, :, :] = 0  # ignore zeros
            glcm = glcm / glcm.sum(axis=(0, 1))  # normalize
            glcm[np.isnan(glcm)] = 0
            glcm = glcm.mean(axis=(2, 3))  # take mean along distances and directions
            glcm = glcm / glcm.sum()  # normalize
            cmat = glcm
            haralick = np.zeros(13, np.double)
            T = cmat.sum()
            maxv = len(cmat)
            k = np.arange(maxv)
            k2 = k ** 2
            tk = np.arange(2 * maxv)
            tk2 = tk ** 2
            i, j = np.mgrid[:maxv, :maxv]
            ij = i * j
            i_j2_p1 = (i - j) ** 2
            i_j2_p1 += 1
            i_j2_p1 = 1. / i_j2_p1
            i_j2_p1 = i_j2_p1.ravel()
            p = cmat / float(T)
            pravel = p.ravel()
            px = p.sum(0)
            py = p.sum(1)
            k_plus = i + j
            px_plus_y = np.array([np.sum(p[k_plus == k]) for k in np.arange(2 * maxv)])
            k_minus = abs(i - j)
            px_minus_y = np.array([np.sum(p[k_minus == k]) for k in np.arange(0, maxv)])
            ux = np.dot(px, k)
            uy = np.dot(py, k)
            vx = np.dot(px, k2) - ux ** 2
            vy = np.dot(py, k2) - uy ** 2
            sx = np.sqrt(vx)
            sy = np.sqrt(vy)
            haralick[0] = np.dot(pravel, pravel)
            haralick[1] = np.dot(k2, px_minus_y)
            if sx == 0. or sy == 0.:
                haralick[2] = 1.
            else:
                haralick[2] = (1. / sx / sy) * (np.dot(ij.ravel(), pravel) - ux * uy)
            haralick[3] = vx
            haralick[4] = np.dot(i_j2_p1, pravel)
            haralick[5] = np.dot(tk, px_plus_y)
            haralick[6] = np.dot(tk2, px_plus_y) - haralick[5] ** 2
            haralick[7] = _entropy(px_plus_y)
            haralick[8] = _entropy(pravel)
            haralick[9] = px_minus_y.var()
            haralick[10] = _entropy(px_minus_y)
            HX = _entropy(px)
            HY = _entropy(py)
            crosspxpy = np.outer(px, py)
            crosspxpy += (crosspxpy == 0)
            crosspxpy = crosspxpy.ravel()
            HXY1 = -np.dot(pravel, np.log2(crosspxpy))
            HXY2 = _entropy(crosspxpy)
            if max(HX, HY) == 0.:
                haralick[11] = (haralick[8] - HXY1)
            else:
                haralick[11] = (haralick[8] - HXY1) / max(HX, HY)
            haralick[12] = np.sqrt(max(0, 1 - np.exp(-2. * (HXY2 - haralick[8]))))

        return haralick

    def getTextureFeatures(self, iarray):
        ''' Calculate Haralick texture features
            using mahotas package and scikit-image package
        Parameters
        ----------
            iarray : ndarray
                2D input data with gray levels
            ws : int
                size of subwindow
            stp : int
                step of sub-window floating
            threads : int
                number of parallel processes
        Returns
        -------
            harImageAnis : ndarray
                [13 x ROWS x COLS] array with texture features descriptors
                13 - nuber of texture features
                ROWS = rows of input image / stp
                COLS = rows of input image / stp
        '''

        # iarray = self.normnormalizeInt(iarray, 255)

        # init parallel processing
        pool = Pool(self.threads)

        # apply calculation of Haralick texture features in many threads
        # in row-wise order
        print('Compute GLCM and extract Haralick texture features')
        harList = []
        for r in range(0, iarray.shape[0] - self.ws + 1, self.stp):
            sys.stdout.write('\rRow number: %5d' % r)
            sys.stdout.flush()
            # collect all subimages in the row into one list
            subImgs = [iarray[r:r + self.ws, c:c + self.ws] for c in range(0, iarray.shape[1] - self.ws + 1, self.stp)]

            # calculate Haralick texture features in all sub-images in this row
            harRow = pool.map(self.computeHaralickTextureFeatures, subImgs)

            # keep vectors with calculated texture features
            harList.append(np.array(harRow))

        # terminate parallel processing. THIS IS IMPORTANT!!!
        pool.close();
        pool.terminate();
        pool.join();

        # convert list with texture features to array
        harImage = np.array(harList)

        # reshape matrix and make images to be on the first dimension
        return np.moveaxis(harImage, 2, 0)

    def anisodiff(self, img, niter=20, kappa=50, gamma=0.25, step=(1., 1.), option=1):
        """
        Anisotropic diffusion.

        Usage:
        imgout = anisodiff(im, niter, kappa, gamma, option)

        Arguments:
                img    - input image
                niter  - number of iterations
                kappa  - conduction coefficient 20-100 ?
                gamma  - max value of .25 for stability
                step   - tuple, the distance between adjacent pixels in (y,x)
                option - 1 Perona Malik diffusion equation No 1
                         2 Perona Malik diffusion equation No 2

        Returns:
                imgout   - diffused image.

        kappa controls conduction as a function of gradient.  If kappa is low
        small intensity gradients are able to block conduction and hence diffusion
        across step edges.  A large value reduces the influence of intensity
        gradients on conduction.

        gamma controls speed of diffusion (you usually want it at a maximum of
        0.25)

        step is used to scale the gradients in case the spacing between adjacent
        pixels differs in the x and y axes

        Diffusion equation 1 favours high contrast edges over low contrast ones.
        Diffusion equation 2 favours wide regions over smaller ones.

        Reference:
        P. Perona and J. Malik.
        Scale-space and edge detection using ansotropic diffusion.
        IEEE Transactions on Pattern Analysis and Machine Intelligence,
        12(7):629-639, July 1990.

        Original MATLAB code by Peter Kovesi
        School of Computer Science & Software Engineering
        The University of Western Australia
        pk @ csse uwa edu au
        <http://www.csse.uwa.edu.au>

        Translated to Python and optimised by Alistair Muldal

        Sep 2017 modified by Denis Demchev
        """

        if img.ndim == 3:
            warnings.warn("Only grayscale images allowed, converting to 2D matrix")
            img = img.mean(2)

        # initialize output array
        img = img.astype('float32')
        imgout = img.copy()

        # initialize some internal variables
        deltaS = np.zeros_like(imgout)
        deltaE = deltaS.copy()
        NS = deltaS.copy()
        EW = deltaS.copy()
        gS = np.ones_like(imgout)
        gE = gS.copy()

        for ii in range(int(niter)):
            # calculate the diffs
            deltaS[:-1, :] = np.diff(imgout, axis=0)
            deltaE[:, :-1] = np.diff(imgout, axis=1)

            # conduction gradients (only need to compute one per dim!)
            if option == 1:
                gS = np.exp(-(deltaS / kappa) ** 2.) / step[0]
                gE = np.exp(-(deltaE / kappa) ** 2.) / step[1]
            elif option == 2:
                gS = 1. / (1. + (deltaS / kappa) ** 2.) / step[0]
                gE = 1. / (1. + (deltaE / kappa) ** 2.) / step[1]

            # update matrices
            E = gE * deltaE
            S = gS * deltaS

            # subtract a copy that has been shifted 'North/West' by one pixel
            NS[:] = S
            EW[:] = E
            NS[1:, :] -= S[:-1, :]
            EW[:, 1:] -= E[:, :-1]

            # update the image
            imgout += gamma * (NS + EW)

        return imgout

    def _texture_filter(self, anisodiff_filtered):
        H_elems = [
            np.gradient(np.gradient(anisodiff_filtered)[ax0], axis=ax1)
            for ax0, ax1 in combinations_with_replacement(range(anisodiff_filtered.ndim), 2)
        ]
        eigvals = feature.hessian_matrix_eigvals(H_elems)
        return eigvals

    def _singlescale_basic_features_singlechannel(self,
            img, niter, intensity=True, edges=False, texture=True
                                                  ):
        results = ()
        anisodiff_filtered = self.anisodiff(img, niter=niter)

        if intensity:
            results += (anisodiff_filtered,)
        if edges:
            results += (filters.sobel(anisodiff_filtered),)
        if texture:
            results += (*self._texture_filter(anisodiff_filtered),)
        return results

    def _mutiscale_basic_features_singlechannel(self,
            img,
            intensity=True,
            edges=False,
            texture=True,
            iter_min=1,
            iter_max=20,
            iter_step=2,
            num_workers=None
                                                ):
        """Features for a single channel nd image.

        Returns
        -------
        features : list
            List of features, each element of the list is an array of shape as img.
        """
        # computations are faster as float32
        img = np.ascontiguousarray(img_as_float32(img))
        iters = range(iter_min, iter_max, iter_step)

        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            out_iters = list(
                ex.map(
                    lambda s: self._singlescale_basic_features_singlechannel(
                        img, s, intensity=intensity, edges=edges, texture=texture
                    ),
                    iters,
                )
            )
        features = itertools.chain.from_iterable(out_iters)
        return features

    def multiscale_basic_features(self,
            image,
            intensity=True,
            edges=False,
            texture=True,
            iter_min=1,
            iter_max=30,
            iter_step=5,
            num_workers=None,
            *,
            channel_axis=None
                                  ):
        """Local features for a single- or multi-channel nd image.
        Intensity, gradient intensity and local structure are computed at
        different scales thanks to anisotropic diffusion filtering.
        Modified from scikit-image code.

        Parameters
        ----------
        image : ndarray
            Input image, which can be grayscale or multichannel.
        multichannel : bool, default False
            True if the last dimension corresponds to color channels.
            This argument is deprecated: specify `channel_axis` instead.
        intensity : bool, default True
            If True, pixel intensities averaged over the different scales
            are added to the feature set.
        edges : bool, default True
            If True, intensities of local gradients averaged over the different
            scales are added to the feature set.
        texture : bool, default True
            If True, eigenvalues of the Hessian matrix after blurring
            at different scales are added to the feature set.
        iter_min : float, optional
            Smallest value of iterations with anisotropic diffusion.
        iter_max : float, optional
            Largest value of iterations with anisotropic diffusion.
        num_iter : int, optional
            Number of values of the anistropic diffusion.
            If None, 2 is used.
        num_workers : int or None, optional
            The number of parallel threads to use. If set to ``None``, the full
            set of available cores are used.
        channel_axis : int or None, optional
            If None, the image is assumed to be a grayscale (single channel) image.
            Otherwise, this parameter indicates which axis of the array corresponds
            to channels.

        Returns
        -------
        features : np.ndarray
            Array of shape ``image.shape + (n_features,)``. When `channel_axis` is
            not None, all channels are concatenated along the features dimension.
            (i.e. ``n_features == n_features_singlechannel * n_channels``)
        """
        if not any([intensity, edges, texture]):
            raise ValueError(
                "At least one of `intensity`, `edges` or `textures`"
                "must be True for features to be computed."
            )
        if channel_axis is None:
            image = image[..., np.newaxis]
            channel_axis = -1
        elif channel_axis != -1:
            image = np.moveaxis(image, channel_axis, -1)

        all_results = (
            self._mutiscale_basic_features_singlechannel(
                image[..., dim],
                intensity=intensity,
                edges=edges,
                texture=texture,
                iter_min=iter_min,
                iter_max=iter_max,
                iter_step=iter_step,
                num_workers=num_workers,
            )
            for dim in range(image.shape[-1])
        )
        features = list(itertools.chain.from_iterable(all_results))
        out = np.stack(features, axis=-1)
        return out

    def getMultiscaleTextureFeatures(self, iarray, intensity=True,
                                     edges=False, texture=True,
                                     iter_min=1, iter_max=20):

        ''' Local features for a single- or multi-channel nd image.
            Intensity, gradient intensity and local structure are computed at
            different scales thanks to Gaussian or anistotropic diffusion filtering.

        Parameters
        ----------
            iarray : ndarray
                2D input data with gray levels
            multichannel : bool, default False
                True if the last dimension corresponds to color channels.
                This argument is deprecated: specify `channel_axis` instead.
            intensity : bool, default True
                If True, pixel intensities averaged over the different scales
                are added to the feature set.
            edges : bool, default True
                If True, intensities of local gradients averaged over the different
                scales are added to the feature set.
            texture : bool, default True
                If True, eigenvalues of the Hessian matrix after Gaussian/Anisotropic diffusion filtering
                at different scales are added to the feature set.
            sigma_min : float, optional
                Smallest value of the Gaussian kernel used to average local
                neighbourhoods before extracting features.
            sigma_max : float, optional
                Largest value of the Gaussian kernel used to average local
                neighbourhoods before extracting features.
            num_sigma : int, optional
                Number of values of the Gaussian kernel between sigma_min and sigma_max.
                If None, sigma_min multiplied by powers of 2 are used.
            num_workers : int or None, optional
                The number of parallel threads to use. If set to ``None``, the full
                set of available cores are used.
            channel_axis : int or None, optional
                If None, the image is assumed to be a grayscale (single channel) image.
                Otherwise, this parameter indicates which axis of the array corresponds
                to channels.
            method: str, optional
                ['gaussian', 'anis_diffusion']
                if gaussian, the gaussian blurring is applied using sigma_min, sigma_max
                and num_sigma.
                if anis_diffusion, the gaussian anisotropic diffusion filtering is applied
                where sigma_min, sigma_max corresponds to minimum and maximum
                number of iterations.

        Returns
        -------
        features : np.ndarray(NUM_FT, NUM_ROWS, NUM_COLUMNS)
            Array of shape ``image.shape + (n_features,)``. When `channel_axis` is
            not None, all channels are concatenated along the features dimension.
            (i.e. ``n_features == n_features_singlechannel * n_channels``)
        '''

        features_func = partial(self.multiscale_basic_features,
                                intensity=intensity, edges=edges, texture=texture,
                                iter_min=iter_min, iter_max=iter_max)

        fts = features_func(iarray)

        return np.moveaxis(fts, 2, 0)

    def calcTexFt(self, speckle_supression=True, type_features='multiscale'):
        '''
        Calculate GLCM features
        '''

        self.glcm_features = {}
        print('\nCalculating GLCM texture features...')

        # Get lats lons first

        # Check if lat lon parameters are exist
        if hasattr(self, 'lats'):
            pass
        else:
            print('\nGetting lats lons...\n')
            self.getLatsLons()
            print('\nDone.\n')

        self.glcm_features['lats'] = self.lats #[0:-self.ws + 1, 0:-self.ws + 1][::self.stp, ::self.stp]
        self.glcm_features['lons'] = self.lons #[0:-self.ws + 1, 0:-self.ws + 1][::self.stp, ::self.stp]

        # self.smooth()

        # Check if data are normilized
        if hasattr(self, 'norm_data'):
            pass
        else:
            print('\nData normalization...\n')
            self.normalizeInt(255)

        start = time.time()
        print('\nStart texture features computation...')

        for iband in self.norm_data.keys():
            print(f'{iband}\n')
            self.glcm_features[iband] = {}

            if speckle_supression:
                if type_features == 'multiscale':
                    print('\nSpeckle supression and calculating MS texture features...\n')
                    texFts = self.getMultiscaleTextureFeatures(filters.median(self.norm_data[iband], np.ones((3, 3))))
                else:
                    print('\nSpeckle supression and calculating Haralick texture features...\n')
                    texFts = self.getTextureFeatures(filters.median(self.norm_data[iband], np.ones((3, 3))))
            else:
                if type_features == 'multiscale':
                    print('\nCalculating MS texture features...\n')
                    texFts = self.getMultiscaleTextureFeatures(self.norm_data[iband])
                else:
                    print('\nCalculating Haralick texture features...\n')
                    texFts = self.getTextureFeatures(self.norm_data[iband])

            # Adding GLCM data
            '''
            for i in range(len(texFts[:, 0, 0])):
                self.glcm_features[iband][self.names_glcm[i + 1]] = {'data': texFts[i, :, :], 'scale_factor': 1.,
                                                                     'units': ''}
                plt.imshow(texFts[i, :, :])
            '''

            for i in range(len(texFts[:, 0, 0])):
                self.glcm_features[iband][str(i)] = {'data': texFts[i, :, :], 'scale_factor': 1.,
                                                                     'units': ''}
        end = time.time()

        print('\nDone in %.2f minutes\n' % ((end - start) / 60.))

    def export_netcdf(self, out_fname):
        '''
        Export results to NetCDF file
        '''

        ks = [x for x in self.glcm_features.keys() if 's0' in x]
        for ivar in ks:
            out_fname_nc = '%s/%s_%s' % (os.path.dirname(out_fname), ivar, os.path.basename(out_fname))
            super().export_netcdf(self.glcm_features['lats'], self.glcm_features['lons'],
                                  self.glcm_features[ivar], out_fname_nc)

class GeoTiff:
    '''
    Basic processing of GeoTiff files
    '''

    def __init__(self, name):
        self.name = name
        ds = gdal.Open(self.name)
        self.data = ds.ReadAsArray()
        del ds

    def getLatsLons(self):
        '''
        Get matrices with lats and lons from GeoTIFF file
        '''

        ds = gdal.Open(self.name)
        if len(self.data.shape) == 2:
            rows, columns = self.data.shape[0], self.data.shape[1]
        else:
            rows, columns = self.data.shape[1], self.data.shape[2]

        lon_2d = np.empty((rows, columns))
        lon_2d[:] = np.nan

        lat_2d = np.empty((rows, columns))
        lat_2d[:] = np.nan

        geotransform = ds.GetGeoTransform()
        old_cs = osr.SpatialReference()
        old_cs.ImportFromWkt(ds.GetProjection())
        new_cs = osr.SpatialReference()
        new_cs.ImportFromProj4('+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs')
        transform = osr.CoordinateTransformation(old_cs, new_cs)
        pixelWidth = geotransform[1]
        pixelHeight = geotransform[-1]

        # 2D indexes for all elements
        idxs_2d = []

        # Compute 2d arrays of lat lon
        for i, idx_line in enumerate(range(0, rows)):
            for j, idx_col in enumerate(range(0, columns)):

                # Convert x and y to lon lat
                xx1 = geotransform[0] + idx_col * pixelWidth
                yy1 = geotransform[3] + idx_line * pixelHeight

                # Check gdal version first
                ss = gdal.__version__
                if int(ss[0]) >= 3:
                    latlon = transform.TransformPoint(float(xx1), float(yy1))
                else:
                    latlon = transform.TransformPoint(float(yy1), float(xx1))

                ilon = latlon[0]
                ilat = latlon[1]

                if np.isnan(ilon):
                    print('Row, Column: %s, %s' % (idx_line, idx_col))

                lon_2d[i, j] = ilon
                lat_2d[i, j] = ilat

        self.lats = lat_2d
        self.lons = lon_2d

        del ds


class VectorData:
    """
    Class for vector data processing

    Args:
        filepath (str): Path to shapefile to rasterize
        raster_ds (str/GDAL object): GDAL object or path to raster (geotiff) file

   """

    def __init__(self, filepath, raster_ds, downsample=False):
        self.filepath = filepath
        if isinstance(raster_ds, gdal.Dataset):
            self.raster_ds = raster_ds

            if downsample:
                print('\nDownsampling...\n')
                gt = self.raster_ds.GetGeoTransform()
                pixelSizeX = gt[1]
                pixelSizeY = -gt[5]
                self.raster_ds = gdal.Translate('', self.raster_ds, xRes=pixelSizeX * 3, yRes=pixelSizeX * 3,
                                                resampleAlg="bilinear", format='vrt')
                print('Done.\n')

        elif isinstance(raster_ds, str) and (raster_ds.endswith('tiff') or raster_ds.endswith('tif')):
            print(f'\nOpening {raster_ds} with GDAL...\n')
            self.raster_ds = gdal.Open(raster_ds)
            print('Done.\n')

            if downsample:
                print('\nDownsampling...\n')
                gt = self.raster_ds.GetGeoTransform()
                pixelSizeX = gt[1]
                pixelSizeY = -gt[5]
                self.raster_ds = gdal.Translate('', self.raster_ds, xRes=pixelSizeX * 3, yRes=pixelSizeX * 3,
                                                resampleAlg="bilinear", format='vrt')
                print('Done.\n')
        else:
            print(f'\nError: {raster_ds} is not supported.\n')

    def rasterize(self, out_filename):
        '''
        Raterizing shapefile
        '''

        # Get georefernecinf from raster ds
        geo_transform = self.raster_ds.GetGeoTransform()
        projection = self.raster_ds.GetProjection()
        x_min = geo_transform[0]
        y_max = geo_transform[3]
        x_max = x_min + geo_transform[1] * self.raster_ds.RasterXSize
        y_min = y_max + geo_transform[5] * self.raster_ds.RasterYSize
        width = self.raster_ds.RasterXSize
        height = self.raster_ds.RasterYSize

        x_res = geo_transform[1]
        y_res = geo_transform[5]

        # Open and reproject vector data
        line_ds = ogr.Open(self.filepath)
        line_layer = line_ds.GetLayer()
        sourceprj = line_layer.GetSpatialRef()
        targetprj = osr.SpatialReference(wkt=self.raster_ds.GetProjection())
        transform = osr.CoordinateTransformation(sourceprj, targetprj)

        mem = ogr.GetDriverByName("ESRI Shapefile")

        # Get the layer data type
        # get the first feature
        ft = line_layer.GetNextFeature()
        geometry = ft.GetGeometryRef()
        geom_type = geometry.GetGeometryName()

        layer_reprojected = mem.CreateDataSource("temp.shp")

        if geom_type == 'LINESTRING':
            print(geom_type)
            outlayer = layer_reprojected.CreateLayer('', targetprj, ogr.wkbLineString)
        elif geom_type == 'POINT':
            print(geom_type)
            outlayer = layer_reprojected.CreateLayer('', targetprj, ogr.wkbPoint)
        else:
            raise ValueError(f'Error: {geom_type} is no supported!')

        for feature in line_layer:
            transformed = feature.GetGeometryRef()
            transformed.Transform(transform)
            geom = ogr.CreateGeometryFromWkb(transformed.ExportToWkb())
            defn = outlayer.GetLayerDefn()
            feat = ogr.Feature(defn)
            feat.SetGeometry(geom)
            outlayer.CreateFeature(feat)
            feat = None

        del layer_reprojected

        rasterized = gdal.Rasterize('MEM', 'temp.shp', format='MEM', outputBounds=[x_min, y_min, x_max, y_max],
                                    xRes=x_res, yRes=y_res, burnValues=[1])

        rasterized_projected = gdal.Warp(out_filename, rasterized, format='GTiff', dstSRS=projection)

        del rasterized
        del rasterized_projected
        del self.raster_ds


class Resampler:
    '''
    Resampling of data from source grid onto another
    '''
    formats = {'nc': 'netcdf', 'tiff': 'geotiff', 'tif': 'geotiff'}

    def __init__(self, f_source, f_target):
        self.f_source = {}
        self.f_target = {}
        self.f_source['name'] = f_source
        self.f_target['name'] = f_target

        self.f_source['format'] = self.__check_file_format(self.f_source['name'])
        self.f_target['format'] = self.__check_file_format(self.f_target['name'])

        print('\nResampling source file: %s ...' % self.f_source['name'])
        if self.f_source['format'] == 'tif':
            self.f_source['format'] = 'tiff'
        func = getattr(self, 'read_%s' % self.f_source['format'])
        self.f_source = func(self.f_source['name'])
        print('Done.\n')

        print('\nResampling target file: %s ...' % self.f_target['name'])
        func = getattr(self, 'read_%s' % self.f_target['format'])
        self.f_target = func(self.f_target['name'])
        print('Done.\n')

    def __check_file_format(self, fname):
        filename, file_extension = os.path.splitext(fname)
        file_extension = file_extension[1:]

        if file_extension in [x for x in self.formats.keys()]:
            return file_extension
        else:
            print(f'\nSorry, {file_extension} is not supported.\n')

    def read_nc(self, f):
        '''
        Get data from NetCDF file
        '''
        var_geo_time = ['time', 'lon', 'lat']
        res = {}
        res['data'] = {}
        nc = Dataset(f, 'r')

        # Get variable names with data
        data_vars = [var for var in nc.variables if not var in var_geo_time]

        for var in data_vars:
            res['data'][var] = nc[var][:].data
            res['units'] = ''
            res['scale_factor'] = 1.

        res['lons'] = nc['lon'][:].data
        res['lats'] = nc['lat'][:].data

        del nc

        return res

    def read_tiff(self, f):
        '''
        Get data and lats lons from GeoTiff file
        '''
        f = GeoTiff(f)
        f.getLatsLons()

        res = {}
        res['data'] = {}
        res['data']['s0'] = f.data
        res['units'] = 'dB'
        res['scale_factor'] = 1.
        res['lons'] = f.lons
        res['lats'] = f.lats

        return res

    def resample(self, orig_lons, orig_lats, targ_lons, targ_lats, data, method='gauss',
                 radius_of_influence=500000, neighbours=50, sigmas=50000, nprocs=10):
        '''
        Resampling data
        '''

        xsize, ysize = targ_lons.shape

        orig_def = pyresample.geometry.SwathDefinition(lons=orig_lons.ravel(), lats=orig_lats.ravel())
        targ_def = pyresample.geometry.SwathDefinition(lons=targ_lons.ravel(), lats=targ_lats.ravel())

        if method == 'gauss':
            data_int = pyresample.kd_tree.resample_gauss(orig_def, data.ravel(), targ_def,
                                                         radius_of_influence=radius_of_influence,
                                                         neighbours=neighbours,
                                                         sigmas=sigmas, fill_value=None, nprocs=nprocs)
        if method == 'nearest':
            data_int = pyresample.kd_tree.resample_nearest(orig_def, data.ravel(), targ_def,
                                                           radius_of_influence=radius_of_influence,
                                                           epsilon=0.5, fill_value=None, nprocs=nprocs)

        return data_int.reshape(xsize, ysize)


class dataReader:
    '''
    Class for reading data in different formats
    '''

    def __init__(self):
        pass

    def read_nc(self, f):
        '''
        Get data from NetCDF file
        '''
        var_geo_time = ['time', 'lon', 'lat']
        res = {}
        res['data'] = {}
        nc = Dataset(f, 'r')

        # Get variable names with data
        data_vars = [var for var in nc.variables if not var in var_geo_time]

        for var in data_vars:
            res['data'][var] = nc[var][:].data
            res['units'] = ''
            res['scale_factor'] = 1.

        res['lons'] = nc['lon'][:].data
        res['lats'] = nc['lat'][:].data

        del nc

        return res

    def read_tiff(self, f):
        '''
        Get data and lats lons from GeoTiff file
        '''
        f = GeoTiff(f)
        f.getLatsLons()

        res = {}
        res['data'] = {}
        res['data']['s0'] = f.data
        res['units'] = 'dB'
        res['scale_factor'] = 1.
        res['lons'] = f.lons
        res['lats'] = f.lats

        return res


class ridgedIceClassifier(dataReader):
    '''
    Train and classify ridged/no ridged ice
    '''

    def __init__(self, glcm_filelist, ridges_filelist, flat_filelist, defo_filelist,
                 s0_filelist, defo_training=True):

        # Feature namelist
        self.glcm_names = ['Angular Second Moment',
                           'Contrast',
                           'Correlation',
                           'Sum of Squares: Variance',
                           'Inverse Difference Moment',
                           'Sum Average',
                           'Sum Variance',
                           'Sum Entropy',
                           'Entropy',
                           'Difference Variance',
                           'Difference Entropy',
                           'Information Measures of Correlation',
                           'Maximal Correlation Coefficient']

        self.glcm_filelist = glcm_filelist
        self.ridges_filelist = ridges_filelist
        self.flat_filelist = flat_filelist
        self.defo_filelist = defo_filelist
        self.s0_filelist = s0_filelist
        self.defo_training = defo_training

        self.glcm_unique_datetimes = self.get_unq_dt_from_files(self.glcm_filelist, 'GLCM')
        self.ridges_unique_datetimes = self.get_unq_dt_from_files(self.ridges_filelist, 'Ridged ice')
        self.flat_unique_datetimes = self.get_unq_dt_from_files(self.flat_filelist, 'Flat ice')
        self.defo_unique_datetimes = self.get_unq_dt_from_files(self.defo_filelist, 'Ice deformation')
        self.s0_unique_datetimes = self.get_unq_dt_from_files(self.s0_filelist, 'sigma zero')

        if glcm_filelist is not None:
            if defo_training:
                self.matched_datetimes = self.get_matched_dt([self.glcm_unique_datetimes, self.ridges_unique_datetimes,
                                                              self.flat_unique_datetimes, self.defo_unique_datetimes,
                                                              self.s0_unique_datetimes])
                self.collocate_data(self.glcm_filelist, self.ridges_filelist, self.flat_filelist, self.defo_filelist,
                                    self.s0_filelist)
            else:
                self.matched_datetimes = self.get_matched_dt([self.glcm_unique_datetimes, self.ridges_unique_datetimes,
                                                              self.flat_unique_datetimes,
                                                              self.s0_unique_datetimes])
                self.collocate_data(self.glcm_filelist, self.ridges_filelist, self.flat_filelist, self.s0_filelist)
        else:
            # Only s0 training
            self.matched_datetimes = self.get_matched_dt([self.ridges_unique_datetimes,
                                                          self.flat_unique_datetimes,
                                                          self.s0_unique_datetimes])

    @staticmethod
    def get_unq_dt_from_files(file_list, name):
        '''
        Get unique date and times form list of files
        '''

        dts_str = []
        dts = []
        for ifile in file_list:
            idt = re.findall(r'\d\d\d\d\d\d\d\d\w\d\d\d\d', ifile)[0]
            dts_str.append(idt)

        dts_str.sort()
        dts_str = set(dts_str)
        dts_str = list(dts_str)
        dts_str.sort()

        # Convert to Python date time format
        if dts_str:
            for idt_str in dts_str:
                idt_str = '%s-%s-%s %s:%s' % (idt_str[0:4], idt_str[4:6], idt_str[6:8], idt_str[9:11], idt_str[11:13])
                idt = datetime.strptime(idt_str, '%Y-%m-%d %H:%M')
                dts.append(idt)
            print(f'Unique dates for {name}: {dts}')
        else:
            print(f'No dates for {name}')

        return dts

    @staticmethod
    def get_matched_dt(file_lists):
        '''
        Get matched date and time for the files
        '''
        dt_matched = []

        for idt in file_lists[0]:
            flag = 1
            for il in file_lists[1:]:
                if not idt in il:
                    flag = 0
                else:
                    pass

            if flag == 1:
                dt_matched.append(idt)

            else:
                pass

        print(f'\nMatched dates for all file lists: {dt_matched}\n')

        return dt_matched

    def collocate_data(self, *args):
        '''
        Collocate all data from file lists
        '''

        # Flat ice GLCM features
        d_flat = {}

        # Ridged ice GLCM features
        d_ridged = {}

        # Form dictonary for results
        if self.glcm_filelist is not None:
            for ft_name in self.glcm_names:
                d_flat.setdefault(ft_name, [])
                d_ridged.setdefault(ft_name, [])

        if self.defo_training:
            # Add divergence and shear
            d_flat.setdefault('div', [])
            d_ridged.setdefault('div', [])

            d_flat.setdefault('shear', [])
            d_ridged.setdefault('shear', [])

        if self.s0_filelist is not None:
            d_flat.setdefault('s0', [])
            d_ridged.setdefault('s0', [])

        if self.matched_datetimes:
            for idt in self.matched_datetimes:
                # print(f'\nData collocation for {idt}\n')
                date_time = idt.strftime('%Y%m%dT%H%M')

                dt_files = []

                for x in args:
                    for ifile in x:
                        if date_time in ifile:
                            dt_files.append(ifile)

                print('\nDone.\n')

                print(f'Files for {idt}:\n {dt_files}\n Start collocation...')

                # Resample data onto one grid
                glcm_file = dt_files[0]
                ridge_file = dt_files[1]
                flat_file = dt_files[2]

                if self.defo_training:
                    defo_file = dt_files[3]
                    s0_file = dt_files[4]
                else:
                    s0_file = dt_files[3]

                # Interpolate sigma zero data
                print(f'\nInterpolating s0 data \n{s0_file} \nto \n{glcm_file}...\n')
                r = Resampler(s0_file, glcm_file)

                data_int_s0 = r.resample(r.f_source['lons'], r.f_source['lats'], r.f_target['lons'],
                                         r.f_target['lats'], r.f_source['data']['s0'],
                                         method='nearest', radius_of_influence=500000)
                print(f'Done\n')

                # Manual charts
                print(f'\nInterpolating manual Ridge data \n{ridge_file} \nto \n{glcm_file}...\n')
                r = Resampler(ridge_file, glcm_file)

                data_int_ridge = r.resample(r.f_source['lons'], r.f_source['lats'], r.f_target['lons'],
                                            r.f_target['lats'],
                                            r.f_source['data']['s0'], method='nearest', radius_of_influence=500000)
                print(f'Done\n')

                print(f'\nInterpolating manual Flat data \n{flat_file} \nto \n{glcm_file}...\n')
                r = Resampler(flat_file, glcm_file)
                data_int_flat = r.resample(r.f_source['lons'], r.f_source['lats'], r.f_target['lons'],
                                           r.f_target['lats'],
                                           r.f_source['data']['s0'], method='nearest', radius_of_influence=500000)
                print(f'Done\n')

                if flat_file == ridge_file:
                    print('\nWarning: Flat file is the same as ridge file!\nInverting flat data...')
                    data_int_flat[data_int_flat == 0] = -999
                    data_int_flat[data_int_flat > 0] = 0
                    data_int_flat[data_int_flat == -999] = 1
                    print('Done.\n')

                if self.defo_training:
                    #defo_file = dt_files[3]
                    r = Resampler(defo_file, glcm_file)

                    print(f'\nInterpolating Deformation data \n{defo_file} \nto \n{glcm_file}...\n')
                    r = Resampler(defo_file, glcm_file)
                    data_int_shear = r.resample(r.f_source['lons'], r.f_source['lats'], r.f_target['lons'],
                                                r.f_target['lats'],
                                                r.f_source['data']['ice_shear'],
                                                method='nearest', radius_of_influence=5000)

                    #r = Resampler(defo_file, glcm_file)
                    data_int_div = r.resample(r.f_source['lons'], r.f_source['lats'], r.f_target['lons'],
                                              r.f_target['lats'], r.f_source['data']['ice_divergence'],
                                              method='nearest', radius_of_influence=5000)
                    print(f'Done\n')
                else:
                    defo_int_shear = None
                    defo_int_div = None

                ############################################
                # Collect data over ridged and flat ice
                ############################################
                print('\nReading netcdf file...\n')
                data_glcm = self.read_nc(glcm_file)
                print('Done.\n')

                for ft_name in self.glcm_names:
                    ft = data_glcm['data'][ft_name][:]
                    # Collect data for flat ice
                    d_flat[ft_name].extend(ft[data_int_flat > 0].ravel())

                    # Collect data for ridged ice
                    d_ridged[ft_name].extend(ft[data_int_ridge > 0].ravel())

                if self.defo_training:
                    data_int_div[np.isnan(data_int_div)] = 0
                    data_int_shear[np.isnan(data_int_shear)] = 0

                    # Collect data for flat ice
                    d_flat['div'].extend(data_int_div[data_int_flat > 0].ravel())
                    d_flat['shear'].extend(data_int_shear[data_int_flat > 0].ravel())

                    # Collect data for ridged ice
                    d_ridged['div'].extend(data_int_div[data_int_ridge > 0].ravel())
                    d_ridged['shear'].extend(data_int_shear[data_int_ridge > 0].ravel())

                if self.s0_filelist is not None:
                    # Collect data for ridged ice
                    d_ridged['s0'].extend(data_int_s0[data_int_ridge > 0].ravel())
                    d_flat['s0'].extend(data_int_s0[data_int_flat > 0].ravel())

        print('\nConverting data to Pandas data frame...\n')
        d_df = {}

        if self.defo_training:
            ft_names = list(self.glcm_names)
            ft_names.extend(['div', 'shear', 's0'])
            for ivar in ft_names:
                d_df[ivar] = list(d_ridged[ivar]) + list(d_flat[ivar])
        else:
            ft_names = list(self.glcm_names)

            if self.s0_filelist is not None:
                ft_names.extend(['s0'])

            for ivar in ft_names:
                d_df[ivar] = list(d_ridged[ivar]) + list(d_flat[ivar])

        # Define class of ice for the each set of features
        # Classes of ice
        # 1 - ridged ice
        # 0 - flat ice
        d_df['ice_class'] = [1 for i in range(len(d_ridged[ivar]))] + [0 for i in range(len(d_flat[ivar]))]

        # The Pandas data frame contain all features for a certain grid cell
        data_train_pca_ridges = pd.DataFrame(d_df)
        data_train_pca_ridges[:]

        self.train_data = data_train_pca_ridges[:]
        print('Done.')

    def train_rf_classifier(self, lf):
        '''
        Train Random Forests classifier
        '''

        print('\nTrain Random-Forests classifier...\n')

        X = self.train_data[lf]  # Features
        y = self.train_data['ice_class']  # Labels

        # Split dataset into training set and test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)  # 70% training and XX% test

        # Create a Gaussian Classifier
        clf = RandomForestClassifier(n_estimators=1000)

        # Train the model using the training sets y_pred=clf.predict(X_test)
        self.classifier = clf.fit(X_train, y_train)

        print('Done.\n')

        y_pred = self.classifier.predict(X_test)

        # Model Accuracy, how often is the classifier correct?
        print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

    def classify_data(self, glcm_file_path, defo_file_path=None, s0_file_path=None, roi=None):
        '''
        Classify SAR image using Random Forest classifier
        '''

        glcm_file = self.read_nc(glcm_file_path)

        num_rows, num_columns = glcm_file['data'][list(glcm_file['data'].keys())[0]].shape

        classified_data = np.zeros((num_rows, num_columns))
        classified_data[:] = np.nan

        if roi:
            r_min, r_max, c_min, c_max = roi
        else:
            r_min, r_max = 0, num_rows
            c_min, c_max = 0, num_columns

        print(f'\nROI: {r_min} {r_max} {c_min} {c_max}\n')

        # r_min, r_max = 0, classified_data.shape[0]
        # c_min, c_max = 0, classified_data.shape[1]

        if defo_file_path:
            r = Resampler(defo_file_path, glcm_file_path)

            data_int_shear = r.resample(r.f_source['lons'], r.f_source['lats'], r.f_target['lons'],
                                        r.f_target['lats'],
                                        r.f_source['data']['ice_shear'],
                                        method='nearest',
                                        radius_of_influence=50000)

            data_int_div = r.resample(r.f_source['lons'], r.f_source['lats'], r.f_target['lons'],
                                      r.f_target['lats'], r.f_source['data']['ice_divergence'],
                                      method='nearest',
                                      radius_of_influence=50000)

            data_int_shear[np.isnan(data_int_shear)] = 0
            data_int_div[np.isnan(data_int_div)] = 0

        if s0_file_path:
            r = Resampler(s0_file_path, glcm_file_path)

            data_int_s0 = r.resample(r.f_source['lons'], r.f_source['lats'], r.f_target['lons'],
                                        r.f_target['lats'],
                                        r.f_source['data']['s0'],
                                        method='nearest',
                                        radius_of_influence=50000)
        # Classification
        start = time.time()
        for row in range(r_min, r_max):
            for column in range(c_min, c_max):
                sys.stdout.write('\rRow, col number: %s %s' % (row, column))
                test_sample = []

                for i_ft in self.glcm_names:
                    temp_data = glcm_file['data'][i_ft][row, column]

                    if not np.isnan(temp_data):
                        test_sample.append(temp_data)
                    else:
                        print('nan! Appending 0')
                        test_sample.append(0)

                if defo_file_path:
                    test_sample.append(data_int_div[row, column])
                    test_sample.append(data_int_shear[row, column])

                if s0_file_path:
                    test_sample.append(data_int_s0[row, column])

                y_pred = self.classifier.predict([test_sample])
                classified_data[row, column] = y_pred

        end = time.time()
        print('\nDone')
        print('\nClassified in %s minutes' % ((end - start) / 60.))
        plt.clf()
        plt.imshow(classified_data[r_min:r_max, c_min:c_max], interpolation='nearest', cmap='jet')
        self.classified_data = classified_data

######################################################################
# TODO: New class for classification based on multi-scale features
######################################################################
class deformedIceClassifier(dataReader):
    '''
    Train and classify SAR image into two classes: ridged and level ice
    '''

    def __init__(self, glcm_filelist, ridges_filelist, flat_filelist, defo_filelist,
                 s0_filelist, defo_training=False):

        self.glcm_filelist = glcm_filelist
        self.ridges_filelist = ridges_filelist
        self.flat_filelist = flat_filelist
        self.defo_filelist = defo_filelist
        self.s0_filelist = s0_filelist
        self.defo_training = defo_training

        # A dictonary with dates and list of files
        dates_and_files = {}

        self.glcm_unique_datetimes = self.get_unq_dt_from_files(self.glcm_filelist, 'GLCM')
        self.ridges_unique_datetimes = self.get_unq_dt_from_files(self.ridges_filelist, 'Ridged ice')
        self.flat_unique_datetimes = self.get_unq_dt_from_files(self.flat_filelist, 'Flat ice')

        if self.s0_filelist is not None and self.s0_filelist != []:
            self.s0_unique_datetimes = self.get_unq_dt_from_files(self.s0_filelist, 'Sigma nought')

        if self.defo_filelist is not None and self.defo_filelist != []:
            self.defo_unique_datetimes = self.get_unq_dt_from_files(self.defo_filelist, 'Ice deformation')

        if glcm_filelist is not None:
            if defo_training:
                self.matched_datetimes = self.get_matched_dt([self.glcm_unique_datetimes, self.ridges_unique_datetimes,
                                                              self.flat_unique_datetimes, self.defo_unique_datetimes])
                #self.collocate_data(self.glcm_filelist, self.ridges_filelist, self.flat_filelist, self.defo_filelist)
            else:
                self.matched_datetimes = self.get_matched_dt([self.glcm_unique_datetimes, self.ridges_unique_datetimes,
                                                              self.flat_unique_datetimes])

                #if self.s0_filelist is not None:
                #    self.collocate_data(self.glcm_filelist, self.ridges_filelist, self.flat_filelist, self.s0_filelist)
                #else:
                #    self.collocate_data(self.glcm_filelist, self.ridges_filelist, self.flat_filelist)
        else:
            # Only s0 training
            self.matched_datetimes = self.get_matched_dt([self.ridges_unique_datetimes,
                                                          self.flat_unique_datetimes,
                                                          self.s0_unique_datetimes])

        # Collect data for all matched date and times
        for idt in self.matched_datetimes:
            idate = idt.strftime('%Y%m%d')
            itime = idt.strftime('%H%M')
            idt_str = idt.strftime('%Y%m%dT%H%M%S')

            dates_and_files[idt_str] = {}
            dates_and_files[idt_str]['textures'] = [x for x in self.glcm_filelist if (idate in x and itime in x)][0]
            dates_and_files[idt_str]['level'] = [x for x in self.flat_filelist if (idate in x and itime in x)][0]
            dates_and_files[idt_str]['ridges'] = [x for x in self.ridges_filelist if (idate in x and itime in x)][0]

            if self.defo_filelist is not None and self.defo_filelist != []:
                dates_and_files[idt_str]['deformation'] = [x for x in self.defo_filelist
                                                           if (idate in x.split('_')[-1] and itime in x.split('_')[-1])][0]
            else:
                dates_and_files[idt_str]['deformation'] = ''

            if self.s0_filelist is not None and self.s0_filelist != []:
                dates_and_files[idt_str]['s0'] = [x for x in self.s0_filelist if (idate in x and itime in x)][0]
            else:
                dates_and_files[idt_str]['s0'] = ''

        self.dates_and_files = dates_and_files

        # Collocate data
        self.collocate_data()

    @staticmethod
    def get_unq_dt_from_files(file_list, name):
        '''
        Get unique date and times form list of files
        '''

        dts_str = []
        dts = []
        for ifile in file_list:
            if ifile.find('defo') > 0:
                idt = re.findall(r'\d\d\d\d\d\d\d\d\w\d\d\d\d', ifile)[1]
            else:
                idt = re.findall(r'\d\d\d\d\d\d\d\d\w\d\d\d\d', ifile)[0]
            dts_str.append(idt)

        dts_str.sort()
        dts_str = set(dts_str)
        dts_str = list(dts_str)
        dts_str.sort()

        # Convert to Python date time format
        if dts_str:
            for idt_str in dts_str:
                idt_str = '%s-%s-%s %s:%s' % (idt_str[0:4], idt_str[4:6], idt_str[6:8], idt_str[9:11], idt_str[11:13])
                idt = datetime.strptime(idt_str, '%Y-%m-%d %H:%M')
                dts.append(idt)
            print(f'Unique dates for {name}: {dts}')
        else:
            print(f'No dates for {name}')

        return dts

    @staticmethod
    def get_matched_dt(file_lists):
        '''
        Get matched date and time for the files
        '''
        dt_matched = []

        for idt in file_lists[0]:
            flag = 1
            for il in file_lists[1:]:
                if not idt in il:
                    flag = 0
                else:
                    pass

            if flag == 1:
                dt_matched.append(idt)

            else:
                pass

        print(f'\nMatched dates for all file lists: {dt_matched}\n')

        return dt_matched

    def collocate_data_matrixes(self):
        '''
        Collocate all data from file lists to matrix
        '''

        for idt in self.dates_and_files.keys():
            print(f'\nData collocation for {idt}\n')
            #date_time = idt.strftime('%Y%m%dT%H%M')

            # Resample data onto one grid
            glcm_file = self.dates_and_files[idt]['textures']
            ridge_file = self.dates_and_files[idt]['ridges']
            flat_file = self.dates_and_files[idt]['level']

            if self.defo_training:
                defo_file = self.dates_and_files[idt]['deformation']
            else:
                defo_file = ''

            # Manual charts
            print(f'\nInterpolating manual Ridge data \n{os.path.basename(ridge_file)} \nto '
                  f'\n{os.path.basename(glcm_file)}...\n')

            r = Resampler(ridge_file, glcm_file)

            data_int_ridge = r.resample(r.f_source['lons'], r.f_source['lats'], r.f_target['lons'],
                                        r.f_target['lats'],
                                        r.f_source['data']['s0'], method='nearest', radius_of_influence=500000)
            print(f'Done\n')

            print(f'\nInterpolating manual Flat data \n{os.path.basename(flat_file)} \nto '
                  f'\n{os.path.basename(glcm_file)}...\n')

            r = Resampler(flat_file, glcm_file)
            data_int_flat = r.resample(r.f_source['lons'], r.f_source['lats'], r.f_target['lons'],
                                       r.f_target['lats'],
                                       r.f_source['data']['s0'], method='nearest', radius_of_influence=500000)
            print(f'Done\n')

            if flat_file == ridge_file:
                print('\nWarning: Level ice file is the same as ridge file!\nInverting flat data...')
                data_int_flat[data_int_flat == 0] = -999
                data_int_flat[data_int_flat > 0] = 0
                data_int_flat[data_int_flat == -999] = 1
                print('Done.\n')
            else:
                pass

            if self.defo_training:
                #defo_file = dt_files[3]
                r = Resampler(defo_file, glcm_file)

                print(f'\nInterpolating Deformation data \n{defo_file} \nto \n{glcm_file}...\n')
                r = Resampler(defo_file, glcm_file)
                data_int_shear = r.resample(r.f_source['lons'], r.f_source['lats'], r.f_target['lons'],
                                            r.f_target['lats'],
                                            r.f_source['data']['ice_shear'],
                                            method='nearest', radius_of_influence=5000)

                data_int_div = r.resample(r.f_source['lons'], r.f_source['lats'], r.f_target['lons'],
                                          r.f_target['lats'], r.f_source['data']['ice_divergence'],
                                          method='nearest', radius_of_influence=5000)

                print(f'Done\n')
            else:
                defo_int_shear = None
                defo_int_div = None

            ######################################################
            # Collect training data
            ######################################################

            # Open data with texture features
            print('\nReading netcdf file with texture features...\n')
            data_glcm = self.read_nc(glcm_file)

            # Get feature names
            ft_names = [str(ft_name) for ft_name in range(len(data_glcm['data'].keys()))]
            #print(ft_names)
            print('Done.\n')

            # Initialize matrix with training labels
            training_labels = np.zeros(data_glcm['data'][ft_names[0]].shape, dtype=np.uint8)

            # Level ice
            training_labels[data_int_flat > 0] = 1

            # Ridges
            training_labels[data_int_ridge > 0] = 2

            # Train labels
            if not hasattr(self, 'training_labels'):
                self.training_labels = training_labels
            else:
                self.training_labels = np.concatenate((self.training_labels, training_labels), 0)

            # Train features
            if self.defo_training:
                z_dim = len(data_glcm['data'].keys()) + 2
            else:
                z_dim = len(data_glcm['data'].keys())

            print(z_dim, data_glcm['data'][ft_names[0]].shape[0], data_glcm['data'][ft_names[0]].shape[1])

            train_features = np.zeros((z_dim, data_glcm['data'][ft_names[0]].shape[0],
                                       data_glcm['data'][ft_names[0]].shape[1]), dtype=np.float)

            # Add texture features
            for iz in range(len(ft_names)):
                train_features[iz, :, :] = data_glcm['data'][ft_names[iz]]

            # Add deformation features (shear, divergence)
            # Add deformation data if defo training
            if self.defo_training:
                train_features[z_dim-2, :, :] = data_int_shear
                train_features[z_dim-1, :, :] = data_int_div
            else:
                pass

            # Replace nan values with 0
            train_features[np.isnan(train_features)] = 0.

            # Move axis
            train_features = np.moveaxis(train_features, 0, 2)

            if not hasattr(self, 'features'):
                self.features = train_features
            else:
                self.features = np.concatenate((self.features, train_features), 0)

    def collocate_data(self):
        '''
        Collocate all data from file lists into Pandas dataframe
        '''

        # Create dictonaries for all features
        d_df = {}

        idt = list(self.dates_and_files.keys())[0]
        glcm_file = self.dates_and_files[idt]['textures']
        data_glcm = self.read_nc(glcm_file)

        # Get texture feature names
        ft_names = [str(ft_name) for ft_name in range(len(data_glcm['data'].keys()))]

        for ft_name in ft_names:
            d_df.setdefault(ft_name, [])

        if self.defo_training:
            d_df.setdefault('div', [])
            d_df.setdefault('shear', [])

        d_df.setdefault('ice_class', [])
        df = pd.DataFrame(d_df)

        # Loop over all files for matched dates
        #  [list(self.dates_and_files.keys())[0]]:
        for idt in self.dates_and_files.keys():
            print(f'\nData collocation for {idt}\n')

            # Resample data onto common grid
            glcm_file = self.dates_and_files[idt]['textures']
            ridge_file = self.dates_and_files[idt]['ridges']
            flat_file = self.dates_and_files[idt]['level']

            if self.defo_training:
                defo_file = self.dates_and_files[idt]['deformation']
            else:
                defo_file = ''

            # Manual charts
            print(f'\nInterpolating manual Ridge data \n{os.path.basename(ridge_file)} \nto '
                  f'\n{os.path.basename(glcm_file)}...\n')

            r = Resampler(ridge_file, glcm_file)

            data_int_ridge = r.resample(r.f_source['lons'], r.f_source['lats'], r.f_target['lons'],
                                        r.f_target['lats'],
                                        r.f_source['data']['s0'], method='nearest', radius_of_influence=500000)
            print(f'Done\n')

            print(f'\nInterpolating manual Flat data \n{os.path.basename(flat_file)} \nto '
                  f'\n{os.path.basename(glcm_file)}...\n')

            r = Resampler(flat_file, glcm_file)
            data_int_flat = r.resample(r.f_source['lons'], r.f_source['lats'], r.f_target['lons'],
                                       r.f_target['lats'],
                                       r.f_source['data']['s0'], method='nearest', radius_of_influence=500000)
            print(f'Done\n')

            if self.defo_training:
                r = Resampler(defo_file, glcm_file)

                print(f'\nInterpolating Deformation data \n{defo_file} \nto \n{glcm_file}...\n')
                r = Resampler(defo_file, glcm_file)
                data_int_shear = r.resample(r.f_source['lons'], r.f_source['lats'], r.f_target['lons'],
                                            r.f_target['lats'],
                                            r.f_source['data']['ice_shear'],
                                            method='nearest', radius_of_influence=5000)

                data_int_div = r.resample(r.f_source['lons'], r.f_source['lats'], r.f_target['lons'],
                                          r.f_target['lats'], r.f_source['data']['ice_divergence'],
                                          method='nearest', radius_of_influence=5000)

                print(f'Done\n')
            else:
                defo_int_shear = None
                defo_int_div = None

            ######################################################
            # Collect training data
            ######################################################

            # Open data with texture features
            print('\nReading netcdf file with texture features...\n')
            data_glcm = self.read_nc(glcm_file)

            # Get texture feature names
            ft_names = [str(ft_name) for ft_name in range(len(data_glcm['data'].keys()))]
            print('Done.\n')

            d_flat = {}
            d_ridged = {}

            # Create dictonary keys with texture feature names
            for ft_name in ft_names:
                d_flat.setdefault(ft_name, [])
                d_ridged.setdefault(ft_name, [])

            for ft_name in ft_names:
                ft = data_glcm['data'][ft_name][:]
                d_flat[ft_name].extend(ft[data_int_flat > 0].ravel())
                d_ridged[ft_name].extend(ft[data_int_ridge > 0].ravel())

            # Create dictonary keys with texture deformation feature names
            if self.defo_training:
                # Add divergence and shear
                d_flat.setdefault('div', [])
                d_ridged.setdefault('div', [])
                d_flat.setdefault('shear', [])
                d_ridged.setdefault('shear', [])

                d_flat['div'].extend(data_int_div[data_int_flat > 0].ravel())
                d_flat['shear'].extend(data_int_shear[data_int_flat > 0].ravel())
                d_ridged['div'].extend(data_int_div[data_int_ridge > 0].ravel())
                d_ridged['shear'].extend(data_int_shear[data_int_ridge > 0].ravel())

            if self.defo_training:
                for ft_name in ft_names:
                    d_df[ft_name] = list(d_ridged[ft_name]) + list(d_flat[ft_name])

                d_df['div'] = list(d_ridged['div']) + list(d_flat['div'])
                d_df['shear'] = list(d_ridged['shear']) + list(d_flat['shear'])

                d_df['ice_class'] = [2 for i in range(len(d_ridged[ft_name]))] + \
                                    [1 for i in range(len(d_flat[ft_name]))]

            else:
                for ft_name in ft_names:
                    d_df[ft_name] = list(d_ridged[ft_name]) + list(d_flat[ft_name])

                    d_df['ice_class'] = [2 for i in range(len(d_ridged[ft_name]))] + \
                                        [1 for i in range(len(d_flat[ft_name]))]

            df_i = pd.DataFrame(d_df)
            df = pd.concat([df, df_i])

        # Drop rows with zeroes for all features
        self.features = df[np.count_nonzero(df.values[:, :-1], axis=1) > len(df.columns)-3]
        print('Data collocation and extracting have done.\n')

    def train_rf_classifier_matrix(self, bbox=None, n_estimators=50, n_jobs=10, max_depth=10, max_samples=0.05):
        '''
        Train Random Forests classifier from matrixes
        '''

        print('\nTrain Random-Forests classifier...\n')

        self.rf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=n_jobs,
                                         max_depth=max_depth, max_samples=max_samples)

        print('Train labels shape: (%s %s)' % self.training_labels.shape)
        print('Features shape: (%s %s %s)' % self.features.shape)

        if bbox is not None:
            self.classifier = future.fit_segmenter(self.training_labels[bbox[0]:bbox[1], bbox[2]:bbox[3]],
                                                   self.features[bbox[0]:bbox[1], bbox[2]:bbox[3], :],
                                                   self.rf)
        else:
            self.classifier = future.fit_segmenter(self.training_labels, self.features, self.rf)

        print('Done.\n')

    def train_rf_classifier(self, bbox=None, n_estimators=50, n_jobs=10, max_depth=10, max_samples=0.05):
        '''
        Train Random Forests classifier from Pandas dataframe
        '''

        print('\nTrain Random-Forests classifier...\n')

        self.rf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=n_jobs,
                                         max_depth=max_depth, max_samples=max_samples)

        X = self.features[list(self.features)[:-1]]  # Features
        y = self.features['ice_class']  # Labels

        # Split dataset into training set and test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)  # 70% training and XX% test

        # Create a Gaussian Classifier
        #clf = RandomForestClassifier(n_estimators=1000)

        # Train the model using the training sets y_pred=clf.predict(X_test)
        self.classifier = self.rf.fit(X_train, y_train)

        print('Done.\n')

        y_pred = self.classifier.predict(X_test)

        # Model Accuracy, how often is the classifier correct?
        print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

        #########################

    def detect_ice_state(self, input_features):
        '''
        Detect sea ice state from features
        '''

        print('\nDetecting ice state...\n')
        print('Input features shape: (%s %s %s)' % input_features.shape)
        print(self.classifier)

        result = future.predict_segmenter(input_features, self.classifier)
        self.result = result
        print('Done.\n')

        return result
