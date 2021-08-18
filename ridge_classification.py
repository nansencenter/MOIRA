class SarImage:
    '''
    Basic S1 L1 GRD image processing
    '''

    def __init__(self, zipPath):
        self.name = zipPath
        # self.bbox = bbox
        self.data = {}

        # s0 range in dB
        self.db_min = -35
        self.db_max = -5

        # Grayscale range
        self.gl_min = 1
        self.gl_max = 255

        self.meta = {}
        self.ds = {}

        # os.makedirs(os.path.dirname(outFname), exist_ok=True)

        try:
            pass
            '''
            print('Opening %s ...' % os.path.basename(zipPath))
            s1 = Sentinel1Image(zipPath)

            if bbox:
                s0 = s1[self.polBand][bbox[0]:bbox[1], bbox[2]:bbox[3]]
            else:
                s0 = s1[self.polBand]
            #nesz = s1['sigma0_%s' % pol]    
            self.data[self.polBand] = 10*np.log10(s0)
            self.data[self.polBand][np.isinf(self.data[self.polBand])] = np.nan
            #self.lons, self.lats = s1.get_geolocation_grids()

            print('Done')
            '''
        except:
            print('\nError while opening %s. Plese check the filepath and poalrization.' % zipPath)

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

    def radiometric_calibration(self, input_tiff_path, calibration_xml_path):
        '''
        Apply calibration values to DN
        '''

        measurement_file = gdal.Open(input_tiff_path)
        measurement_file_array = np.array(measurement_file.GetRasterBand(1).ReadAsArray().astype(np.float32))

        radiometric_coefficients_array = self.get_coefficients_array(calibration_xml_path, 'calibrationVectorList',
                                                                     'sigmaNought', measurement_file.RasterXSize,
                                                                     measurement_file.RasterYSize)
        print('Radiometric calibration...')
        tiff_name = os.path.basename(input_tiff_path)
        self.data[tiff_name] = (measurement_file_array * measurement_file_array) / (
                    radiometric_coefficients_array * radiometric_coefficients_array)

        # save_array_as_geotiff_gcp_mode(calibrated_array, output_tiff_path, measurement_file)

    def calibrate_project(self, t_srs, res, mask=False, write_file=True, out_path='.'):
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

        for tiffPath in tiffPaths:
            print('Prcessing %s ...' % os.path.basename(tiffPath))
            # Find calibration and noise annotation files
            calib_f = glob.glob(
                os.path.dirname(tiffPaths[0]).replace('measurement', 'annotation/calibration/*calibration*%s*.xml' %
                                                      os.path.basename(tiffPaths[0]).split('.')[0]))[0]
            noise_f = \
            glob.glob(os.path.dirname(tiffPaths[0]).replace('measurement', 'annotation/calibration/*noise*%s*.xml' %
                                                            os.path.basename(tiffPaths[0]).split('.')[0]))[0]

            if calib_f:
                print('\nStart calibration...')
                self.radiometric_calibration(tiffPath, calib_f)
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
                    ds_warp2 = gdal.Warp('%s/%s' % (out_path, os.path.basename(tiffPath)), tmp_ds, format="GTiff",
                                         dstSRS="EPSG:%s" % t_srs,
                                         xRes=res, yRes=res, multithread=True, callback=clb)

                    self.data['s0_%s' % pol] = {}
                    self.data['s0_%s' % pol]['data'] = ds_warp2.ReadAsArray()
                    self.data['s0_%s' % pol]['units'] = 'dB'
                    self.data['s0_%s' % pol]['scale_factor'] = 1.

                    ds_fname = 'ds_s0_%s' % var_name
                    self.ds[ds_fname] = ds_warp2

                    self.meta['ds_fname'] = ds_fname
                else:
                    ds_warp2 = gdal.Warp('', tmp_ds, format="MEM", dstSRS="EPSG:%s" % t_srs,
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

    def exportNetCDF4(self, out_path='.'):
        '''
        Make NetCDF4 file

        '''

        # Check if lat lon parameters are exist
        if hasattr(self, 'lats'):
            pass
        else:
            print('\nGetting lats lons...\n')
            self.getLatsLons()

        out_fname = '%s/%s.nc' % (out_path, os.path.basename(self.name).split('.')[0])
        try:
            os.remove(out_fname)
        except OSError:
            pass

        ds = Dataset(out_fname, 'w', format='NETCDF4_CLASSIC')

        # Dimensions
        y_dim = ds.createDimension('y', self.lons.shape[0])
        x_dim = ds.createDimension('x', self.lons.shape[1])
        time_dim = ds.createDimension('time', None)
        # data_dim = ds.createDimension('data', len([k for k in data.keys()]))

        # Variables
        times = ds.createVariable('time', np.float64, ('time',))
        latitudes = ds.createVariable('lat', np.float32, ('y', 'x',))
        longitudes = ds.createVariable('lon', np.float32, ('y', 'x',))

        for var_name in self.data.keys():
            globals()[var_name] = ds.createVariable(var_name, np.float32, ('y', 'x',))
            globals()[var_name][:, :] = self.data[var_name]['data']
            globals()[var_name].units = self.data[var_name]['units']
            globals()[var_name].scale_factor = self.data[var_name]['scale_factor']

        # Global Attributes
        ds.description = ''  # % self.data.keys()
        ds.history = 'Created ' + time.ctime(time.time())
        ds.source = 'NERSC'

        # Variable Attributes
        latitudes.units = 'degree_north'
        longitudes.units = 'degree_east'
        times.units = 'hours since 0001-01-01 00:00:00'
        times.calendar = 'gregorian'

        # Put variables
        latitudes[:, :] = self.lats
        longitudes[:, :] = self.lons

        ds.close()
        print('\n%s has been successefully created.\n' % out_fname)
        self.nc_file = out_fname

    def normalizeInt(self, r):
        '''
        Rescale data into defined range of integer values
        '''
        # r - range of values

        self.norm_data = {}

        for iband in self.data.keys():
            arr = self.data[iband]['data']
            arr[np.isnan(arr)] = 0
            self.norm_data['norm_%s' % iband] = (r * (arr - np.min(arr)) / np.ptp(arr)).astype(int)

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
            globals()[var_name][:, :] = data[var_name]['data']
            globals()[var_name].units = data[var_name]['units']
            globals()[var_name].scale_factor = data[var_name]['scale_factor']

        # Global Attributes
        ds.description = ''  # % self.data.keys()
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

    def __init__(self, zipPath, ws, stp, threads):
        super().__init__(zipPath)
        # self.zipPath = zipPath

        self.ws = ws
        self.stp = stp
        self.threads = threads

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

    def calcTexFt(self):
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
            print('Done.\n')

        self.glcm_features['lats'] = self.lats[::self.stp, ::self.stp]
        self.glcm_features['lons'] = self.lons[::self.stp, ::self.stp]

        # self.smooth()

        # Check if data are normilized
        if hasattr(self, 'norm_data'):
            pass
        else:
            print('\nData normalization...\n')
            self.normalizeInt(255)

        start = time.time()
        print('\nStart GLCM computation...')

        for iband in self.norm_data.keys():
            print(iband)
            self.glcm_features[iband] = {}
            texFts = self.getTextureFeatures(self.norm_data[iband])

            # Adding GLCM data
            for i in range(len(texFts[:, 0, 0])):
                self.glcm_features[iband][self.names_glcm[i + 1]] = {'data': texFts[i, :, :], 'scale_factor': 1.,
                                                                     'units': ''}

        end = time.time()

        print('\nDone in %s minutes\n' % ((end - start) / 60.))

    def export_netcdf(self, out_fname):
        '''
        Export results to NetCDF file
        '''

        ks = [x for x in t.glcm_features.keys() if 's0' in x]
        for ivar in ks:
            out_fname_nc = '%s/%s_%s' % (os.path.dirname(out_fname), ivar, os.path.basename(out_fname))
            super().export_netcdf(self.glcm_features['lats'], self.glcm_features['lons'],
                                  self.glcm_features[ivar], out_fname_nc)
