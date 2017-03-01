__author__ = 'nb14'

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os, sys
from osgeo import gdal



def read_plot_3band_tif(inDir, raster_path):

    raster_dataset = gdal.Open(raster_path, gdal.GA_ReadOnly)
    geo_transform = raster_dataset.GetGeoTransform()
    proj = raster_dataset.GetProjectionRef()

    bands_data = []
    for b in range(1,raster_dataset.RasterCount+1):
        band = raster_dataset.GetRasterBand(b)
        bands_data.append(band.ReadAsArray())
    bands_data = np.dstack(bands_data).astype(np.float32)
    rows, cols, n_bands = bands_data.shape

    print bands_data[0].dtype
    print 'rows: %s, cols: %s, n_bands: %s' %(rows,cols,n_bands)
    for i in range(0, n_bands):
        print 'Band '+str(i)+':  min= %s, max= %s' %(bands_data[:,:,i].min(), bands_data[:,:,i].max())

    # Bands normalization to 0-1
    bands_data_norm = bands_data/np.amax(bands_data.reshape(rows*cols,n_bands),axis=0)
    for i in range(0, n_bands):
        print 'Normalized band '+str(i)+':  min= %s, max= %s' %(bands_data_norm[:,:,i].min(), bands_data_norm[:,:,i].max())

    ### Plot
    cmap = plt.get_cmap('viridis')
    x,y = range(0,cols),range(0,rows)
    z1, z2, z3 = bands_data[:,:,0], bands_data[:,:,1], bands_data[:,:,2]

    fig, (ax0,ax1,ax2) = plt.subplots(nrows=1, ncols=3)
    im0 = ax0.pcolormesh(x, y, z1, cmap=cmap)
    ax0.set_title('Band 1')
    im1 = ax1.pcolormesh(x, y, z2, cmap=cmap)
    ax1.set_title('Band 2')
    im2 = ax2.pcolormesh(x, y, z3, cmap=cmap)
    ax2.set_title('Band 3')
    fig.colorbar(im0, ax=ax0)
    fig.colorbar(im1, ax=ax1)
    fig.colorbar(im2, ax=ax2)
    fig.show()

    fig1, ax0 = plt.subplots()
    ax0.imshow(bands_data_norm)
    ax0.set_title('Color Composite')
    fig1.show()





def






if __name__ == '__main__':
    inDir = '/Volumes/PORTABLE/SYNC/GIS_Projects/DSTL_challenge/data'
    raster_path = inDir + '/three_band/6010_0_0.tif'
    read_plot_3band_tif(inDir, raster_path)

