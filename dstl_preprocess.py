'''

'''

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os, sys
from osgeo import gdal
import tifffile as tiff
from shapely.wkt import loads as wkt_loads
import cv2



##########################################################################################
################################ DSTL preprocessing ######################################
##########################################################################################

def _convert_coordinates_to_raster(coords, img_size, xymax):
    '''
    This function supports generate_mask_for_image_and_class().
    Author: visoft
    https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask/code
    '''
    Xmax,Ymax = xymax
    H,W = img_size
    W1 = 1.0*W*W/(W+1)
    H1 = 1.0*H*H/(H+1)
    xf = W1/Xmax
    yf = H1/Ymax
    coords[:,1] *= yf
    coords[:,0] *= xf
    coords_int = np.round(coords).astype(np.int32)
    return coords_int


def _get_xmax_ymin(grid_sizes_panda, imageId):
    '''
    This function supports generate_mask_for_image_and_class().
    Author: visoft
    https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask/code
    '''
    xmax, ymin = grid_sizes_panda[grid_sizes_panda.ImageId == imageId].iloc[0,1:].astype(float)
    return (xmax,ymin)


def _get_polygon_list(wkt_list_pandas, imageId, cType):
    '''
    This function supports generate_mask_for_image_and_class().
    Author: visoft
    https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask/code
    '''
    df_image = wkt_list_pandas[wkt_list_pandas.ImageId == imageId]
    multipoly_def = df_image[df_image.ClassType == cType].MultipolygonWKT
    polygonList = None
    if len(multipoly_def) > 0:
        assert len(multipoly_def) == 1
        polygonList = wkt_loads(multipoly_def.values[0])
    return polygonList


def _get_and_convert_contours(polygonList, raster_img_size, xymax):
    '''
    This function supports generate_mask_for_image_and_class().
    Author: visoft
    https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask/code
    '''
    perim_list = []
    interior_list = []
    if polygonList is None:
        return None
    for k in range(len(polygonList)):
        poly = polygonList[k]
        perim = np.array(list(poly.exterior.coords))
        perim_c = _convert_coordinates_to_raster(perim, raster_img_size, xymax)
        perim_list.append(perim_c)
        for pi in poly.interiors:
            interior = np.array(list(pi.coords))
            interior_c = _convert_coordinates_to_raster(interior, raster_img_size, xymax)
            interior_list.append(interior_c)
    return perim_list,interior_list


def _plot_mask_from_contours(raster_img_size, contours, class_value = 1):
    '''
    This function supports generate_mask_for_image_and_class().
    Author: visoft
    https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask/code
    '''
    img_mask = np.zeros(raster_img_size,np.uint8)
    if contours is None:
        return img_mask
    perim_list,interior_list = contours
    cv2.fillPoly(img_mask,perim_list,class_value)
    cv2.fillPoly(img_mask,interior_list,0)
    return img_mask


def generate_mask_for_image_and_class(raster_size, imageId, class_type, grid_sizes_panda,
                                     wkt_list_pandas):
    '''
    This function should be called to generate a raster mask from train/test vectors on 
    a raster image.
    Author: visoft
    https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask/code
    '''
    xymax = _get_xmax_ymin(grid_sizes_panda,imageId)
    polygon_list = _get_polygon_list(wkt_list_pandas,imageId,class_type)
    contours = _get_and_convert_contours(polygon_list,raster_size,xymax)
    mask = _plot_mask_from_contours(raster_size,contours,1)
    return mask


def _get_image_names(base_path, imageId):
    '''
    Get the names of the tiff files
    Author: visoft
    '''
    d = {'3': path.join(base_path,'three_band/{}.tif'.format(imageId)),             # (3, 3348, 3403)
         'A': path.join(base_path,'sixteen_band/{}_A.tif'.format(imageId)),         # (8, 134, 137)
         'M': path.join(base_path,'sixteen_band/{}_M.tif'.format(imageId)),         # (8, 837, 851)
         'P': path.join(base_path,'sixteen_band/{}_P.tif'.format(imageId)),         # (3348, 3403)
         }
    return d


def _get_image_sizes(imageId):
    '''
    Get the dimensions of the tiff files in tuple format
    Author: NB
    '''
    d = {'3': (3, 3348, 3403),\
         'A': (8, 134, 137),\
         'M': (8, 837, 851),\
         'P': (3348, 3403),\
         }
    split_id = imageId.split('_')[-1].split('.')[0]
    if split_id == 'A' or split_id == 'M' or split_id == 'P':
        return d[split_id]
    else:
        return d['3']


def make_masks(inDir, imageId, verbose=False):
    '''
    Generates masks for one raster image over all 10 classes in dstl challenge. 
    Outputs a numpy array of size [im_size_rows, im_size_cols, # classes(10) ].
    Author: NB
    '''
    df = pd.read_csv(inDir + '/train_wkt_v4.csv')
    gs = pd.read_csv(inDir + '/grid_sizes.csv', names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)
    
    im_size = _get_image_sizes(imageId)
    
    masks = np.empty((im_size[1],im_size[2],10))
    for class_value in range(0,10):
        mask = generate_mask_for_image_and_class((im_size[1],im_size[2]),imageId,class_value,gs,df)
        masks[:,:,class_value] = mask
    if verbose == True: print type(masks), masks, masks.shape
    return masks




def stretch():
    '''
    https://www.kaggle.com/mehmetbercan/dstl-satellite-imagery-feature-detection/display-a-polygon-class-on-16-band-m/run/759322/notebook
    
    '''
    pass
    




def pansharpen():
    '''
    https://www.kaggle.com/resolut/dstl-satellite-imagery-feature-detection/panchromatic-sharpening/run/961801/notebook
    
    '''
    pass


##########################################################################################
############################## Plotting/Visualization ####################################
##########################################################################################

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



if __name__ == '__main__':
    inDir = '/Volumes/PORTABLE/SYNC/GIS_Projects/DSTL_challenge/data'
    raster_path = inDir + '/three_band/6010_0_0.tif'
    read_plot_3band_tif(inDir, raster_path)

