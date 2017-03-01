'''
Traditional classification of dstl satellite imagery using random forest and svm machine learning algorithms.

Export of pixel-wise mask from:
https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask/code

Classification from:
https://github.com/machinalis/satimg
'''

import numpy as np
import os

from osgeo import gdal
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import cv2
from shapely.wkt import loads as wkt_loads
import tifffile as tiff




def _get_image_names(base_path, imageId):
    '''
    Get the names of the tiff files
    '''
    d = {'3': path.join(base_path,'three_band/{}.tif'.format(imageId)),             # (3, 3348, 3403)
         'A': path.join(base_path,'sixteen_band/{}_A.tif'.format(imageId)),         # (8, 134, 137)
         'M': path.join(base_path,'sixteen_band/{}_M.tif'.format(imageId)),         # (8, 837, 851)
         'P': path.join(base_path,'sixteen_band/{}_P.tif'.format(imageId)),         # (3348, 3403)
         }
    return d




def _convert_coordinates_to_raster(coords, img_size, xymax):
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
    xmax, ymin = grid_sizes_panda[grid_sizes_panda.ImageId == imageId].iloc[0,1:].astype(float)
    return (xmax,ymin)


def _get_polygon_list(wkt_list_pandas, imageId, cType):
    df_image = wkt_list_pandas[wkt_list_pandas.ImageId == imageId]
    multipoly_def = df_image[df_image.ClassType == cType].MultipolygonWKT
    polygonList = None
    if len(multipoly_def) > 0:
        assert len(multipoly_def) == 1
        polygonList = wkt_loads(multipoly_def.values[0])
    return polygonList


def _get_and_convert_contours(polygonList, raster_img_size, xymax):
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
    img_mask = np.zeros(raster_img_size,np.uint8)
    if contours is None:
        return img_mask
    perim_list,interior_list = contours
    cv2.fillPoly(img_mask,perim_list,class_value)
    cv2.fillPoly(img_mask,interior_list,0)
    return img_mask


def generate_mask_for_image_and_class(raster_size, imageId, class_type, grid_sizes_panda,
                                     wkt_list_pandas):
    xymax = _get_xmax_ymin(grid_sizes_panda,imageId)
    polygon_list = _get_polygon_list(wkt_list_pandas,imageId,class_type)
    contours = _get_and_convert_contours(polygon_list,raster_size,xymax)
    mask = _plot_mask_from_contours(raster_size,contours,1)
    return mask


inDir = "/Volumes/PORTABLE/SYNC/GIS_Projects/DSTL_challenge/data"

df = pd.read_csv(inDir + '/train_wkt_v4.csv')

gs = pd.read_csv(inDir + '/grid_sizes.csv', names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)

masks = np.empty((3348,3403,10))
for class_value in range(0,10):
    mask = generate_mask_for_image_and_class((3348, 3403),"6120_2_2",class_value,gs,df)
    masks[:,:,class_value] = mask
    #masks.append(mask)
    #cv2.imwrite("/Volumes/PORTABLE/SYNC/GIS_Projects/DSTL_challenge/output_classification/mask1.png",mask*255)
print type(mask), mask
print masks.shape

############# Warning: some training class polygons overlap other classes. ###############


# next:
# Choose one image. For each class, extract pixel values of each band.
# Make a new array where each pixel is a row and columns are bands.
# Make a new labels array, relating each pixel in previous step to a label class.
# Find a way to relate pixel back to its position in original array

path = inDir + '/three_band/'+'6120_2_2.tif'
a = tiff.imread(path)

arr = np.swapaxes(np.swapaxes(np.array(a),0,2),0,1)
print arr.shape

vals = np.empty((0,3))
labels = np.empty(0)
for i in range(masks.shape[2]):
    ind_rows, ind_cols = np.where(masks[:,:,i]==1)
    labels = np.concatenate((labels,np.ones(ind_rows.shape[0])*i+1),axis=0)
    vals = np.concatenate((vals,arr[ind_rows,ind_cols,:]),axis=0)
    print i
print vals, vals.shape
print labels, labels.shape    



'''
RandomForestClassifier().fit(X,Y)

where 
X = [[ b1, b2, b3]
       b1, b2, b3]
       b1, b2, b3]
     ]
Y = [id1, id2, id3]  #labels ex. ['roof', 'road','water']
'''

#classifier = RandomForestClassifier(n_jobs=-1, n_estimators=10)
#classifier.fit(vals, labels)




