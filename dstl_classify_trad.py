'''
Traditional classification of dstl satellite imagery using random forest, svm and
neural network machine learning algorithms.

Classification inspired from:
https://github.com/machinalis/satimg
'''
from osgeo import gdal
import numpy as np
import os, time, sys
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import pandas as pd
import tifffile as tiff

import dstl_preprocess


def write_geotiff(fname, data):
    '''
    Create a GeoTIFF file with the given data.
    Author: machinalis
    https://github.com/machinalis/satimg
    '''
    driver = gdal.GetDriverByName('GTiff')
    rows, cols = data.shape
    dataset = driver.Create(fname, cols, rows, 1, gdal.GDT_Byte)
    #dataset.SetGeoTransform(geo_transform)
    #dataset.SetProjection(projection)
    band = dataset.GetRasterBand(1)
    band.WriteArray(data)
    dataset = None  # Close the file




inDir = "/Volumes/PORTABLE/SYNC/GIS_Projects/DSTL_challenge/data"

################################### Training #############################################

imageId = '6120_2_2'
path = inDir + '/three_band/'+imageId+'.tif'


im_train = tiff.imread(path)

print 'Creating raster masks from vector training polygons...'
masks = dstl_preprocess.make_masks(inDir, imageId)

print 'Applyig masks to image and reshaping numpy array for ingestion in classifier...'
arr = np.swapaxes(np.swapaxes(np.array(im_train),0,2),0,1)
vals = np.empty((0,3))
labels = np.empty(0)
for i in range(masks.shape[2]):
    ind_rows, ind_cols = np.where(masks[:,:,i]==1)
    labels = np.concatenate((labels,np.ones(ind_rows.shape[0])*i+1),axis=0)
    vals = np.concatenate((vals,arr[ind_rows,ind_cols,:]),axis=0)
    print 'Class '+str(i)
print '\nTraining samples:\n',vals, vals.shape
print 'Training labels:\n',labels, labels.shape, '\n'    

'''
RandomForestClassifier().fit(X,Y)

where 
X = [[ b1, b2, b3]
       b1, b2, b3]
       b1, b2, b3]
     ]
Y = [id1, id2, id3]  #labels ex. ['roof', 'road','water']
'''
print 'Training classifier...'; start_time = time.time()
classifier = RandomForestClassifier(n_jobs=-1, n_estimators=10)
classifier.fit(vals, labels)
print 'Done training classifier. Training took {} seconds for {} samples/pixels\n'.format(str(time.time()-start_time), str(labels.shape[0]))

##################################### Testing ############################################

image_id_test = '6010_0_0'
path = inDir + '/three_band/'+imageId+'.tif'
out_tif = '/Volumes/PORTABLE/SYNC/GIS_Projects/DSTL_challenge/output_classification/test1.tif'

im_test = tiff.imread(path)

n_bands, rows, cols = dstl_preprocess._get_image_sizes(imageId)
if n_bands!=im_test.shape[0] or rows!=im_test.shape[1] or cols!=im_test.shape[2]:
    print 'Error: Image dimensions do not match that predicted by dstl_preprocess._get_image_sizes()'
n_samples = rows*cols
flat_pixels = im_test.reshape((n_samples, n_bands))

print 'Predicting classes for image {}'.format(imageId); start_time = time.time()
result = classifier.predict(flat_pixels)
print 'Done predicting classes for image {}. Took {} seconds for {} samples/pixels'.format(imageId, str(time.time()-start_time), str(n_samples))

classification = result.reshape((rows, cols))

print 'Writing classified output to tif'
write_geotiff(out_tif, classification)


# shapefiles = [os.path.join(validation_data_path, "%s.shp"%c) for c in classes]
# verification_pixels = vectors_to_raster(shapefiles, rows, cols, geo_transform, proj)
# for_verification = np.nonzero(verification_pixels)
# verification_labels = verification_pixels[for_verification]
# predicted_labels = classification[for_verification]
# 
# print("Confussion matrix:\n%s" %
#       metrics.confusion_matrix(verification_labels, predicted_labels))
# target_names = ['Class %s' % s for s in classes]
# print("Classification report:\n%s" %
#       metrics.classification_report(verification_labels, predicted_labels,
#                                     target_names=target_names))
# print("Classification accuracy: %f" %
#       metrics.accuracy_score(verification_labels, predicted_labels))
# 



