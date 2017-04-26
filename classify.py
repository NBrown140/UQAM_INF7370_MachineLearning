'''
A simple script to perform SVM, random forest, kNN and multilayer perceptron algorithms on a training
and test dataset. Training: one geotiff + training shapefiles. Test: one geotiff + test shapefiles.

The script was used in the context of a machine learning class at UQAM in 2017.

Histogram stretching doesn't seem to work.

Modified from: https://github.com/machinalis/satimg
'''

import numpy as np
import os, sys, time
from osgeo import gdal
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib

################################################## Functions ###########################################################

def create_mask_from_vector(vector_data_path, cols, rows, geo_transform,
                            projection, target_value=1):
    """Rasterize the given vector (wrapper for gdal.RasterizeLayer)."""
    data_source = gdal.OpenEx(vector_data_path, gdal.OF_VECTOR)
    layer = data_source.GetLayer(0)
    driver = gdal.GetDriverByName('MEM')  # In memory dataset
    target_ds = driver.Create('', cols, rows, 1, gdal.GDT_UInt16)
    target_ds.SetGeoTransform(geo_transform)
    target_ds.SetProjection(projection)
    gdal.RasterizeLayer(target_ds, [1], layer, burn_values=[target_value])
    return target_ds

def vectors_to_raster(file_paths, rows, cols, geo_transform, projection):
    """Rasterize all the vectors in the given directory into a single image."""
    labeled_pixels = np.zeros((rows, cols))
    for i, path in enumerate(file_paths):
        label = i+1
        ds = create_mask_from_vector(path, cols, rows, geo_transform,
                                     projection, target_value=label)
        band = ds.GetRasterBand(1)
        labeled_pixels += band.ReadAsArray()
        ds = None
    return labeled_pixels

def write_geotiff(fname, data, geo_transform, projection):
    """Create a GeoTIFF file with the given data."""
    driver = gdal.GetDriverByName('GTiff')
    rows, cols = data.shape
    dataset = driver.Create(fname, cols, rows, 1, gdal.GDT_Byte)
    dataset.SetGeoTransform(geo_transform)
    dataset.SetProjection(projection)
    band = dataset.GetRasterBand(1)
    band.WriteArray(data)
    dataset = None  # Close the file

def read_geotiff(geotiff_path):
    """Read a geotiff with gdal library. Outputs:
        goe_transform, proj, rows, cols, n_bands, bands_data"""
    raster_dataset = gdal.Open(geotiff_path, gdal.GA_ReadOnly)
    geo_transform = raster_dataset.GetGeoTransform()
    proj = raster_dataset.GetProjectionRef()
    bands_data = []
    for b in range(1, raster_dataset.RasterCount+1):
        band = raster_dataset.GetRasterBand(b)
        bands_data.append(band.ReadAsArray())
    bands_data = np.dstack(bands_data)
    rows, cols, n_bands = bands_data.shape
    return geo_transform, proj, [rows, cols, n_bands], bands_data

def histeq(im, nbr_bins=256):
  """ Histogram Equalization  
  https://www.safaribooksonline.com/library/view/programming-computer-vision/9781449341916/ch01.html """
  out = np.zeros_like(im)
  n = im.shape[2]
  for i in range(n):
      im1 = im[:,:,i]
      # get image histogram
      imhist,bins = np.histogram(im1.flatten(),nbr_bins,normed=True)
      cdf = imhist.cumsum() # cumulative distribution function
      cdf = 255 * cdf / cdf[-1] # normalize
      # use linear interpolation of cdf to find new pixel values
      im2 = np.interp(im1.flatten(),bins[:-1],cdf)
      out[:,:,i] = im2.reshape(im1.shape)
  return out

def plot_rgb_histograms(data, save_path, stretched=False):
    Red = data[:,:,0].flatten(); Green = data[:,:,1].flatten(); Blue = data[:,:,2].flatten()
    fig = plt.figure()
    plt.hist(Red, bins=100, normed=1, facecolor='red', alpha=0.50)
    plt.hist(Green, bins=100, normed=1, facecolor='green', alpha=0.50)
    plt.hist(Blue, bins=100, normed=1, facecolor='blue', alpha=0.50)
    plt.xlabel('Intensity')
    plt.ylabel('Frenquency')
    plt.title('RGB Histograms, stretched = {}'.format(str(stretched)))
    #plt.axis([40, 160, 0, 0.03])
    plt.grid(True)
    #plt.show()
    fig.savefig(save_path, dpi=200)


def get_spaced_colors(N):
    ''' Taken from kquinn at http://stackoverflow.com/questions/876853/generating-color-ranges-in-python'''
    import colorsys
    HSV_tuples = [(x * 1.0 / N, 0.5, 0.5) for x in range(N)]
    RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
    return RGB_tuples

def scatter3D(labels, samples, save_name, every_ith):
    import matplotlib.patches as mpatches
    from mpl_toolkits.mplot3d import Axes3D
    print 'Plotting 3d scatter...'
    X=samples[::every_ith,0]; Y=samples[::every_ith,1]; Z=samples[::every_ith,2]
    labels = labels[::every_ith]
    unique_labels = np.unique(labels)
    colors = get_spaced_colors(unique_labels.shape[0])
    color_dict ={}
    for i in range(0,unique_labels.shape[0]):
        color_dict[np.unique(labels)[i]] = colors[i]
    color_labels = []
    for i in labels:    #switch labels to colors
        color_labels.append(color_dict[i])
    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X, Y, Z, c=color_labels, marker='o', edgecolors='none', s=10)
    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')
    # Add legend
    recs = []
    for i in range(0,len(colors)):
        recs.append(mpatches.Rectangle((0,0),1,1,fc=colors[i]))
    plt.legend(recs,unique_labels,loc=4)
    fig.savefig(save_name)

def Most_Common(lst):
    from collections import Counter
    data = Counter(lst)
    return data.most_common(1)[0][0]

def aggregate_pixels(image, n):
    ''' Input a 2D image. Replace pixels which are surrounded by less than n (usually 2) pixels
     of the same value as themselves in a 3x3 window by the most common pixel in surrounding'''
    print 'Aggregating pixels...'
    [rows, cols] = image.shape
    out = np.copy(image)
    for i in range(1,rows-1):
        #print str(float(i)/rows*100)+' %'
        for j in range(1,cols-1):
            value = image[i,j]
            window = [image[i-1,j],image[i+1,j],image[i,j-1],image[i,j+1],image[i-1,j-1],image[i-1,j+1],image[i+1,j-1],image[i+1,j+1]]
            if window.count(value)<=n:
                out[i,j] = Most_Common(window)
            else:
                out[i,j] = value
    return out

############################################### Main Script ############################################################
CLASSIFIERS = {
    # http://scikit-learn.org/dev/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    'random-forest': RandomForestClassifier(n_jobs=-1, n_estimators=10, class_weight='balanced'),
    # http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    'svm': SVC(kernel='rbf', class_weight='balanced', max_iter=100),
    # http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
    'mlp': MLPClassifier(solver='adam', activation='relu', alpha=1e-5, hidden_layer_sizes = (16,8,4)),
    # http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
    'knn': KNeighborsClassifier(n_neighbors=10, weights='distance', algorithm='auto', n_jobs=-1)}

# INPUT
aggregate = True
stretch = False
cwd = 'D:\\UQAM_INF7370\\Examples\\satimg-master\\nick_data'
outname = 'mlp_16_8_4_trainS2_testS3_noStretch_aggregate'
train_raster_path = os.path.join(cwd,'buildings_scene2.tif')
train_vector_path = os.path.join(cwd,'shp_scene2')
test_raster_path = os.path.join(cwd,'buildings_scene3.tif')
test_vector_path = os.path.join(cwd,'shp_scene3')
method = 'mlp'
plot_scatter = False

output_histogram_path = os.path.join(cwd,'output','{}.png'.format(outname))
output_evaluationFile_path = os.path.join(cwd,'output','{}.txt'.format(outname))
output_classifier_path = os.path.join(cwd,'output','{}.pkl'.format(outname))
output_tif_path = os.path.join(cwd,'output','{}.tif'.format(outname))



### Read training tiff
geo_transform, proj, [rows,cols,n_bands], bands_data = read_geotiff(train_raster_path)

# STRETCH HISTOGRAMS
if stretch == True:
    print 'Stretching histograms...'
    bands_data = histeq(bands_data)
plot_rgb_histograms(bands_data, output_histogram_path, stretched=stretch)

# TRAINING
### Convert training vector classes to raster mask
files = [f for f in os.listdir(train_vector_path) if f.endswith('.shp')]
classes = [f.split('.')[0] for f in files]
shapefiles = [os.path.join(train_vector_path, f)for f in files if f.endswith('.shp')]
print 'Using training files: '+str(shapefiles)
labeled_pixels = vectors_to_raster(shapefiles, rows, cols, geo_transform, proj)  # shape:(rows,cols). With 1,2,...,n for n classes.
### Preprocess for classifier input
is_train = np.nonzero(labeled_pixels)    # find index of labeled pixels
training_labels = labeled_pixels[is_train]  # extract labels
training_samples = bands_data[is_train] # extract samples
if plot_scatter==True: scatter3D(training_labels, training_samples, os.path.join(cwd,'output','scatter3D.pdf'), 1000)
### Train classifier
print "Training classifier... Started at: {}".format(time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.localtime())); start_time = time.time()
classifier = CLASSIFIERS[method]
classifier.fit(training_samples, training_labels)
joblib.dump(classifier, output_classifier_path)
print 'Done training classifier. Training took {} seconds for {} samples\n'.format(str(time.time()-start_time), str(training_labels.shape[0]))

# PREDICTING
### Read test tiff
geo_transform, proj, [rows,cols,n_bands], bands_data = read_geotiff(test_raster_path)
### Predict with classifier
print "Predicting..."
n_samples = rows*cols
flat_pixels = bands_data.reshape((n_samples, n_bands))
result = classifier.predict(flat_pixels)
classification = result.reshape((rows, cols))
if aggregate == True:
    classification = aggregate_pixels(classification,2)
# write classification to geotiff
print "Writing output to geotiff..."
write_geotiff(output_tif_path, classification, geo_transform, proj)


# EVALUATE PREDICTION
shapefiles = [os.path.join(test_vector_path, "%s.shp"%c) for c in classes]
print 'Using test files: '+str(shapefiles)
test_pixels = vectors_to_raster(shapefiles, rows, cols, geo_transform, proj)
is_test = np.nonzero(test_pixels)
test_labels = test_pixels[is_test]
predicted_labels = classification[is_test]
print 'Saving test evaluation to file: {}'.format(output_evaluationFile_path)
target_names = ['Class %s' % s for s in classes]
print target_names
f= open(output_evaluationFile_path,"w+")
f.write("Confussion matrix:\n{}\n\n".format(metrics.confusion_matrix(test_labels, predicted_labels)))
f.write("Classification report:\n{}\n\n".format(metrics.classification_report(test_labels, predicted_labels, target_names=target_names)))
f.write("Classification accuracy: {}".format(str(metrics.accuracy_score(test_labels, predicted_labels))))
f.close()

print("Confussion matrix:\n%s" %
      metrics.confusion_matrix(test_labels, predicted_labels))
target_names = ['Class %s' % s for s in classes]
print("Classification report:\n%s" %
      metrics.classification_report(test_labels, predicted_labels, target_names=target_names))
print("Classification accuracy: %f" %
      metrics.accuracy_score(test_labels, predicted_labels))
