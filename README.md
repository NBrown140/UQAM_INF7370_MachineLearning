# UQAM_INF7370_MachineLearning

Repository used for a machine learning class project.

The script can take one training and one test geotiff image. Then use respective 'building' and 'non-building' training/test polygons (shp) to create numpy arrays that can be used by scikit-learn machine learning classifiers kNN, random forest, svm and multilayer perceptron. It is alos possible to streth histograms (not working), create a 3D scatter of the training data and aggregate the pixels of an image after it wa predicted by a classifier to get better results.

Some train/test polygons are mislabeled. We still get an acurcy around 70%, which is alright for simple pixel-wise classification with RGB bands only.

The data we used can be dowloaded at: https://www.dropbox.com/s/uklsson5kjv8zjb/data.zip?dl=0
