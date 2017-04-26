# UQAM_INF7370_MachineLearning

Repository used for a machine learning class project.

The goal was to experiment with pixel-wise classification of buildings vs non-buildings in aerial imagery (30 cm resolution) from Open Street Map (OSM) training and test data.

The script can take one training and one test geotiff image. Then use respective 'building' and 'non-building' training/test polygons (shp) to create numpy arrays that can be used by scikit-learn machine learning classifiers kNN, random forest, svm and multilayer perceptron. It is also possible to stretch histograms (not working), create a 3D scatter of the training data and aggregate the pixels of an image after it was predicted by a classifier to get better results.

Some train/test polygons are mislabeled. We still get an accuracy around 70%, which is alright for simple pixel-wise classification with RGB bands only.

The data we used can be dowloaded at: https://www.dropbox.com/s/uklsson5kjv8zjb/data.zip?dl=0


![hello](https://www.dropbox.com/s/0dzdhyb8jl7kump/mlp_16_8_4_aggregate_150dpi.png)
