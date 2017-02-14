This repository includes the support material for the following blog posts:
 
# Python for geospatial data processing

http://www.machinalis.com/blog/python-for-geospatial-data-processing/

To try this code you need to install the packages described in the [requirements.txt](https://raw.githubusercontent.com/machinalis/satimg/master/requirements.txt) file 
and you can use [this sample data](https://drive.google.com/open?id=0B64odlXwDnHeUVBWNXVocU84SkU)

#### Build GDAL from source

This code was developed in Ubuntu (Linux) using the GDAL library, version 2.0.1. You need to install it in your machine before installing the Python bindings. It is all explained in the following links:

* Download: http://download.osgeo.org/gdal/2.0.1/gdal-2.0.1.tar.gz
* Install: https://trac.osgeo.org/gdal/wiki/GdalOgrInPython
* Python bindings: https://trac.osgeo.org/gdal/wiki/BuildingOnUnix
* PyPi: https://pypi.python.org/pypi/GDAL/


# Python for Object Based Image Analysis (OBIA) 

http://www.machinalis.com/blog/obia/

This post is a continuation of the first one. The development of the subject is in a 
[Jupyter Notebook](https://github.com/machinalis/satimg/blob/master/object_based_image_analysis.ipynb)

To try it out, you can use [the same data as in the other post](https://drive.google.com/open?id=0B64odlXwDnHeUVBWNXVocU84SkU).


# Searching for aliens 

http://www.machinalis.com/blog/searching-for-aliens/

The development of the subject is in a [Jupyter Notebook](https://github.com/machinalis/satimg/blob/master/Searching%20for%20aliens.ipynb)

It has a couple of specific requirements:

* affine==2.0.0.post1
* OpenCV (which I installed following this: https://gist.github.com/Asymptote/6d95396a1a45b55e3b63c3ee4d2b7c24)
