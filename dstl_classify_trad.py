'''
Traditional classification of dstl satellite imagery using random forest and svm machine learning algorithms.

Classification from:
https://github.com/machinalis/satimg
'''

import numpy as np
import os
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import pandas as pd
import tifffile as tiff

import dstl_preprocess


inDir = "/Volumes/PORTABLE/SYNC/GIS_Projects/DSTL_challenge/data"
imageId = '6120_2_2'

masks = dstl_preprocess.make_masks(inDir, imageId)


path = inDir + '/three_band/'+imageId+'.tif'
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




