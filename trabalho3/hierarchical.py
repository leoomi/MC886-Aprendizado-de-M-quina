import matplotlib.pyplot as plt
import numpy as np
import sys
from numpy import genfromtxt
import scipy
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.metrics import pairwise_distances
import scipy.cluster.hierarchy as hcluster

data = genfromtxt('documents/data.csv', delimiter=',')

kf = KFold(n_splits=2)
kf.get_n_splits(data)

for train_index, val_index in kf.split(data):
    #for i in np.arange(0.01, 1, 0.01):
    for i in np.flip(np.arange(0.1, 0.4, 0.1), axis=0):
        #try:
        data_train, data_val = data[train_index], data[val_index]
        clusters = hcluster.fclusterdata(data_train, i, criterion="distance")
    
        sil_score = metrics.silhouette_score(data_train, clusters, metric='euclidean')
        ch_score = metrics.calinski_harabaz_score(data_train, clusters)
    
        print("Threshold %f -- Silhoute Score: %f, CH Score: %f " % (i, sil_score, ch_score))
        #except:
            #print("Unexpected error:", sys.exc_info()[0])
            #break


