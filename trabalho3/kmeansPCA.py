
print("Importing libraries")
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


print("Reading data")
data_path = "documents/data.csv"
data_file = open(data_path)
data = data_file.readlines()[:10000]
data = [list(map(str.strip, i.split(","))) for i in data]
data = [list(map(float, i)) for i in data]
X = np.array(data)
print("done")

for i in [1000, 500, 250, 100, 10]:
    print("Starting PCA")
    pca = PCA(n_components=i, whiten=True)
    pca.fit(X)
    pcaX = pca.transform(X)
    
    print(pcaX.shape)
    print("PCA finished")
    
    clusterer = KMeans(n_clusters=151, random_state=10)
    cluster_labels = clusterer.fit_predict(pcaX)
    silhouette_avg = silhouette_score(pcaX, cluster_labels)
    print(i)
    print("Score: ", silhouette_avg)
