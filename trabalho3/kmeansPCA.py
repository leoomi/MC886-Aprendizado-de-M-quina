
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

print("Starting PCA")
pca = PCA(n_components=1000, whiten=True)
pca.fit(data)
data = pca.transform(data)

print(data.shape)
print("PCA finished")

clusterer = KMeans(n_clusters=151, random_state=10)
cluster_labels = clusterer.fit_predict(X)
silhouette_avg = silhouette_score(X, cluster_labels)
print("Score: ", silhouette_avg)
