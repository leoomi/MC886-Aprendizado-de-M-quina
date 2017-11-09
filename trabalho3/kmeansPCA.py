
#import
print("Importing libraries")
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import numpy as np
print("done")

# read data
print("Reading data")
data_path = "documents/data.csv"
data_file = open(data_path)
data = data_file.readlines()
data = [list(map(str.strip, i.split(","))) for i in data]
data = [list(map(float, i)) for i in data]
data = np.array(data)
print("done")

print("Starting PCA")
pca = PCA(n_components=1000, whiten=True)
pca.fit(data)
data = pca.transform(data)

print(data.shape)
print("PCA finished")

# compute kmeans
print("Computing k-means")
kmeans = KMeans(n_clusters=151, random_state=0, verbose=False, max_iter=10, n_init=3).fit(data)
print(kmeans.inertia_)

print("done");
