
#import
print("Importing libraries")
from sklearn.cluster import KMeans
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

# compute kmeans
print("Computing k-means")
for i in range(10, 1001, 10):
    kmeans = KMeans(n_clusters=i, random_state=0, verbose=False, max_iter=10, n_init=3).fit(data)
    print(i, kmeans.inertia_)
print("done");
