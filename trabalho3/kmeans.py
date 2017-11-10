
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

# read ids
print("Reading ids")
ids_path = "documents/ids"
ids_file = open(ids_path)
ids = ids_file.readlines()
ids = [i.strip() for i in ids]
print("done")

# ## compute kmeans in a range (compute graph points for using elbow method)
# print("Computing k-means")
# for i in range(10, 1001, 10):
#     kmeans = KMeans(n_clusters=i, random_state=0, verbose=False, max_iter=10, n_init=3).fit(data)
#     print(i, kmeans.inertia_)
# print("done");

kmeans = KMeans(n_clusters=151, random_state=0, verbose=False, max_iter=300, n_init=3).fit(data)
buck = []
cluster = 17
for i in range(len(data)):
    t = kmeans.predict([data[i]])
    if t == cluster:
        buck.append((kmeans.transform([data[i]])[0][cluster], i))
    print(i)

print(buck)
