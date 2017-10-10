
# coding: utf-8

# In[1]:

import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
import scipy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from skimage import io
from skimage.color import rgb2gray
from skimage.filters import roberts, sobel, scharr, prewitt
from sklearn.decomposition import PCA
train_set_dir = 'cifar-10/train/'
test_set_dir = 'cifar-10/test/'

img_labels = genfromtxt(train_set_dir + 'labels')
img_labels, img_labels.shape


# In[3]:

img_data_original = np.empty([50000, 1024])

for i in range(0, 50000):
    filename = '{0:05d}'.format(i) + '.png'
    img = rgb2gray(io.imread(train_set_dir + filename)) #Reading file, converting to Grayscale
    img_data_original[i, :] = img.flatten()

for i in range(10,50000,100):
    pca = PCA(n_components=i, whiten=True)
    pca.fit(img_data)
    img_data = pca.transform(img_data_original)

    # Split the data using K-Folds, using 5 different sets
    kf = KFold(n_splits=5)
    kf.get_n_splits(img_data)

    count = 0
    train_score = np.zeros(5)
    val_score = np.zeros(5)
    for train_index, val_index in kf.split(img_data):
        img_data_train, img_data_val = img_data[train_index], img_data[val_index]
        img_labels_train, img_labels_val = img_labels[train_index], img_labels[val_index]
    
        regr = LogisticRegression(multi_class='ovr')
        regr.fit(img_data_train, img_labels_train)

        count += 1
        train_score[count-1] = regr.score(img_data_train, img_labels_train)
        val_score[count-1] = regr.score(img_data_val, img_labels_val)
        print("Set %d -- Train Score: %.2f Validation score: %.2f"
              % (count, train_score[count-1], val_score[count-1]))

        print("PCA: %.2f Mean Score Train: %.2f Mean Score Validation: %.2f" % (i, np.average(train_score), np.average(val_score)))




