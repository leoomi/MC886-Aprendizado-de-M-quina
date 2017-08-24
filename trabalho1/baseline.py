import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
import scipy
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

print("Loading...")
music_data = genfromtxt('year-prediction-msd-train.txt', delimiter=',')
print("Training data loaded")

# Split year values from the array
music_y = music_data[:, 0]

# Split the first timbre average as feature
music_x = music_data[:, np.newaxis, 2]
print (music_x)

# Split the data using train_test_split (training data and validation, considering we already have a testing set)
music_x_train, music_x_val, music_y_train, music_y_val =  train_test_split(music_x, music_y, test_size=0.05, random_state=0)

print(music_x_train.shape, music_x_val.shape, music_y_train.shape, music_y_val.shape)

plt.scatter(music_x_train, music_y_train,  color='blue')
plt.show()

# Create linear regression object
regr = linear_model.LinearRegression()
# Train the model using the training sets
regr.fit(music_x_train, music_y_train)
# Make predictions using the testing set
music_y_pred = regr.predict(music_x_val)

# The coefficients
print('Estimated intercept: ', regr.intercept_)

# The coefficients
print('Coefficients: ', regr.coef_)

# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(music_y_val, music_y_pred))

plt.scatter(music_x_train, music_y_train,  color='pink')
plt.scatter(music_x_val, music_y_val,  color='blue')
plt.plot(music_x_val, music_y_pred, color='black', linewidth=3)

plt.show()
