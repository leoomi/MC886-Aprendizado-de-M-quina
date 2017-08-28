import numpy as np

def linearRegression (data_x, data_y, learning_rate, precision):
    lr = learning_rate
    n = data_x.shape[1]
    m = data_x.shape[0]

    intercept = 0
    coeff = np.zeros(n)
    iterations = 0
    
    while True:
        stop = True
        temp = np.zeros(n)

        error = (np.sum(data_x * coeff, axis=1) + intercept - data_y)
        temp_i = intercept - lr * (1/m) * np.sum(error)
        
        for i in range(n):
            temp[i] = coeff[i] - lr * (1/m) * np.sum(error * data_x[:,i])
            print(error*data_x[:,i])
            
        diff = temp - coeff
        diff [diff > precision] = True
        if temp_i - intercept > precision or diff.shape[0] < n:
            stop = False
            
        intercept = temp_i
        coeff = temp
        iterations += 1
        
        #print(intercept, coeff)
        if stop:
            break
        
    return intercept, coeff, iterations

data_x = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])
data_y = np.array([1, 2, 3, 4])

print(linearRegression(data_x, data_y, 1.0e-2, 1.0e-7))
