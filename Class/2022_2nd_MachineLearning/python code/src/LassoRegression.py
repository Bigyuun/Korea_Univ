import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso

# Preparing Data
perch_length = np.array(
    [13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0,
     21.0, 21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5,
     22.5, 22.7, 23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5,
     27.3, 27.5, 27.5, 27.5, 28.0, 28.7]
     )
perch_weight = np.array(
    [32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0,
     110.0, 115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0,
     130.0, 150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0,
     197.0, 218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0]
     )

# Initializing w(weight), b(bias), learning rate(lamda), iteration
w = np.random.rand(1)
b = np.random.rand(1)
print('init w : ', w)
print('init b : ', b)
lamda = 1e-3
iteration = 5000

# data arange & dividing train and test data set
N = np.size(perch_weight)
train_input, test_input, train_target, test_target = train_test_split(perch_length, perch_weight, random_state=42)
train_input = train_input.reshape(-1,1)
train_target = train_target.reshape(-1,1)
test_input = test_input.reshape(-1,1)
test_target = test_target.reshape(-1,1)

''' Main Algorithm'''
''' ---- START ---'''
# Using iteration method
for i in tqdm(range(0,iteration)) :
    for j in range(0, len(train_input)):
        output = np.dot(np.asarray([w,b]).T, np.asarray([train_input[j], 1]))

        dedw = train_input[j]
        dLde = output - train_target[j]
        dedb = 1
        dLdw = dLde * dedw
        dLdb = dLde * dedb

        # w = w - lamda * dLdw
        if w >= 0:
            w = w - lamda*dLdw - lamda*0.1*w
            b = b - lamda * dLdb
        else:
            w = w - lamda*dLdw + lamda*0.1*w
            b = b - lamda * dLdb

# Using sklearn Class
lr = Lasso()
lr.fit(train_input, train_target)

''' Main Algorithm'''
''' ----- END ----'''

# Analysis
print('Weight(w) : iteration) ', w[0], ' sklearn) ', lr.coef_[0])
print('Bais(b)   : iteration) ', b[0], ' sklearn) ', lr.intercept_)
print('Score (train data set) : ', lr.score(train_input, train_target))
print('Score (test data set) : ', lr.score(test_input, test_target))

# Drawing Plot Graph
plt.subplot(1,2,1)
plt.scatter(perch_length, perch_weight, marker='o')
plt.plot(perch_length, w*perch_length+b, '-r')
plt.title('Regression using Gradient Descent Method')

plt.subplot(1,2,2)
plt.scatter(perch_length, perch_weight)
plt.plot(train_input, lr.coef_ * train_input+lr.intercept_)
plt.title('Regression using sklearn LinearRegression Class')

plt.show()







