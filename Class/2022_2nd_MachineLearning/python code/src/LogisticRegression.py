import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# Global macro
G_USE_DATA_FROM_CSV = 0

# user Functions
def sigmoid(x):
    return 1/(1+np.exp(-x))

def loss_function(predict, y):  # cross-entropy method
    loss = -y * np.log(predict) - (1-y)*np.log(1-predict)
    return loss

"""
data set from .csv
"""
feature = ('FG%','FT%','3PM','PTS','TREB','AST','STL','BLK','TO')
if G_USE_DATA_FROM_CSV:
    data = pd.read_csv('../docs/NBA Rookie DRAFT Rank.csv')
    print(data.head())

    # Split data to input & target
    data_input = data[['FG%','FT%','3PM','PTS','TREB','AST','STL','BLK','TO']].to_numpy()
    data_target = data[['Entry of draft']].to_numpy()
    train_input, test_input, train_target, test_target = train_test_split(data_input, data_target, test_size=0.4, random_state=42)
    print(".csv is used.\n")
else:
    input = np.array([[0.461, 0.722, 0.9, 21., 6.5, 3.6, 0.7, 0.6, 2.9],
                  [0.365, 0.859, 2.1, 11., 6.9, 0.8, 0.4, 1.0, 1.1],
                  [0.413, 0.734, 1.5, 15., 4.9, 4.3, 1.2, 0.3, 2.7],
                  [0.425, 0.805, 2.1, 18., 3.9, 1.6, 0.7, 0.1, 1.9],
                  [0.460, 1.000, 1.4, 7.8, 2.6, 3.7, 0.9, 0.2, 1.2],
                  [0.414, 0.852, 1.9, 11., 4.1, 0.9, 0.8, 0.6, 1.6],
                  [0.755, 0.565, 0.0, 5.8, 5.3, 0.6, 0.2, 1.7, 0.7],
                  [0.416, 0.828, 0.7, 8.4, 5.4, 0.9, 1.3, 0.4, 0.8],
                  [0.529, 0.799, 0.7, 10., 3.1, 2.6, 0.6, 0.3, 1.6],
                  [0.493, 1.000, 1.5, 9.6, 2.0, 0.7, 1.1, 0.1, 0.7],
                  [1.000, 1.000, 0.0, 4.0, 3.0, 1.0, 0.0, 1.0, 0.0],
                  [0.473, 0.550, 0.4, 8.1, 4.1, 2.0, 1.0, 0.5, 1.6],
                  [0.493, 0.645, 0.6, 5.3, 3.9, 2.2, 0.7, 0.3, 0.7],
                  [0.592, 0.467, 0.0, 6.5, 6.7, 0.6, 0.6, 0.9, 1.0],
                  [0.413, 1.000, 1.2, 5.1, 3.1, 1.0, 0.6, 0.2, 0.9],
                  [0.396, 0.947, 0.5, 6.5, 2.6, 1.0, 0.6, 0.1, 0.8],
                  [0.636, 0.875, 0.0, 4.2, 3.8, 0.2, 0.4, 0.2, 0.6],
                  [0.454, 0.646, 0.0, 4.1, 3.7, 0.4, 0.2, 1.2, 0.6],
                  [0.434, 0.885, 0.5, 4.0, 1.3, 1.0, 0.3, 0.4, 0.4],
                  [0.468, 0.610, 0.9, 8.0, 2.4, 0.5, 0.3, 0.2, 0.7],
                  [0.370, 0.792, 1.0, 4.3, 2.5, 0.8, 0.3, 0.1, 0.3],
                  [0.391, 0.687, 1.0, 6.1, 2.6, 0.6, 0.3, 0.2, 0.7],
                  [0.385, 0.701, 0.8, 5.3, 2.9, 0.4, 0.5, 0.1, 1.2],
                  [0.400, 0.000, 0.5, 2.5, 1.5, 0.0, 0.8, 0.0, 0.3],
                  [0.372, 0.258, 0.5, 3.5, 2.7, 1.5, 0.3, 0.3, 0.4],
                  [0.667, 0.500, 1.0, 8.0, 1.0, 2.0, 0.0, 0.0, 1.0],
                  [0.750, 0.000, 0.0, 1.5, 0.3, 0.3, 0.8, 0.0, 0.0],
                  [0.571, 1.000, 0.0, 3.7, 2.7, 0.0, 0.0, 0.0, 0.7],
                  [0.391, 0.750, 0.7, 2.8, 2.2, 0.5, 0.3, 0.0, 0.2],
                  [0.328, 0.802, 0.7, 5.3, 1.6, 1.8, 0.2, 0.2, 1.0],
                  [0.406, 0.532, 0.3, 2.9, 2.0, 0.7, 0.4, 0.1, 0.3],
                  [0.375, 0.000, 0.2, 1.4, 0.6, 0.4, 0.4, 0.2, 0.0],
                  [0.438, 0.660, 0.3, 3.0, 1.0, 0.5, 0.2, 0.2, 0.7],
                  [0.385, 0.674, 0.1, 1.9, 2.2, 0.7, 0.1, 0.2, 0.3],
                  [0.296, 0.432, 0.1, 2.5, 1.3, 2.0, 0.5, 0.3, 0.9],
                  [0.455, 0.667, 0.0, 2.8, 2.0, 0.6, 0.2, 0.0, 0.4],
                  [0.500, 0.000, 0.1, 1.3, 1.6, 0.0, 0.1, 0.0, 0.0],
                  [0.500, 0.000, 0.1, 1.0, 0.6, 0.0, 0.1, 0.0, 0.0],
                  [0.273, 0.750, 0.1, 1.0, 0.2, 0.5, 0.1, 0.2, 0.4],
                  [0.475, 0.568, 0.1, 3.1, 1.3, 0.1, 0.1, 0.0, 0.0],
                  [0.332, 1.000, 0.4, 2.1, 1.0, 0.4, 0.0, 0.0, 1.7],
                  [0.275, 1.000, 0.0, 1.1, 0.6, 0.3, 0.0, 0.1, 0.1],
                  [0.248, 0.000, 0.0, 0.7, 0.3, 0.3, 0.3, 0.0, 0.3],
                  [0.333, 0.000, 0.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0],
                  [0.200, 0.000, 0.3, 1.3, 0.8, 0.3, 0.0, 0.0, 0.5],
                  [0.000, 0.000, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                  [0.000, 0.500, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                  ])
    target = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]).T.reshape(-1,1)
    train_input, test_input, train_target, test_target = train_test_split(input, target, test_size=0.5, random_state=42)
    print("manual data is used\n")

# Using iteration method
learning_rate = 1e-3
iteration = 1000
W = np.random.rand(train_input.shape[1],1)  # shape[0] : rows, shape[1] : columns
b = np.random.rand(1,1)
print('init w : ', W.T)
print('init b : ', b)

for i in tqdm(range(0,iteration)):
    output = np.dot(train_input,W)
    predict = sigmoid(output)

    Loss = loss_function(predict,train_target)
    Loss = np.sum(Loss)

    # get dL/dW & dL/db
    dLdsig = -1 * ( train_target/predict - (1-train_target)/(1-predict) )    #dLoss()/dsigmoid()
    dsigdy = predict * (1-predict)                                           #dsigmoid()/dy
    dLdW = np.dot(train_input.T, dLdsig*dsigdy)
    dLdb = np.sum(dLdsig * dsigdy)

    # Update W, b
    W = W - learning_rate * dLdW
    b = b - learning_rate * dLdb

# Using sklearn Class
lr = LogisticRegression(C=1, max_iter=iteration)
lr.fit(train_input, train_target)
pred_lr = lr.predict(test_input)
pred_proba = lr.predict_proba(test_input)
precision = precision_score(test_target, pred_lr)
conf_matrix = confusion_matrix(test_target, pred_lr)
class_report = classification_report(test_target, pred_lr)


print(feature)
print('Weight(w) - iteration : ', W.T)
print("          - sklearn   : ", lr.coef_[0])
print('Bais(b)   - iteration : ', b[0][0])
print('          - sklearn   : ', lr.intercept_[0])
print('Score (train data set) : ', lr.score(train_input, train_target))
print('Score (test data set) : ', lr.score(test_input, test_target))
#
# print('Weight(w) : iteration) ', W.T)
# print('              sklearn) ', lr.coef_)
# print('Bais(b)   : iteration) ', b[0])
# print('              sklearn) ', lr.intercept_)
# print('Score (train data set) : ', lr.score(train_input, train_target))
# print('Score (test data set) : ', lr.score(test_input, test_target))
print('learning finish')

xn = np.arange(0, 17, 0.01)
yn = np.exp(-xn-b[0])/(1+np.exp(-xn-b[0]))

# plt.plot(xn,yn)

y = np.dot(input, lr.coef_.T)
# print(y)
columns, rows = input.shape
k= input[:,0]
c = np.arange(0,columns,1)

plt.figure("NBA Rookie Draft Ranking")
plt.subplot(1,4,1)
plt.title("Data set for each feature")
for i in range(0,rows):
    # plt.scatter(c, input[:,i], label=feature[i])
    plt.plot(c, input[:, i], label=feature[i])
plt.legend()

plt.subplot(1,4,2)
plt.title("Data set for each feature (Detail)")
plt.ylim(0, 2)
for i in range(0,rows):
    # plt.scatter(c, input[:,i], label=feature[i])
    plt.plot(c, input[:, i], label=feature[i])
plt.legend()

plt.subplot(1,4,3)
plt.title("Logic Values")
plt.scatter(y, target, marker="o")
plt.xlabel('entropy')
plt.ylabel('Sigmoid')

plt.subplot(1,4,4)
plt.title("Logisitc Regression")
plt.scatter(y, target, marker="o")
plt.scatter(y, 1/(1+np.exp(-y-lr.intercept_)), marker= 's')
plt.plot(xn, 1/(1+np.exp(-xn-lr.intercept_)), '-r')
# plt.plot(xn,yn)

# sample_data = np.array([[0.3, 0.7, 0.8, 2.7, 1.2, 0.3, 0.4, 0.1, 0.8]])
sample_data = np.array([[0.460, 1.000, 1.4, 7.8, 2.6, 3.7, 0.9, 0.2, 1.2]])

predict_sample = np.dot(sample_data, W.ravel())
# print( np.exp(predict_sample+b[0])/(1+np.exp(predict_sample+b[0])))
# print(1/(1+np.exp(-predict_sample-b[0])))
# print(lr.predict(sample_data))



plt.show()
print("process finished")





