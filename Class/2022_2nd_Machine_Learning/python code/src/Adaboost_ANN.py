import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.wrappers.scikit_learn import KerasRegressor
from keras.optimizers import SGD
from sklearn.model_selection import cross_val_score
import keras.utils
from sklearn import metrics


cancer_data = load_breast_cancer()
input_train, input_test, target_train, target_test = train_test_split(cancer_data['data'], cancer_data['target'], test_size=0.2, random_state=42)

model_ann = Sequential()
model_ann.add(Dense(512, input_dim=input_train.shape[1], kernel_initializer='normal', activation='relu'))
# model_ann.add(Dropout)      # 과적합 방지
model_ann.compile(optimizer='adam',
                  # loss='sparse_categorical_crossentropy',
                  loss='mean_squared_error',
                  metrics=['accuracy'])
model_ann.fit(input_train, target_train, epochs=10, verbose=2)

model_ABC = AdaBoostClassifier(n_estimators=1, learning_rate=1)
model_ABC_100 = AdaBoostClassifier(n_estimators=10, learning_rate=1)
model_ABC_200 = AdaBoostClassifier(n_estimators=20, learning_rate=1)
model_ABC_400 = AdaBoostClassifier(n_estimators=40, learning_rate=1)


model_ABC.fit(input_train, target_train)
model_ABC_100.fit(input_train, target_train)
model_ABC_200.fit(input_train, target_train)
model_ABC_400.fit(input_train, target_train)

pred_ABC     = model_ABC.predict(input_test)
pred_ABC_100 = model_ABC_100.predict(input_test)
pred_ABC_200 = model_ABC_200.predict(input_test)
pred_ABC_400 = model_ABC_400.predict(input_test)

score = cross_val_score(model_ABC, input_test, target_test, cv=5)
print("<score>")
print(score.mean())
# print(score.std())

class_ABC = model_ABC.get_params()
print(class_ABC)

# target과 predict 값을 연산비교하여 일치하지 않는 개수가 얼마나 되는지 확인
fail_pred_ABC     = np.logical_xor(target_test, pred_ABC    )
fail_pred_ABC_100 = np.logical_xor(target_test, pred_ABC_100)
fail_pred_ABC_200 = np.logical_xor(target_test, pred_ABC_200)
fail_pred_ABC_400 = np.logical_xor(target_test, pred_ABC_400)
# 일치하지 않는 비율 구하기
rate_of_fail_pred_ABC = fail_pred_ABC.sum() / len(fail_pred_ABC)
rate_of_fail_pred_ABC_100 = fail_pred_ABC_100.sum() / len(fail_pred_ABC_100) * 100
rate_of_fail_pred_ABC_200 = fail_pred_ABC_200.sum() / len(fail_pred_ABC_200) * 100
rate_of_fail_pred_ABC_400 = fail_pred_ABC_400.sum() / len(fail_pred_ABC_400) * 100

print("------------------------")
print("ㅣ Ada boost ANN Result ㅣ")
print("------------------------")
print("<<<rate of fail matches>>>")
print("target = ", target_test)
print("number of estimater")
print("         1 : ", rate_of_fail_pred_ABC)
print("        10 : ", rate_of_fail_pred_ABC_100)
print("        20 : ", rate_of_fail_pred_ABC_200)
print("        40 : ", rate_of_fail_pred_ABC_400)

print("<<<Accuracy>>>")
print("         1  : ", metrics.accuracy_score(target_test, pred_ABC))
print("        10  : ", metrics.accuracy_score(target_test, pred_ABC_100))
print("        20  : ", metrics.accuracy_score(target_test, pred_ABC_200))
print("        40  : ", metrics.accuracy_score(target_test, pred_ABC_400))

print('process finished')