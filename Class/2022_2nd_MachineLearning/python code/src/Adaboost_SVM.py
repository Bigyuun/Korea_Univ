import numpy as np
import matplotlib as plt
import pandas as pd

from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import metrics

cancer_data = load_breast_cancer()
# cancer_data = load_iris()
input_train, input_test, target_train, target_test = train_test_split(cancer_data['data'], cancer_data['target'], test_size=0.2, random_state=42)

model_svm = SVC(probability=True, kernel='linear', verbose=0)

model_ABC = AdaBoostClassifier(n_estimators=1, estimator=model_svm, learning_rate=0.1)
model_ABC_100 = AdaBoostClassifier(n_estimators=10, estimator=model_svm, learning_rate=1)
model_ABC_200 = AdaBoostClassifier(n_estimators=20, estimator=model_svm, learning_rate=1)
model_ABC_300 = AdaBoostClassifier(n_estimators=30, estimator=model_svm, learning_rate=1)
model_ABC_400 = AdaBoostClassifier(n_estimators=40, estimator=model_svm, learning_rate=1)

model_ABC.fit(input_train, target_train)
model_ABC_100.fit(input_train, target_train)
model_ABC_200.fit(input_train, target_train)
model_ABC_300.fit(input_train, target_train)
model_ABC_400.fit(input_train, target_train)

pred_ABC     = model_ABC.predict(input_test)
pred_ABC_100 = model_ABC_100.predict(input_test)
pred_ABC_200 = model_ABC_200.predict(input_test)
pred_ABC_300 = model_ABC_300.predict(input_test)
pred_ABC_400 = model_ABC_400.predict(input_test)

# target과 predict 값을 연산비교하여 일치하지 않는 개수가 얼마나 되는지 확인
fail_pred_ABC     = np.logical_xor(target_test, pred_ABC    )
fail_pred_ABC_100 = np.logical_xor(target_test, pred_ABC_100)
fail_pred_ABC_200 = np.logical_xor(target_test, pred_ABC_200)
fail_pred_ABC_300 = np.logical_xor(target_test, pred_ABC_300)
fail_pred_ABC_400 = np.logical_xor(target_test, pred_ABC_400)

print("target = ", target_test)
print("pred = ", pred_ABC)
print("accuracy = ", metrics.accuracy_score(target_test, pred_ABC))
print("accuracy = ", metrics.accuracy_score(target_test, pred_ABC_100))
print("accuracy = ", metrics.accuracy_score(target_test, pred_ABC_200))
print("accuracy = ", metrics.accuracy_score(target_test, pred_ABC_300))
print("accuracy = ", metrics.accuracy_score(target_test, pred_ABC_400))



print('process finished')