import numpy as np
import matplotlib as plt
import pandas as pd

from sklearn.svm import SVC
# from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import plot_tree
from sklearn.model_selection import cross_val_score

cancer_data = load_breast_cancer()
# cancer_data = load_iris()
input_train, input_test, target_train, target_test = train_test_split(cancer_data['data'], cancer_data['target'], test_size=0.2, random_state=42)

model_svm = SVC(probability=True, kernel='linear', verbose=1)

model_ABC     = AdaBoostClassifier(n_estimators=1, estimator=model_svm, learning_rate=1)
model_ABC_100 = AdaBoostClassifier(n_estimators=10, estimator=model_svm, learning_rate=1)
model_ABC_200 = AdaBoostClassifier(n_estimators=20, estimator=model_svm, learning_rate=1)
model_ABC_400 = AdaBoostClassifier(n_estimators=40, estimator=model_svm, learning_rate=1)

model_ABC.fit(input_train, target_train)
model_ABC_100.fit(input_train, target_train)
model_ABC_200.fit(input_train, target_train)
model_ABC_400.fit(input_train, target_train)

pred_ABC     = model_ABC.predict(input_test)
pred_ABC_100 = model_ABC_100.predict(input_test)
pred_ABC_200 = model_ABC_200.predict(input_test)
pred_ABC_400 = model_ABC_400.predict(input_test)




score = cross_val_score(model_ABC, input_test, target_test, cv=5)
print(score.mean())
print(score.std())

# imp = model_ABC.feature_importances_
class_ABC = model_ABC.get_params()
#
# print(imp)
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
print("------------- rate of fail matches ----------------")
print(rate_of_fail_pred_ABC)
print(rate_of_fail_pred_ABC_100)
print(rate_of_fail_pred_ABC_200)
print(rate_of_fail_pred_ABC_400)

print("target = ", target_test)
# print("pred = ", pred_ABC)
print("accuracy = ", metrics.accuracy_score(target_test, pred_ABC))
print("accuracy = ", metrics.accuracy_score(target_test, pred_ABC_100))
print("accuracy = ", metrics.accuracy_score(target_test, pred_ABC_200))
print("accuracy = ", metrics.accuracy_score(target_test, pred_ABC_400))

print('process finished')