from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# 데이터 불러오기
raw_iris = datasets.load_iris()
# 피쳐, 타겟 데이터 지정
X = raw_iris.data
y = raw_iris.target
# target label 값을 0과 0 이외로 치환
y = np.where(y >= 1, 1, y)
print(y)
print(X.shape)
print(pd.DataFrame(X).head())
# 트레이닝/테스트 데이터 분할
X_tn, X_te, y_tn, y_te=train_test_split(X,y,random_state=1)
# 데이터 표준화
std_scale = StandardScaler()
std_scale.fit(X_tn)
X_tn_std = std_scale.transform(X_tn)
X_te_std = std_scale.transform(X_te)
print(X_tn_std.shape)
# 데이터 크기 확인
print(y_tn.shape)
print(pd.DataFrame(y_tn).head())
feature_id = 3
# y 레이블 확인
# index of feature
patient_num = 50 # first number of iris
plt.figure(1, [10, 5])
plt.subplot(121)
plt.plot(np.linspace(0, 4, 5), X_tn_std[0:5, feature_id], marker = "x")
plt.title('Feature plot', fontsize = 20)
plt.subplot(122)
plt.scatter(X_tn_std[0:patient_num, feature_id], y_tn[0:patient_num])
plt.title('First ' + str(patient_num) + ' iris for ' + str(feature_id) + '-th feature', fontsize = 20)
plt.show()
x = X_tn_std[0:patient_num, feature_id]
y = y_tn[0:patient_num]
print(y)
float_epsilon = np.finfo(float).eps
Y = np.log(y/(1 - y + float_epsilon) + float_epsilon)
print(Y)
A = [[np.sum(x * x), np.sum(x)], [np.sum(x), len(x)]]
b = [np.sum(Y * x), np.sum(Y)]
U = np.linalg.solve(A, b)
print(U)
W = U[0]
b = U[1]
plt.figure(2, [10, 5])
plt.subplot(121)
plt.scatter(x, y)
plt.subplot(121)
x_ = np.linspace(-2, 2, 20)
plt.plot(x_, 1 / (1 + np.exp(-W * x_ - b)), 'r') # logistic app. sol.
plt.title('Original plot for iris data')
plt.subplot(122)
plt.scatter(x, 1 / (1 + np.exp(-W * x - b)))
plt.subplot(122)
x_ = np.linspace(-2, 2, 20)
plt.plot(x_, 1 / (1 + np.exp(-W * x_ - b)), 'r') # logistic app. sol.
plt.title('Logistic plot for iris data')
plt.show()
clf_logi_l2 = LogisticRegression(penalty='l2')
clf_logi_l2.fit(X_tn_std, y_tn)
pred_logistic = clf_logi_l2.predict(X_te_std)
pred_proba = clf_logi_l2.predict_proba(X_te_std)
precision = precision_score(y_te, pred_logistic)
conf_matrix = confusion_matrix(y_te, pred_logistic)
class_report = classification_report(y_te, pred_logistic)