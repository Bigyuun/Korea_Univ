import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from pandas.plotting import scatter_matrix
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.wrappers.scikit_learn import KerasRegressor
from keras.optimizers import SGD
from sklearn.model_selection import cross_val_score
import keras.utils
from sklearn import metrics

G_SHOW_VARIANCE = 0

# 1. 데이터 준
cancer_data = load_breast_cancer()
df_cancer_data = pd.DataFrame(data=cancer_data.data, columns=cancer_data.feature_names)
df_cancer_data['target'] = cancer_data.target

print(cancer_data.feature_names)
print(df_cancer_data.head())
# 기본 데이터 분산정도 확인
if G_SHOW_VARIANCE:
    plt.figure("Variance of cancer data (all)")
    plt.title("Variance of cancer data (all)")
    scatter_matrix(df_cancer_data, c=cancer_data['target'], marker='o', s=10, alpha=.8)
    plt.show()


# 2. 데이터 전처리
scaler = StandardScaler()
cancer_data_std = scaler.fit_transform(df_cancer_data)

# 일부 특성의 기본 데이터셋 분산도
markers = ["*","s"]
fig = plt.figure('3-feature variances', figsize=(20,20))
ax = fig.add_subplot(131, projection='3d')
for i,markers in enumerate(markers):
    f1 = df_cancer_data[df_cancer_data["target"]==i]['mean radius']
    f2 = df_cancer_data[df_cancer_data["target"]==i]['mean texture']
    f3 = df_cancer_data[df_cancer_data["target"]==i]['mean perimeter']
    ax.scatter(f1,f2,f3, marker='s', label=cancer_data.target_names[i])
ax.set_xlabel('mean radius')
ax.set_ylabel('mean texture')
ax.set_zlabel('mean perimeter')
plt.title("random feature graph")
plt.legend()

# PCA
# 10th dimension
pca10 = PCA(n_components=10)
pca10.fit(cancer_data_std)
df_pca10 = pca10.transform(cancer_data_std)
df_pca10 = pd.DataFrame(data=df_pca10)
df_pca10['target'] = cancer_data['target']

# 3rd dimension
pca3 = PCA(n_components=3)
pca3.fit(cancer_data_std)
df_pca3 = pca3.transform(cancer_data_std)
df_pca3 = pd.DataFrame(data=df_pca3)
df_pca3['target'] = cancer_data['target']
# 차원 축소 데이터 그래프
markers = ["*","s"]
ax = fig.add_subplot(132, projection='3d')
for i,markers in enumerate(markers):
    f1 = df_pca3[df_pca3["target"]==i][0]
    f2 = df_pca3[df_pca3["target"]==i][1]
    f3 = df_pca3[df_pca3["target"]==i][2] # scaling
    ax.scatter(f1,f2,f3, marker='s', label=cancer_data.target_names[i])
ax.set_xlabel('mean radius')
ax.set_ylabel('mean texture')
ax.set_zlabel('mean perimeter')
plt.title("3-feature graph")
plt.legend()

# 2nd dimension
pca2 = PCA(n_components=2)
pca2.fit(cancer_data_std)
df_pca2 = pca2.transform(cancer_data_std)
df_pca2 = pd.DataFrame(data=df_pca2)
df_pca2['target'] = cancer_data['target']
# 차원 축소 데이터 그래프
markers = ["*","s"]
ax = fig.add_subplot(133)
for i,markers in enumerate(markers):
    f1 = df_pca2[df_pca2["target"]==i][0]
    f2 = df_pca2[df_pca2["target"]==i][1]
    ax.scatter(f1,f2, marker='s', label=cancer_data.target_names[i])
ax.set_xlabel('mean radius')
ax.set_ylabel('mean texture')
plt.title("2-feature graph")
plt.legend()


plt.show()

var_ratio_pca10 = pca10.explained_variance_ratio_
var_ratio_pca3 = pca3.explained_variance_ratio_
var_ratio_pca2 = pca2.explained_variance_ratio_
print("<<<Variance Ratio>>>")
print("     Dim) 10 : ", var_ratio_pca10)
print("          3 : ", var_ratio_pca3)
print("          2 : ", var_ratio_pca2)
print("각 차원의 초기 2(3)개의 주성분 요소로 원본데이터의 약 {}%를 표현" .format(np.sum(var_ratio_pca10[0:3])*100))

# Random Forest 분류기로 비교
rf = RandomForestClassifier()
scores_pure = cross_val_score(rf, cancer_data['data'], cancer_data['target'], scoring='accuracy', cv=5)
scores_pca10 = cross_val_score(rf, df_pca10.iloc[:,:-1], cancer_data['target'], scoring='accuracy', cv=5)
scores_pca3 = cross_val_score(rf, df_pca3.iloc[:,:-1], cancer_data['target'], scoring='accuracy', cv=5)
scores_pca2 = cross_val_score(rf, df_pca2.iloc[:,:-1], cancer_data['target'], scoring='accuracy', cv=5)
print("<<<Accuracy>>>")
print("     Dim) 30(pure) : ", np.mean(scores_pure))
print("          10       : ", np.mean(scores_pca10))
print("           3       : ", np.mean(scores_pca3))
print("           2       : ", np.mean(scores_pca2))
























