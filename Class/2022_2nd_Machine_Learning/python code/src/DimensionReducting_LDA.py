import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from pandas.plotting import scatter_matrix
from sklearn.model_selection import cross_val_score

G_SHOW_VARIANCE = 0

# 1. 데이터 준비
cancer_data = load_breast_cancer()
df_cancer_data = pd.DataFrame(data=cancer_data.data, columns=cancer_data.feature_names)
df_cancer_data['target'] = cancer_data.target

print(cancer_data.feature_names)
print(len(cancer_data.feature_names))
print(df_cancer_data.head())
# 기본 데이터 분산정도 확인
if G_SHOW_VARIANCE:
    plt.figure("Variance of cancer data (all)")
    plt.title("Variance of cancer data (all)")
    scatter_matrix(df_cancer_data, c=cancer_data['target'], marker='o', s=10, alpha=.8)
    plt.show()

# 2. 데이터 전처리
scaler = StandardScaler()
cancer_data_std = scaler.fit_transform(cancer_data.data)

# 일부 특성의 기본 데이터셋 분산도
markers = ["*","s"]
fig = plt.figure('3-feature variances', figsize=(20,20))
ax = fig.add_subplot(121, projection='3d')
for i,markers in enumerate(markers):
    f1 = df_cancer_data[df_cancer_data["target"]==i]['mean radius']
    f2 = df_cancer_data[df_cancer_data["target"]==i]['mean texture']
    f3 = df_cancer_data[df_cancer_data["target"]==i]['mean perimeter']
    ax.scatter(f1,f2,f3, marker='s', label=cancer_data.target_names[i])
ax.set_xlabel('mean radius')
ax.set_ylabel('mean texture')
ax.set_zlabel('mean perimeter')
plt.legend()

# LDA
lda2 = LinearDiscriminantAnalysis(n_components=1)
lda2.fit(cancer_data_std, cancer_data['target'])
df_lda2 = lda2.transform(cancer_data_std)
df_lda2 = pd.DataFrame(data=df_lda2)
df_lda2['target'] = cancer_data['target']

# 차원 축소 데이터 그래프
markers = ["s","s"]
ax = fig.add_subplot(122)
for i,markers in enumerate(markers):
    f1 = df_lda2[df_lda2["target"]==i][0]
    ax.plot(f1, marker='s', label=cancer_data.target_names[i])
ax.set_xlabel('component 1')
ax.set_ylabel('component 2')
plt.title("LDA Dimension Reduction")
plt.legend()

plt.show()
var_ratio_lda2 = lda2.explained_variance_ratio_
print("<<<Variance Ratio>>>")
print("           2 : ", var_ratio_lda2)


# Random Forest 분류기로 비교
rf = RandomForestClassifier()
scores_pure = cross_val_score(rf, cancer_data['data'], cancer_data['target'], scoring='accuracy', cv=5)
scores_lda2 = cross_val_score(rf, df_lda2.iloc[:,:-1], cancer_data['target'], scoring='accuracy', cv=5)
print("<<<Accuracy>>>")
print("     Dim) 30(pure) : ", np.mean(scores_pure))
print("           2       : ", np.mean(scores_lda2))

print(scores_pure)


input_train, input_test, target_train, target_test = train_test_split(cancer_data['data'], cancer_data['target'], test_size=0.2, random_state=42)

























