import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# 1. 데이터 준비
data = pd.read_csv('../docs/badminton_weather_data.csv')
print(data)
'''
   weather  wind  temperature  humidity  target
0        0     1            0         1       0
1        1     1            2         1       0
2        0     0            2         1       1
3        2     0            1         2       1
4        0     0            1         0       1
5        1     0            0         2       0
6        1     0            1         2       0
7        0     1            2         0       0
'''

# 2. 데이터 전처리
feature = ['weather','wind','temperature','humidity']
data_input = data[feature].to_numpy()
data_target = data[['target']].to_numpy()
input_train, input_test, target_train, target_test = train_test_split(data_input, data_target, random_state=42)

# 3. 모델 생성 및 학습
model_dctree = DecisionTreeClassifier(max_depth=None)
model_dctree.fit(input_train, target_train)

imp = model_dctree.feature_importances_
a = model_dctree.n_outputs_
b = model_dctree.ccp_alpha
c = model_dctree.class_weight
d = model_dctree.criterion
e = model_dctree.tree_.impurity
f = model_dctree.tree_.value

print(e)
print(f)
indices = np.argsort(imp)

print(model_dctree.score(input_train, target_train))
print(model_dctree.score(input_test, target_test))
print("--------- importance (impurity) ----------")
print(feature)
print(imp)
# plt.figure(figsize=(7,7))
plt.title('Feature Importances')
plt.barh(range(len(indices)), imp[indices], color='g', align='center')
plt.yticks(range(len(indices)), [feature[i] for i in indices])
plt.show()

print('process finished')