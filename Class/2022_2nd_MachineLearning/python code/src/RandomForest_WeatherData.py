import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime

from sklearn.ensemble import RandomForestClassifier
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
data_target = data[['target']].to_numpy().ravel()
input_train, input_test, target_train, target_test = train_test_split(data_input, data_target, random_state=42)

# 3. 모델 생성 및 학습
model = RandomForestClassifier(n_estimators=10, max_depth=2, random_state=0)
model.fit(data_input, data_target)

# 4. 결과 분석
imp = model.feature_importances_
indices = np.argsort(imp)
print(model.score(input_train, target_train))
print(model.score(input_test, target_test))
print("--------- importance (impurity) ----------")
print(feature)
print(imp)


# -------------------- graph ---------------------------------
indices = np.argsort(imp)
plt.title('Feature Importances')
plt.barh(range(len(indices)), imp[indices], color='g', align='center')
plt.yticks(range(len(indices)), [feature[i] for i in indices])
plt.show()

print('process finished')











