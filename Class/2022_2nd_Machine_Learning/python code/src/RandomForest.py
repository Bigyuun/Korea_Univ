import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 1. 데이터 준비
cancer_data = load_breast_cancer()

# 2. 데이터 전처리
input_train, input_test, target_train, target_test = train_test_split(cancer_data['data'], cancer_data['target'], random_state=42)

# 3. 모델 생성 및 학습
model = RandomForestClassifier(n_estimators=10, max_depth=2, random_state=0)
model.fit(input_train, target_train)

# 4. 결과 분석
imp = model.feature_importances_
indices = np.argsort(imp)

print(model.score(input_train, target_train))
print(model.score(input_test, target_test))

# -------------------- graph ---------------------------------
plt.title('Feature Importances')
plt.barh(range(len(indices)), imp[indices], color='g', align='center')
plt.show()

print('process finished')











