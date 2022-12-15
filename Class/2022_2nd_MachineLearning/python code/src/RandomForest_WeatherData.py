import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

data = pd.read_csv('../docs/badminton_weather_data.csv')
print(data.head())

# Split data to input & target
feature = ['weather','wind','temperature','humidity']
data_input = data[feature].to_numpy()
data_target = data[['target']].to_numpy().ravel()

input_train, input_test, target_train, target_test = train_test_split(data_input, data_target, random_state=42)

model = RandomForestClassifier(n_estimators=10, max_depth=2, random_state=0)
model.fit(data_input, data_target)

imp = model.feature_importances_
indices = np.argsort(imp)

print(model.score(input_train, target_train))
print(model.score(input_test, target_test))

# plt.figure(figsize=(7,7))
plt.title('Feature Importances')
plt.barh(range(len(indices)), imp[indices], color='g', align='center')
plt.yticks(range(len(indices)), [feature[i] for i in indices])
plt.show()

print('process finished')











