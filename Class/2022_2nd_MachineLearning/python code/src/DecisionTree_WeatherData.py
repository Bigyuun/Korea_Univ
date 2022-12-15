import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

data = pd.read_csv('../docs/badminton_weather_data.csv')
print(data.head())

# Split data to input & target
feature = ['weather','wind','temperature','humidity']
data_input = data[feature].to_numpy()
print(data_input)
data_target = data[['target']].to_numpy()

input_train, input_test, target_train, target_test = train_test_split(data_input, data_target, random_state=42)
# input_train, target_train = data_input, data_target

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
# print(model_dctree.score(input_test, target_test))

# plt.figure(figsize=(7,7))
plt.title('Feature Importances')
plt.barh(range(len(indices)), imp[indices], color='g', align='center')
plt.yticks(range(len(indices)), [feature[i] for i in indices])
plt.show()

print('process finished')