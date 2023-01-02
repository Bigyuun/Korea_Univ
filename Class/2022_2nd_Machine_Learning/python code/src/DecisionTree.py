import numpy as np
import matplotlib as plt
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer_data = load_breast_cancer()
# input_train, input_test, target_train, target_test = train_test_split(cancer_data.data, cancer_data.target, random_state=42)
input_train, input_test, target_train, target_test = train_test_split(cancer_data['data'], cancer_data['target'], random_state=42)

model_dctree = DecisionTreeClassifier(random_state=0)
model_dctree.fit(input_train, target_train)

print(model_dctree.score(input_train, target_train))
print(model_dctree.score(input_test, target_test))

print('process finished')