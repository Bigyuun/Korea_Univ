import numpy as np
import matplotlib as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.datasets import fetch_openml


handwrite_data = np.loadtxt('../docs/HandwriteNumber/csv/data_total_grayscale_C2.csv', delimiter=',')
print("handwrite data is opened")
x_handwrite_data = handwrite_data[:, 1:]
y_handwrite_data = handwrite_data[:, 0]


mnist = fetch_openml('mnist_784', version=1)
print(mnist)
input, target = mnist['data'], mnist['target']

print(input)
print(target)

x_train, x_test, y_train, y_test = train_test_split(input, target, test_size=0.2, random_state=42)

model = LinearSVC(random_state=42)
model.fit(x_train, y_train)

pr = model.predict(x_handwrite_data)
print(pr)