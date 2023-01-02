import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd

set_model = tf.keras.models.load_model('../src_model/XOR_ANN_epoch10.h5')

handwrite_data = np.loadtxt('../docs/HandwriteNumber/csv/data_total_grayscale_C2.csv', delimiter=',')
print("handwrite data is opened")
x_handwrite_data = handwrite_data[:, 1:].reshape(10,28,28)
y_handwrite_data = handwrite_data[:, 0]

predict = set_model.predict(x_handwrite_data)
print(predict[0])
pr = []
for i in range(0,10):
    print(predict[i])
    pr.append(np.argmax(predict[i]))
print(pr)






