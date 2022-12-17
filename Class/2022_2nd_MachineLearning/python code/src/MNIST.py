import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from os.path import exists
import pandas as pd


# 1. 데이터 준비
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

handwrite_data = np.loadtxt('../docs/HandwriteNumber/csv/data_total_grayscale_C2.csv', delimiter=',')
print("handwrite data is opened")
x_handwrite_data = handwrite_data[:, 1:].reshape(10,28,28)
y_handwrite_data = handwrite_data[:, 0]

# 2. 데이터 전처리
x_train, x_test = (x_train/255.0 + 0.001), (x_test/255.0 + 0.001)

# 3. 모델 구성
file_exist_flag = exists("../src_model/XOR_ANN_epoch1.h5")
if file_exist_flag :
    model = tf.keras.models.load_model('../src_model/XOR_ANN_epoch10.h5')
else:
    model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)),
                                        tf.keras.layers.Dense(512, activation=tf.nn.relu),
                                        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
                                        ])
# 4. 모델 컴파일
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
# 5. 모델 훈련
    history = model.fit(x_train, y_train, epochs=10, verbose=1)

# 6. 정확도 평가
train_loss, train_acc = model.evaluate(x_train, y_train, verbose=0)
print("Loss (Train) : ", train_loss, " Accuracy (Train) : ", train_acc)
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print("Loss (Test) : ", test_loss, " Accuracy (Test) : ", test_acc)

# 모델 저장
model.save("../src_model/XOR_ANN_epoch10.h5")

# 7. 손글씨 데이터 예측
predict = model.predict(x_handwrite_data)
pr = []
for i in range(0,10):
    pr.append(np.argmax(predict[i]))
print("예측된 숫자 : ", pr)

history_loss = history.history['loss']
history_acc = history.history['accuracy']


# -------------------- graph ---------------------------------
plt.subplot(1,2,1)
plt.title("Loss of model")
plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot(range(len(history_loss)), history_loss, 'r')
plt.grid()
plt.subplot(1,2,2)
plt.title("Accuracy of model")
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.plot(range(len(history_acc)), history_acc)
plt.grid()

plt.show()

print("end")



