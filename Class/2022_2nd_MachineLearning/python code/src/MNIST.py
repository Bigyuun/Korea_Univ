import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd

# 1. MNIST 데이터셋 임포트
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

handwrite_data = np.loadtxt('../docs/HandwriteNumber/csv/data_total_grayscale_C2.csv', delimitemnistr=',')
print("handwrite data is opened")
x_handwrite_data = handwrite_data[:, 1:].reshape(10,28,28)
y_handwrite_data = handwrite_data[:, 0]

# 2. 데이터 전처리
x_train, x_test = (x_train/255.0 + 0.001), (x_test/255.0 + 0.001)

# 2.1 손글씨 배경이 완전 검정색이 아님에 따라, 배경을 검정으로 만들어주는 전처리 진행
# max_handwrite_data = np.max(x_handwrite_data)
# min_handwrite_data = np.min(x_handwrite_data)
# coef = max_handwrite_data - min_handwrite_data
# zero_3d = np.zeros((10,28,28))
# x_handwrite_data = np.maximum( (x_handwrite_data-60), zero_3d )
# x_handwrite_data = (x_handwrite_data/255.0 + 0.001)
# max2_x_handwrite_data = np.max(x_handwrite_data)
# x_handwrite_data = x_handwrite_data*1/max2_x_handwrite_data
# print(np.max(x_handwrite_data))

# 3. 모델 구성
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)),
                                    tf.keras.layers.Dense(512, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
                                    ])

# 4. 모델 컴파일
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5. 모델 훈련
history = model.fit(x_train, y_train, epochs=10)

# 6. 정확도 평가
train_loss, train_acc = model.evaluate(x_train, y_train, verbose=0)
print("Loss (Train) : ", train_loss, " Accuracy (Train) : ", train_acc)
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print("Loss (Test) : ", test_loss, " Accuracy (Test) : ", test_acc)

# 모델 저장
model.save("./save_model.h5")

# 7. 손글씨 데이터 예측
predict = model.predict(x_handwrite_data)
pr = []
for i in range(0,10):
    pr.append(np.argmax(predict[i]))
print(pr)

history_loss = history.history['loss']
history_acc = history.history['accuracy']

plt.subplot(1,2,1)
plt.plot(range(len(history_loss)), history_loss)
plt.grid()
plt.subplot(1,2,2)
plt.plot(range(len(history_acc)), history_acc)

plt.show()

print("end")



