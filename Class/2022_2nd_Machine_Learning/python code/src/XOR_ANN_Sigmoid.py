import numpy as np
import matplotlib.pyplot as plt

input = [[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]]
output = [0, 1, 1, 0]

N = np.size(input, 0)  # number of samples

Ni = np.size(input, 1)  # dimension of the samples of input

No = 1  # dimension of the sample of output

Nh = 5  # number of hidden units

Ws = 1 / 4 * np.random.rand(Nh, Ni + 1)
# print(Ws)

Wo = 1 / 4 * np.random.rand(No, Nh)
# print(Wo)

alpha = 0.1  # Learning rate\

t_ = []

loss_ = []


def sigmoid(x):
    f = 1 / (1 + np.exp(-x))
    return f


## train the model ====================================================================
for epoch in range(0, 3000):
    loss = 0
    for id_ in range(0, N):
        dWs = 0 * Ws
        dWo = 0 * Wo

        x = np.append(input[id_], 1)

        S = np.dot(Ws, x)

        y = np.dot(Wo, sigmoid(S))

        d = output[id_]

        for j in range(0, Nh):
            for i in range(0, No):
                dWo[i, j] = dWo[i, j] + sigmoid(S[j]) * (y[i] - d)

        Wo = Wo - alpha * dWo

        for k in range(0, Ni + 1):
            for j in range(0, Nh):
                for i in range(0, No):
                    dWs[j, k] = dWs[j, k] + x[k] * Wo[i, j] * sigmoid(S[j]) * (1 - sigmoid(S[j])) * (y[i] - d)

        Ws = Ws - alpha * dWs

        loss = loss + 1 / 2 * np.linalg.norm(y - d)

    if np.mod(epoch, 50) == 0:
        print(epoch, "-th epoch trained")

        t_ = np.append(t_, epoch)

        loss_ = np.append(loss_, loss)

plt.figure(num=0, figsize=[10, 5])
plt.plot(t_, loss_, marker="")
plt.title('Loss decay')
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.show()
print('====================================================================')

## test the trained model ====================================================================
for id_ in range(0, N):
    x = np.append(input[id_], 1)

    S = np.dot(Ws, x)

    y = np.dot(Wo, sigmoid(S))

    print(y)




