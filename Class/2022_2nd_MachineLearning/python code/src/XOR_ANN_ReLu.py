import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 1. 데이터 준비
input = [[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]]
output = [0, 1, 1, 0]

N = np.size(input, 0)  # number of samples
Ni = np.size(input, 1)  # dimension of the samples of input
No = 1  # dimension of the sample of output
Nh = 5  # number of hidden units
Ws = 1 / 4 * np.random.rand(Nh, Ni + 1)
# print(Ws)
Wo = 1 / 4 * np.random.rand(No, Nh)

# 2. 학습 파라미터 설정
alpha = 0.1  # Learning rate
iteration = 3000

t_ = []
loss_ = []

def ReLu(x):
    return np.maximum(0,x)

# 3. 모델 훈련
for epoch in tqdm(range(0, iteration)):
    loss = 0
    for id_ in range(0, N):
        dWs = 0 * Ws
        dWo = 0 * Wo

        x = np.append(input[id_], 1)
        S = np.dot(Ws, x)
        y = np.dot(Wo, ReLu(S))
        d = output[id_]

        for j in range(0, Nh):
            for i in range(0, No):
                dWo[i, j] = dWo[i, j] + ReLu(S[j]) * (y[i] - d)

        Wo = Wo - alpha * dWo

        for k in range(0, Ni + 1):
            for j in range(0, Nh):
                for i in range(0, No):
                    dWs[j, k] = dWs[j, k] + x[k] * Wo[i, j] * ReLu(S[j]) * (1 - ReLu(S[j])) * (y[i] - d)

        Ws = Ws - alpha * dWs

        loss = loss + 1 / 2 * np.linalg.norm(y - d)


##-------------------------- for display graph ----------------------------------------
    if np.mod(epoch, 5) == 0:
        # print(epoch, "-th epoch trained")
        t_ = np.append(t_, epoch)
        loss_ = np.append(loss_, loss)

    if (epoch == 0) or (epoch == iteration-1) :

        plt.figure(num=0, figsize=[10, 5])
        plt.plot(t_, loss_, marker="")
        plt.title('Loss decay')
        plt.xlabel('epoch')
        plt.ylabel('Loss')

        ## figure out the function shape the model==========================================
        xn = np.linspace(0, 1, 20)
        yn = np.linspace(0, 1, 20)
        xm, ym = np.meshgrid(xn, yn)
        xx = np.reshape(xm, np.size(xm, 0) * np.size(xm, 1))
        yy = np.reshape(ym, np.size(xm, 0) * np.size(xm, 1))
        Z = []

        for id__ in range(0, np.size(xm)):
            x = np.append([xx[id__], yy[id__]], [1, 1])
            S = np.dot(Ws, x)
            y_ = np.dot(Wo, ReLu(S))
            Z = np.append(Z, y_)

        if epoch==0:
            fig = plt.figure(num=1, figsize=[10, 5])
            ax = fig.add_subplot(121, projection='3d')
            surf = ax.plot_surface(xm, ym, np.reshape(Z, (np.size(xm, 0), np.size(xm, 1))), cmap='coolwarm', linewidth=0,
                                   antialiased=False)
        if epoch==iteration-1:
            ax = fig.add_subplot(122, projection='3d')
            surf = ax.plot_surface(xm, ym, np.reshape(Z, (np.size(xm, 0), np.size(xm, 1))), cmap='coolwarm', linewidth=0,
                                   antialiased=False)
print("====================================================================")

# 4. 결과 확인
for id_ in range(0, N):
    x = np.append(input[id_], 1)
    S = np.dot(Ws, x)
    y = np.dot(Wo, ReLu(S))
    print("Input : ", input[id_], "predict : ", y)

# -------------------- graph ---------------------------------
plt.figure(num=0, figsize=[10, 5])
plt.plot(t_, loss_, marker="")
plt.title('Loss decay')
plt.xlabel('epoch')
plt.ylabel('Loss')

plt.show()

print('end')