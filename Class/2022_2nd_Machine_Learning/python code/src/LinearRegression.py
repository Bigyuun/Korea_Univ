import numpy as np
from matplotlib import pyplot as plt
from time import sleep
from tqdm import tqdm

perch_length = np.array(
    [13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0,
     21.0, 21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5,
     22.5, 22.7, 23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5,
     27.3, 27.5, 27.5, 27.5, 28.0, 28.7]
     )
perch_weight = np.array(
    [32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0,
     110.0, 115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0,
     130.0, 150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0,
     197.0, 218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0]
     )

# plt.scatter(bream_length, bream_weight)
# plt.xlabel('length')
# plt.ylabel('weight')
# plt.show()

w = np.random.rand(1)
b = np.random.rand(1)
print('init w : ', w)
print('init b : ', b)
lamda = 1e-3
iteration = 20000

N = np.size(perch_weight)

input = perch_length
target = perch_weight

for i in tqdm(range(0,iteration)) :
    for j in range(0, len(input)):
        output = np.dot(np.asarray([w,b]).T, np.asarray([input[j], 1]))

        dedw = input[j]
        dLde = output - target[j]
        dedb = 1
        dLdw = dLde * dedw
        dLdb = dLde * dedb

        w = w - lamda * dLdw
        b = b - lamda * dLdb

cost = np.sum( ((input * w + b) - target )**2 ) / len(input)

print('weight : ', w)
print('bais   : ', b)
plt.plot(perch_length, w*perch_length+b, '-r', input,target,marker="x")
plt.title('Regression using Gradient Descent Method')
plt.show()







