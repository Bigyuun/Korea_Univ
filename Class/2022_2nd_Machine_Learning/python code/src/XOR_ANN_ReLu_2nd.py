
import numpy as np
import matplotlib as plt
from tqdm import tqdm

# user Functions
def ReLu(x):
    return np.maximum(0,x)

print(ReLu(-22.33223))
print(ReLu(0.134234324))
# data set for XOR
input = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float_)
target = np.array([[0],  [1],  [1],  [0] ] , dtype=np.float_)

NumOfData, Columns = np.shape(input)      # number of data size
DimOfData = np.size(input,1)    # Dimension of data
DimOfTarget = np.size(target,1)
NumOfH = 4                      # number of hidden units

Wh = np.random.rand(NumOfH, DimOfData)
#print(Ws)
Wo = np.random.rand(NumOfH, DimOfTarget)
#print(Wo)
output_hidden = np.zeros((NumOfH,1))
output_fin = np.zeros((1,1))

epoch = 3000
learning_rate = 1e-3

for i in tqdm(range(0, epoch)):
    for j in range(0,NumOfData):
        # Feed-Forward
        debug = input[j]
        output_hidden = np.dot(Wh,input[j])
        output_hidden = ReLu(output_hidden).reshape(-1,1)
        output_fin = np.dot(Wo.T, output_hidden)
        output_fin = ReLu(output_fin).reshape(-1,1)

        # Back-Propagation
        grad_output = target[j] - output_fin
        grad_hidden = Wo*grad_output
        # grad_hidden = np.sum(Wo*grad_output)

        Wo = Wo - learning_rate * grad_output * output_hidden
        Wh = Wh - learning_rate * grad_hidden * input[j].T





print(input[1])
print(input[1,1])
print("end")




