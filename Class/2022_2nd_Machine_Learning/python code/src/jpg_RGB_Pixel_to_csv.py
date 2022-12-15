from PIL import Image
import numpy as np

img = Image.open('../docs/HandwriteNumber/28x28 black/hand write_0.jpg')

arr_img = np.asarray(img)
print(arr_img.shape)

arr_list = []
for row in arr_img:
    tmp=[]
    for col in row:
        tmp.append(str(col))
    arr_list.append(tmp)


with open('test_file.csv', 'w') as f:
    for row in arr_list:
        f.write(','.join(row) + '/n')