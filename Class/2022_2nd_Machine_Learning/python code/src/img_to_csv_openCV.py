import datetime
import numpy as np
import cv2
import pandas as pd

G_MAKE_CSV = 0

img = cv2.imread('../docs/HandwriteNumber/28x28/hand write_9.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img_gray_cnvt = 255 - img_gray
img_gray_1order = 255 - img_gray.flatten()
img_gray_1order = img_gray_1order.reshape(1,-1)

if G_MAKE_CSV:
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    np.savetxt('../docs/HandwriteNumber/csv/'+now+'.csv', img_gray_1order, fmt='%d', delimiter=',')
# pd.DataFrame(img_gray_1order).to_csv(now+'.csv', index=False)

# cv2.imshow('origin', img)
cv2.imshow('gray', img_gray_cnvt)
cv2.waitKey(0)

