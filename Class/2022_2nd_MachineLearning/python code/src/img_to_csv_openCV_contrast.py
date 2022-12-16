import datetime
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd

G_MAKE_CSV = 0

# img = cv2.imread('../docs/HandwriteNumber/28x28/hand write_9.jpg')
img = cv2.imread('../docs/HandwriteNumber/hand write_9.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# img_mod_pixel = cv2.resize(img, dsize=(28,28), interpolation=cv2.INTER_CUBIC)

img_gray_cnvt = 255 - img_gray
img_gray_1order = 255 - img_gray.flatten()
img_gray_1order = img_gray_1order.reshape(1,-1)

# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# img_gray_contrast = clahe.apply(img_gray_cnvt)
alpha_contrast = 2.0
dst1 = np.clip((1+alpha_contrast)*img_gray_cnvt - 128 * alpha_contrast, 0, 255).astype(np.uint8)
img_mod_pixel_origin = cv2.resize(dst1, dsize=(28,28), interpolation=cv2.INTER_AREA)
img_mod_pixel = img_mod_pixel_origin.reshape(1,-1)

if G_MAKE_CSV:
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    np.savetxt('../docs/HandwriteNumber/csv/'+now+'.csv', img_mod_pixel, fmt='%d', delimiter=',')
# pd.DataFrame(img_gray_1order).to_csv(now+'.csv', index=False)

plt.subplot(1,3,1)
plt.imshow(img)
plt.subplot(1,3,2)
plt.imshow(img_gray_cnvt)
plt.subplot(1,3,3)
plt.imshow(img_mod_pixel_origin)
plt.show()

# cv2.imshow('origin', img)
# cv2.imshow('gray', img_gray_cnvt)
# # cv2.imshow('dst1', dst1)
# cv2.imshow('dst1', img_mod_pixel_origin)
# cv2.waitKey(0)
# cv2.destoryAllWindows()

