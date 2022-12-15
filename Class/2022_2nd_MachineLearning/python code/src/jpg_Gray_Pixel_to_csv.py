import cv2
import time

road = cv2.imread('../docs/HandwriteNumber/28x28 black/hand write_0.jpg')
gray = cv2.cvtColor(road, cv2.COLOR_BGR2GRAY)
height, width = gray.shape
th = 150

start = time.perf_counter()
for i in range(height):
    for j in range(width):
        if gray[i][j] > th:
            gray[i][j] = 255
        else:
            gray[i][j] = 0

finish = time.perf_counter()
print(finish - start)	# 0.343 s

cv2.imshow('after', gray)
cv2.waitKey()

print('done')