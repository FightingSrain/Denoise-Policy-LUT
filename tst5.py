


import numpy as np
import cv2
import time

img = cv2.imread("D://Dataset/DND_patches/28_noise.png", 1)
imgs = cv2.imread("D://Dataset/DND_patches/02_noise.png")[:, :, 2]
# res = cv2.GaussianBlur(img, (7, 7), 1.5)
# res = cv2.GaussianBlur(res, (7, 7), 1.5)
# res = cv2.GaussianBlur(res, (7, 7), 1.5)
# res = cv2.bilateralFilter(res, 5, 1.5, 5)
# res = cv2.boxFilter(img, ddepth=-1, ksize=(5, 5))
# res = cv2.medianBlur(res, 7)
# res = cv2.blur(res, (5, 5))
t1 = time.time()
res = cv2.fastNlMeansDenoising(img,  h=20, templateWindowSize=7, searchWindowSize=21)
t2 = time.time()
print(t2-t1)
cv2.imshow("ins", img)
cv2.imshow("res", res)
cv2.imshow("imgs", imgs)
cv2.waitKey(0)


