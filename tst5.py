


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
# t1 = time.time()
# res = cv2.fastNlMeansDenoising(img,  h=20, templateWindowSize=7, searchWindowSize=21)
# t2 = time.time()
# print(t2-t1)
# cv2.imshow("ins", img)
# cv2.imshow("res", res)
# cv2.imshow("imgs", imgs)
# cv2.waitKey(0)

import cv2
import numpy as np

# 读取原始图像和加噪图像
img = cv2.imread('./Train_Hybrid_Policy/res_img/Hybrid_BSD68/res0.png', 0) / 255.
noisy_img = cv2.imread('D://Dataset/BSD68/test001.png', 0) / 255.
noise = np.random.normal(0, 25, img.shape).astype(img.dtype) / 255.
noisy_img = np.clip(img + noise, a_min=0., a_max=1.)

# 定义引导滤波的参数
r = 5 # 引导滤波的半径
eps = 0.01 # 引导滤波的正则化参数

# 使用opencv自带的GuidedFilter类进行引导滤波
gf = cv2.ximgproc.createGuidedFilter((img*255-10).astype(np.uint8), r, eps)
denoised_img = gf.filter((noisy_img*255).astype(np.uint8))

# 显示结果并保存
cv2.imshow('Original', (img*255).astype(np.uint8))
cv2.imshow('Noisy', (noisy_img*255).astype(np.uint8))
cv2.imshow('Denoised', (denoised_img).astype(np.uint8))
# cv2.imwrite('denoised.jpg', denoised_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
