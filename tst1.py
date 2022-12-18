from PIL import Image

# # Open the image
# im = Image.open('./img_tst/test001.png')
#
# # Use the ImageDraw module to draw lines on the image
# from PIL import ImageDraw
# draw = ImageDraw.Draw(im)
#
# # Draw lines on the image to create a "liquid" effect
# for i in range(0, im.size[0], 10):
#     draw.line((i, 0) + im.size, fill=128)
#     draw.line((0, i) + im.size, fill=128)
#
# # Save the liquidized image
# im.save('./img_tst/test001s.png')



import cv2
import time
import copy
import numpy as np

img = cv2.imread('./img_tst/test001.png', 0)
raw_n = np.random.normal(0, 15, img.shape).astype(np.float32) / 255.
imgs = np.clip(img/255. + raw_n, a_min=0., a_max=1.)
imgs = (imgs * 255).astype(np.uint8)

cv2.imshow('imgs', imgs)
cv2.waitKey(0)

print(cv2.PSNR(img, imgs))

res = None

len = 1
t1 = time.time()
for i in range(len):
    # imgs = cv2.GaussianBlur(imgs, ksize=(5, 5), sigmaX=0.4) # 0.001s
    # imgs = cv2.bilateralFilter(imgs, d=5, sigmaColor=0.1, sigmaSpace=5) # 0.002s
    # imgs = cv2.boxFilter(imgs, ddepth=-1, ksize=(5, 5)) # 很快，基本不花时间
    # imgs = cv2.medianBlur(imgs, ksize=5) # 0.001s
    imgs = cv2.fastNlMeansDenoising(imgs, h=10, templateWindowSize=7, searchWindowSize=21) # 0.14s

    if i == len - 1:
        res = copy.deepcopy(imgs)
t2 = time.time()
cv2.imshow('res', res)
cv2.waitKey(0)
print(cv2.PSNR(img, res))
print(t2 - t1)