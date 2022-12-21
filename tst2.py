import cv2

def bi_demo(image):      #双边滤波
    dst = cv2.bilateralFilter(image, 0, 100, 5)
    cv2.imshow("bi_demo", dst)

def shift_demo(image):   #均值迁移
    dst = cv2.pyrMeanShiftFiltering(image, 10, 50)
    cv2.imshow("shift_demo", dst)

src = cv2.imread('./img_tst/test001.png')
img = cv2.resize(src, None, fx=0.8, fy=0.8,
                 interpolation=cv2.INTER_CUBIC)
cv2.imshow('input_image', img)

bi_demo(img)
shift_demo(img)

cv2.waitKey(0)
cv2.destroyAllWindows()