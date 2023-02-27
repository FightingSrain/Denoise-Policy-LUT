



import numpy as np
import cv2

# 读取文件夹中的所有文件
# def get_filelist(path):
#     import os
#     filelist = []
#     for root, dirs, files in os.walk(path):
#         for file in files:
#             filelist.append(os.path.join(root, file))
#     return filelist

# path = "D://BaiduNetdiskDownload/SIDD-train/train"
# res = get_filelist(path)
# print(res)


# 读取文件夹中的所有含有GT的.PNG文件保存为SIDDGT.txt文件
def get_filelist(path):
    import os
    filelist = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.find("NOISY") != -1 and file.find(".PNG") != -1:
                filelist.append(os.path.join(root, file))
    return filelist

# 保存为SIDDGT.txt文件
def save_filelist(filelist, path):
    with open(path, "w") as f:
        for file in filelist:
            f.write(file + "\n")
# 生成SIDDGT.txt文件和SIDDNOISY.txt文件
# path = "D://BaiduNetdiskDownload/SIDD-train/train"
# res = get_filelist(path)
# save_filelist(res, "SIDDNOISY.txt")
# print(res)
#---------------------------



# 读取.txt文件中的所有文件路径
def get_filelist1(path):
    with open(path, "r") as f:
        filelist = f.readlines()
    return filelist
M = 512 # 图像块的高度
N = 512 # 图像块的宽度
x_step = M # x方向上的步长
y_step = N # y方向上的步长
res = get_filelist1("SIDDGT.txt")
for i in range(len(res)):
    res[i] = res[i].strip()
    img = cv2.imread(res[i])
    # print(img.shape)
    print(i, len(res))
    print("=================================")
    # 用python将一张大小为(3000, 5328, 3)的图像按顺序从左到右，从上到下切分为（512，512，3）大小的图像块，用循环实现
    for x in range(0, img.shape[0], x_step):
        for y in range(0, img.shape[1], y_step):
            tile = img[x:x + M, y:y + N]
            # print(tile.shape)
            # print("D://BaiduNetdiskDownload/SIDD-train/train_512/Noisy/" + str(i) + "_" + str(x) + "_" + str(y) + ".png")
            cv2.imwrite("D://BaiduNetdiskDownload/SIDD-train/train_512/GT/" + str(i) + "_" + str(x) + "_" + str(y) + ".png", tile)
    # cv2.imwrite(res[i], img)
    # print(img.shape)


















