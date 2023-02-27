# 导入必要的库
import torch
import torch.nn as nn
import torch.nn.functional as F


# import pytorch_lightning as pl
import cv2
import torch
import torch.nn.functional as F
import numpy as np
# 假设输入图像为（3, 3, 256,356）
# input = torch.randn(3, 3, 256, 356)
imgs = cv2.imread('D://Dataset/Kodak24/kodim01.png')
h, w, c = imgs.shape
sample = 4
img = torch.Tensor(imgs.transpose(2, 0, 1)/255.).unsqueeze(0)
# print(img.size())
# 创建一个网格（3，128，178，2），其中每个元素是（x,y）坐标
grid = torch.zeros(1, h//sample, w//sample, 2)
for i in range((h//sample-1)):
    for j in range((w//sample-1)):
        # 将网格坐标从[0，127]和[0，177]映射到[-1，1]
        grid[:, i , j ,0] = (j / (h//sample-1)) * 2 - 1
        grid[:, i , j ,1] = (i / (w//sample-1)) * 2 - 1

# 使用F.grid_sample采样输入图像
output = F.grid_sample(img ,grid)

# 输出结果为（3，3，128，178）
print(output.shape)
cv2.imshow("img", imgs)
cv2.imshow("output", (output[0].permute(1, 2, 0).numpy()*255).astype(np.uint8))
cv2.waitKey(0)