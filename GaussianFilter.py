import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class Gaussfilter(nn.Module):
    def __init__(self, channels=1, kernel_size=2, sigma=0.5):
        super(Gaussfilter, self).__init__()
        self.channels = channels
        self.k_size = kernel_size
        kernel = np.zeros((kernel_size, kernel_size))

        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def __call__(self, x):  # 输入的X应该维度增加过.unsqueeze(0).unsqueeze(0)
        # x = x.unsqueeze(0).unsqueeze(0)
        x = F.conv2d(x, self.weight, stride=1,  padding=1, groups=self.channels)
        return x

import cv2
gauss = Gaussfilter()

img = cv2.imread('./img_tst/test001.png', 0)
raw_n = np.random.normal(0, 15, img.shape).astype(np.float32) / 255.
imgs = np.clip(img/255. + raw_n, a_min=0., a_max=1.)
# imgs = (imgs * 255).astype(np.uint8)

res = gauss(torch.from_numpy(imgs).unsqueeze(0).unsqueeze(0).float())

