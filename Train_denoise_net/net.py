
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import copy
import torch.optim as optim
torch.manual_seed(1234)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.action_n = Action_N
        kernel_size = 5
        kernel = torch.randn((64, 1, kernel_size, kernel_size))

        self.weight = nn.Parameter(data=kernel, requires_grad=True)
        self.weight[:, :, 0:2, :].detach().fill_(0.)
        self.weight[:, :, :, 0:2].detach().fill_(0.)
        self.weight[:, :, 3:4, :].detach().fill_(0.)
        self.weight[:, :, :, 3:4].detach().fill_(0.)
        bias = torch.zeros((64))
        self.bias = nn.Parameter(data=bias, requires_grad=True)

        self.conv2 = nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1)
        self.conv3 = nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1)
        self.conv4 = nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1)

        self.conv4p = nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1)
        self.conv5p = nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1)
        self.conv6p = nn.Conv2d(64, 1, 1, stride=1, padding=0, dilation=1)

        self.conv4v = nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1)
        self.conv5v = nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1)
        self.conv6v = nn.Conv2d(64, 1, 1, stride=1, padding=0, dilation=1)

        # Init weights
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.lower().find('conv') != -1:
                nn.init.kaiming_normal(m.weight)
                if m.bias is not None:
                    nn.init.constant(m.bias, 0)
            elif classname.find('bn') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def pi_and_v(self, x):
        B, _, H, W = x.size()
        x_in = x[:, 0:1, :, :].reshape(B * 1, 1, H, W)

        x = F.conv2d(x_in, self.weight, stride=1, padding=2, padding_model='reflect', groups=1, bias=self.bias)
        x = self.conv2(F.relu(x))
        x = self.conv3(F.relu(x))
        x = self.conv4(F.relu(x))
        p = self.conv4p(F.relu(x))
        p = self.conv5p(F.relu(p))
        res = F.sigmoid(self.conv6p(p), dim=1)

        return res

    def forward(self, x):
        x1 = copy.deepcopy(x)
        x1[:, 0:1, :, :] = torch.rot90(x1[:, 0:1, :, :], 1, [2, 3])
        res1 = self.pi_and_v(x1)
        res1 = torch.rot90(res1, 3, [2, 3])

        x2 = copy.deepcopy(x)
        x2[:, 0:1, :, :] = torch.rot90(x2[:, 0:1, :, :], 2, [2, 3])
        res2 = self.pi_and_v(x2)
        res2 = torch.rot90(res2, 2, [2, 3])

        x3 = copy.deepcopy(x)
        x3[:, 0:1, :, :] = torch.rot90(x3[:, 0:1, :, :], 3, [2, 3])
        res3 = self.pi_and_v(x3)
        res3 = torch.rot90(res3, 1, [2, 3])

        x4 = copy.deepcopy(x)
        res4 = self.pi_and_v(x4)
        res = (res1 + res2 + res3 + res4) / 4.
        return res