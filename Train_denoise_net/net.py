
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

        self.conv1 = nn.Conv2d(3, 64, [2, 2], stride=1, padding=0, dilation=2)

        self.conv2 = nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1)
        self.conv3 = nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1)
        self.conv4 = nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1)

        self.conv5 = nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1)
        self.conv6 = nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1)
        self.conv7 = nn.Conv2d(64, 3, 1, stride=1, padding=0, dilation=1)

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
        x_in = x.reshape(B, 3, H, W)
        x = self.conv1(x_in)
        x = self.conv2(F.relu(x))
        x = self.conv3(F.relu(x))
        x = self.conv4(F.relu(x))
        x = self.conv5(F.relu(x))
        x = self.conv6(F.relu(x))
        res = self.conv7(F.relu(x))

        return res

    def forward(self, x):
        x1 = copy.deepcopy(x)
        x1 = torch.rot90(x1, 1, [2, 3])
        x1 = F.pad(x1, (0, 2, 0, 2), mode='reflect')
        res1 = self.pi_and_v(x1)
        res1 = torch.rot90(res1, 3, [2, 3])

        x2 = copy.deepcopy(x)
        x2 = torch.rot90(x2, 2, [2, 3])
        x2 = F.pad(x2, (0, 2, 0, 2), mode='reflect')
        res2 = self.pi_and_v(x2)
        res2 = torch.rot90(res2, 2, [2, 3])

        x3 = copy.deepcopy(x)
        x3 = torch.rot90(x3, 3, [2, 3])
        x3 = F.pad(x3, (0, 2, 0, 2), mode='reflect')
        res3 = self.pi_and_v(x3)
        res3 = torch.rot90(res3, 1, [2, 3])

        x4 = copy.deepcopy(x)
        x4 = F.pad(x4, (0, 2, 0, 2), mode='reflect')
        res4 = self.pi_and_v(x4)
        # print(res1.size())
        # print(res2.size())
        # print(res3.size())
        # print(res4.size())
        # print("---------")
        res = torch.tanh((res1 + res2 + res3 + res4) / 4.) + x
        return res