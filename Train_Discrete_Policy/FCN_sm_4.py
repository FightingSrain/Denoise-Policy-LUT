
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import copy
import torch.optim as optim
torch.manual_seed(1234)

class PPO(nn.Module):
    def __init__(self, Action_N):
        super(PPO, self).__init__()
        self.action_n = Action_N

        self.conv1 = nn.Conv2d(1, 64, [2, 2], stride=1, padding=0, dilation=2)
        self.conv2 = nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1)
        self.conv3 = nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1)
        self.conv4 = nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1)

        self.conv4p = nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1)
        self.conv5p = nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1)
        self.conv6p = nn.Conv2d(64, self.action_n, 1, stride=1, padding=0, dilation=1)

        self.conv4v = nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1)
        self.conv5v = nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1)
        self.conv6v = nn.Conv2d(64, 1, 1, stride=1, padding=0, dilation=1)

        # Init weights
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.lower().find('conv') != -1:
                nn.init.orthogonal(m.weight)  # 正交初始化
                # nn.init.kaiming_normal(m.weight)  # He初始化
                if m.bias is not None:
                    nn.init.constant(m.bias, 0)
            elif classname.find('bn') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def pi_and_v(self, x):
        B, _, H, W = x.size()
        x_in = x.reshape(B, 1, H, W)

        x1 = self.conv1(x_in)
        x1 = F.relu(x1)
        x2 = self.conv2(x1)
        x2 = F.relu(x2)
        x3 = self.conv3(x2)
        x3 = F.relu(x3)
        x4 = self.conv4(x3 + x1)
        x4 = F.relu(x4)


        p1 = self.conv4p(x4)
        p1 = F.relu(p1)
        p2 = self.conv5p(p1)
        p2 = F.relu(p2)
        # policy = F.softmax(self.conv6p(F.relu(p)), dim=1)
        policy = self.conv6p(p2)

        v1 = self.conv4v(x4)
        v1 = F.relu(v1)
        v2 = self.conv5v(v1)
        v2 = F.relu(v2)
        value = self.conv6v(v2)

        return policy, value

    def forward(self, x):
        x1 = copy.deepcopy(x)
        x1[:, 0:1, :, :] = torch.rot90(x1[:, 0:1, :, :], 1, [2, 3])
        policy1, value1 = self.pi_and_v(F.pad(x1, (0, 2, 0, 2), mode='reflect'))
        policy1 = torch.rot90(policy1, 3, [2, 3])
        value1 = torch.rot90(value1, 3, [2, 3])

        x2 = copy.deepcopy(x)
        x2[:, 0:1, :, :] = torch.rot90(x2[:, 0:1, :, :], 2, [2, 3])
        policy2, value2 = self.pi_and_v(F.pad(x2, (0, 2, 0, 2), mode='reflect'))
        policy2 = torch.rot90(policy2, 2, [2, 3])
        value2 = torch.rot90(value2, 2, [2, 3])

        x3 = copy.deepcopy(x)
        x3[:, 0:1, :, :] = torch.rot90(x3[:, 0:1, :, :], 3, [2, 3])
        policy3, value3 = self.pi_and_v(F.pad(x3, (0, 2, 0, 2), mode='reflect'))
        policy3 = torch.rot90(policy3, 1, [2, 3])
        value3 = torch.rot90(value3, 1, [2, 3])

        x4 = copy.deepcopy(x)
        policy4, value4 = self.pi_and_v(F.pad(x4, (0, 2, 0, 2), mode='reflect'))

        policy = F.softmax((policy1 + policy2 + policy3 + policy4) / 4., dim=1)
        # policy = (policy1 + policy2 + policy3 + policy4) / 4.
        value = (value1 + value2 + value3 + value4) / 4.
        return policy, value


# fcn = PPO(10)
# x = torch.randn(1, 65, 11, 11)
# policy, value, h_t = fcn.pi_and_v(x)
# print(policy.shape)
# print(value.shape)