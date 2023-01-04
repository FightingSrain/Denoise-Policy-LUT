import copy

import torch.nn.functional as F
import torch
import torch.nn as nn
import math
import torch.optim as optim
torch.manual_seed(1234)

class PPO(nn.Module):
    def __init__(self, Action_N):
        super(PPO, self).__init__()
        self.action_n = Action_N
        kernel_size = 3
        kernel = torch.randn((64, 1, kernel_size, kernel_size))
        self.weight = nn.Parameter(data=kernel, requires_grad=True)
        self.weight[:, :, 0:1, :].detach()
        self.weight[:, :, :, 0:1].detach()
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
                nn.init.kaiming_normal(m.weight)
                if m.bias is not None:
                    nn.init.constant(m.bias, 0)
            elif classname.find('bn') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def pi_and_v(self, x):
        B, _, H, W = x.size()
        h_t = x[:, -64:, :, :]
        x_in = x[:, 0:1, :, :].reshape(B * 1, 1, H, W)

        # x_in = torch.rot90(x_in, 1, [2,3])
        x = F.conv2d(x_in, self.weight, stride=1, padding=1, groups=1)
        x = self.conv2(F.relu(x))
        x = self.conv3(F.relu(x))
        x = self.conv4(F.relu(x))

        p = self.conv4p(F.relu(x))
        p = self.conv5p(F.relu(p))
        policy = F.softmax(self.conv6p(p), dim=1)

        v = self.conv4v(F.relu(x))
        v = self.conv5v(F.relu(v))
        value = self.conv6v(F.relu(v))

        return policy, value, h_t

    def forward(self, x):
        x1 = copy.deepcopy(x)
        x1[:, 0:1, :, :] = torch.rot90(x1[:, 0:1, :, :], 1, [2, 3])
        policy1, value1, h_t1 = self.pi_and_v(x1)
        policy1 = torch.rot90(policy1, 3, [2, 3])
        value1 = torch.rot90(value1, 3, [2, 3])
        h_t1 = torch.rot90(h_t1, 3, [2, 3])

        x2 = copy.deepcopy(x)
        x2[:, 0:1, :, :] = torch.rot90(x2[:, 0:1, :, :], 2, [2, 3])
        policy2, value2, h_t2 = self.pi_and_v(x2)
        policy2 = torch.rot90(policy2, 2, [2, 3])
        value2 = torch.rot90(value2, 2, [2, 3])
        h_t2 = torch.rot90(h_t2, 2, [2, 3])

        x3 = copy.deepcopy(x)
        x3[:, 0:1, :, :] = torch.rot90(x3[:, 0:1, :, :], 3, [2, 3])
        policy3, value3, h_t3 = self.pi_and_v(x3)
        policy3 = torch.rot90(policy3, 1, [2, 3])
        value3 = torch.rot90(value3, 1, [2, 3])
        h_t3 = torch.rot90(h_t3, 1, [2, 3])

        x4 = copy.deepcopy(x)
        # x4[:, 0:1, :, :] = x4[:, 0:1, :, :]
        policy4, value4, h_t4 = self.pi_and_v(x4)

        policy = (policy1 + policy2 + policy3 + policy4) / 4.
        value = (value1 + value2 + value3 + value4) / 4.
        h_t = (h_t1 + h_t2 + h_t3 + h_t4) / 4.

        return policy, value, h_t




# fcn = PPO(10)
#
# x = torch.randn(1, 65, 11, 11)
# policy, value, h_t = fcn.pi_and_v(x)
# print(policy.shape)
# print(value.shape)