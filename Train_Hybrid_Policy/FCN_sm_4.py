
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

        self.conv4pd = nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1)
        self.conv5pd = nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1)
        self.conv6pd = nn.Conv2d(64, self.action_n, 1, stride=1, padding=0, dilation=1)

        self.conv4pc = nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1)
        self.conv5pc = nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1)
        self.mean = nn.Conv2d(64, self.action_n, 1, stride=1, padding=0, dilation=1)
        self.logstd = nn.Parameter(torch.zeros(1, self.action_n), requires_grad=True)

        self.conv4v = nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1)
        self.conv5v = nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1)
        self.conv6v = nn.Conv2d(64, 1, 1, stride=1, padding=0, dilation=1)
        kernel = torch.ones((1, 1, 33, 33))
        self.weight = nn.Parameter(data=kernel, requires_grad=True)
        self.bias = nn.Parameter(data=torch.zeros(1), requires_grad=False)
        # self.convR = nn.Conv2d(1, 1, self.weight, stride=1, padding=16, bias=False)

        # Init weights
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.lower().find('conv') != -1:
                # nn.init.orthogonal(m.weight)  # 正交初始化
                nn.init.kaiming_normal(m.weight)  # He初始化
                if m.bias is not None:
                    nn.init.constant(m.bias, 0)
            elif classname.find('bn') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def parse_p(self, u_out):
        p = torch.mean(u_out.view(u_out.shape[0], u_out.shape[1], -1), dim=2)
        return p
    def conv_smooth(self, x):
        x = F.conv2d(x, self.weight, self.bias, stride=1, padding=16)
        return x

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


        pd1 = self.conv4pd(x4)
        pd1 = F.relu(pd1)
        pd2 = self.conv5pd(pd1)
        pd2 = F.relu(pd2)
        Dpolicy = self.conv6pd(pd2)

        pc1 = self.conv4pc(x4)
        pc1 = F.relu(pc1)
        pc2 = self.conv5pc(pc1)
        pc2 = F.relu(pc2)
        mean = self.parse_p(self.mean(pc2))
        logstd = self.logstd.expand([B, self.action_n])

        v1 = self.conv4v(x4)
        v1 = F.relu(v1)
        v2 = self.conv5v(v1)
        v2 = F.relu(v2)
        value = self.conv6v(v2)

        return Dpolicy, mean, logstd, value

    def ensemble_pi_and_v(self, x, rot1=0, rot2=0, pad=2):
        if rot1 == 0 and rot2 == 0:  # no rotation
            policy, mean, logstd, value = self.pi_and_v(F.pad(x, (0, pad, 0, pad), mode='reflect'))
        elif rot1 == -1 and rot2 == -1:  # no rotation and pad = 1
            policy, mean, logstd, value = self.pi_and_v(F.pad(x, (pad, pad, pad, pad), mode='reflect'))
        else: # rotation
            x = torch.rot90(x, rot1, [2, 3])
            policy, mean, logstd, value = self.pi_and_v(F.pad(x, (0, pad, 0, pad), mode='reflect'))
            policy = torch.rot90(policy, rot2, [2, 3])
            value = torch.rot90(value, rot2, [2, 3])

        return policy, mean, logstd, value

    def forward(self, x):
        policy1, mean1, logstd1, value1 = self.ensemble_pi_and_v(x, rot1=1, rot2=3, pad=2)
        policy2, mean2, logstd2, value2 = self.ensemble_pi_and_v(x, rot1=2, rot2=2, pad=2)
        policy3, mean3, logstd3, value3 = self.ensemble_pi_and_v(x, rot1=3, rot2=1, pad=2)
        policy4, mean4, logstd4, value4 = self.ensemble_pi_and_v(x, rot1=0, rot2=0, pad=2)
        # policy5, mean5, logstd5, value5 = self.ensemble_pi_and_v(x, rot1=-1, rot2=-1, pad=1)

        policy = F.softmax((policy1 + policy2 + policy3 + policy4) / 4., dim=1)
        value = (value1 + value2 + value3 + value4) / 4.
        mean = (mean1 + mean2 + mean3 + mean4) / 4.
        logstd = (logstd1 + logstd2 + logstd3 + logstd4) / 4.
        return policy, mean, logstd, value


# fcn = PPO(10)
# x = torch.randn(1, 65, 11, 11)
# policy, value, h_t = fcn.pi_and_v(x)
# print(policy.shape)
# print(value.shape)