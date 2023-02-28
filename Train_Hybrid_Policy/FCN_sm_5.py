
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import copy
import torch.optim as optim
torch.manual_seed(1234)

# class conv1c(nn.Module):
#     def __init__(self,):
#         super(conv1c, self).__init__()
#         self.K = 3
#         self.S = 1
#         self.conv = nn.Conv2d(1, 64, (1, 4),
#                               stride=1, padding=0, dilation=1, bias=True)
#         self.P = self.K - 1
#
#         nn.init.kaiming_normal_(self.conv.weight)
#         nn.init.constant_(self.conv.bias, 0)
#     def forward(self, x):
#         B, C, H, W = x.shape
#         x = F.unfold(x, self.K)  # B,C*K*K,L
#         x = x.view(B, C, self.K * self.K, (H - self.P) * (W - self.P))  # B,C,K*K,L
#         x = x.permute((0, 1, 3, 2))  # B,C,L,K*K
#         x = x.reshape(B * C * (H - self.P) * (W - self.P),
#                       self.K, self.K)  # B*C*L,K,K
#         x = x.unsqueeze(1)  # B*C*L,l,K,K
#         x = torch.cat([x[:, :, 0, 0], x[:, :, 1, 1],
#                        x[:, :, 1, 2], x[:, :, 2, 1]], dim=1)
#
#         x = x.unsqueeze(1).unsqueeze(1)
#
#         x = self.conv(x)  # B*C*L,K,K
#         x = x.squeeze(1)
#         x = x.reshape(B, C, (H - self.P) * (W - self.P), -1)  # B,C,K*K,L
#         x = x.permute((0, 1, 3, 2))  # B,C,K*K,L
#         x = x.reshape(B, -1, (H - self.P) * (W - self.P))  # B,C*K*K,L
#         x = F.fold(x, ((H - self.P) * self.S, (W - self.P) * self.S),
#                    self.S, stride=self.S)
#
#         return x

class PPO(nn.Module):
    def __init__(self, Action_N):
        super(PPO, self).__init__()
        self.action_n = Action_N

        self.conv1a = nn.Conv2d(1, 64, 2, stride=1, padding=0, dilation=1)
        self.conv1b = nn.Conv2d(1, 64, 2, stride=1, padding=0, dilation=2)
        self.conv1c = nn.Conv2d(1, 64, (1, 4), stride=1, padding=0, dilation=1)
        #------------
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
        # kernel = torch.ones((1, 1, 33, 33))
        # self.weight = nn.Parameter(data=kernel, requires_grad=True)
        # self.bias = nn.Parameter(data=torch.zeros(1), requires_grad=False)
        # self.convR = nn.Conv2d(1, 1, self.weight, stride=1, padding=16, bias=False)

        self.mods = ['a', 'b', 'c']

        # Init weights
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.lower().find('conv') != -1:
                # if m == self.conv1c:
                #     continue
                # nn.init.orthogonal(m.weight)  # 正交初始化
                nn.init.kaiming_normal(m.weight)  # He初始化
                if m.bias is not None:
                    nn.init.constant(m.bias, 0)
            elif classname.find('bn') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def parse_p(self, u_out):
        p = torch.mean(u_out.contiguous().view(u_out.shape[0], u_out.shape[1], -1), dim=2)
        return p

    def conv_smooth(self, x):
        x = F.conv2d(x, self.weight, self.bias, stride=1, padding=16)
        return x

    def cal(self, x):
        B, C, H, W = x.size()
        x1 = F.relu(x)
        x2 = self.conv2(x1)
        x2 = F.relu(x2)
        x3 = self.conv3(x2)
        x3 = F.relu(x3)
        x4 = self.conv4(x3)
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
        mean = self.mean(pc2)
        # logstd = self.logstd.expand([B, self.action_n])

        v1 = self.conv4v(x4)
        v1 = F.relu(v1)
        v2 = self.conv5v(v1)
        v2 = F.relu(v2)
        value = self.conv6v(v2)
        return Dpolicy, mean, value

    def pi_and_v(self, x, mod):
        B, C, H, W = x.size()
        x_in = x.reshape(B * C, 1, H, W)
        # print(x_in.shape)
        if mod == 'a':
            x1 = self.conv1a(x_in)
            Dpolicy, mean, value = self.cal(x1)
            logstd = self.logstd.expand([B*C, self.action_n])
            # print(Dpolicy.size())
            # print(mean.size())
            # print(logstd.size())
            # print(value.size())
            # print("*************")
            return Dpolicy.contiguous().reshape(x_in.shape[0], self.action_n, H - 1, W - 1), \
                   mean.contiguous().reshape(x_in.shape[0], self.action_n, H - 1, W - 1), \
                   logstd.contiguous().reshape(x_in.shape[0], self.action_n), \
                   value.contiguous().reshape(x_in.shape[0], 1, H - 1, W - 1)

        elif mod == 'b':
            x1 = self.conv1b(x_in)
            Dpolicy, mean, value = self.cal(x1)
            logstd = self.logstd.expand([B * C, self.action_n])
            return Dpolicy.contiguous().reshape(x_in.shape[0], self.action_n, H - 2, W - 2), \
                   mean.contiguous().reshape(x_in.shape[0], self.action_n, H - 2, W - 2), \
                   logstd.contiguous().reshape(x_in.shape[0], self.action_n), \
                   value.contiguous().reshape(x_in.shape[0], 1, H - 2, W - 2)
        else: # mod == 'c':
            self.K = 3
            self.S = 1
            # x1 = self.conv1c(x_in)

        self.P = self.K - 1
        B, C, H, W = x.shape
        x = F.unfold(x, self.K)  # B,C*K*K,L
        x = x.view(B, C, self.K * self.K, (H - self.P) * (W - self.P))  # B,C,K*K,L
        x = x.permute((0, 1, 3, 2))  # B,C,L,K*K
        x = x.reshape(B * C * (H - self.P) * (W - self.P),
                      self.K, self.K)  # B*C*L,K,K
        x = x.unsqueeze(1)  # B*C*L,l,K,K

        if mod == 'c':
            x = torch.cat([x[:, :, 0, 0], x[:, :, 1, 1],
                           x[:, :, 1, 2], x[:, :, 2, 1]], dim=1)

            x = x.unsqueeze(1).unsqueeze(1)
        # print(x.size())
        Dpolicy, mean, value = self.cal(self.conv1c(x))  # B*C*L,K,K
        Dpolicy = Dpolicy.squeeze(1)
        Dpolicy = Dpolicy.reshape(B, C, (H - self.P) * (W - self.P), -1)  # B,C,K*K,L
        Dpolicy = Dpolicy.permute((0, 1, 3, 2))  # B,C,K*K,L
        Dpolicy = Dpolicy.reshape(B, -1, (H - self.P) * (W - self.P))  # B,C*K*K,L
        Dpolicy = F.fold(Dpolicy, ((H - self.P) * self.S, (W - self.P) * self.S),
                   self.S, stride=self.S)
        mean = mean.squeeze(1)
        mean = mean.reshape(B, C, (H - self.P) * (W - self.P), -1)  # B,C,K*K,L
        mean = mean.permute((0, 1, 3, 2))  # B,C,K*K,L
        mean = mean.reshape(B, -1, (H - self.P) * (W - self.P))  # B,C*K*K,L
        mean = F.fold(mean, ((H - self.P) * self.S, (W - self.P) * self.S),
                         self.S, stride=self.S)
        value = value.squeeze(1)
        value = value.reshape(B, C, (H - self.P) * (W - self.P), -1)  # B,C,K*K,L
        value = value.permute((0, 1, 3, 2))  # B,C,K*K,L
        value = value.reshape(B, -1, (H - self.P) * (W - self.P))  # B,C*K*K,L
        value = F.fold(value, ((H - self.P) * self.S, (W - self.P) * self.S),
                      self.S, stride=self.S)

        # logstd = logstd.squeeze(1)
        # logstd = logstd.reshape(B, C, (H - self.P) * (W - self.P), -1)  # B,C,K*K,L
        # logstd = logstd.permute((0, 1, 3, 2))  # B,C,K*K,L
        # logstd = logstd.reshape(B, -1, (H - self.P) * (W - self.P))  # B,C*K*K,L
        # logstd = F.fold(logstd, ((H - self.P) * self.S, (W - self.P) * self.S),
        #                self.S, stride=self.S)
        logstd = self.logstd.expand(x_in.shape[0], self.action_n)
        # print(logstd.size())
        return Dpolicy, mean, self.parse_p(logstd), value
    @staticmethod
    def round_func(input):
        # Backward Pass Differentiable Approximation (BPDA)
        # This is equivalent to replacing round function (non-differentiable)
        # with an identity function (differentiable) only when backward,
        forward_value = torch.round(input)
        out = input.clone()
        out.data = forward_value.data
        return out

    def ensemble_pi_and_v(self, x, mod, policys, means, logstds, values, rot1=0, rot2=0, pad=2):
        if rot1 == 0 and rot2 == 0:  # no rotation
            policy, mean, logstd, value = self.pi_and_v(F.pad(x, (0, pad, 0, pad), mode='replicate'), mod)
        elif rot1 == -1 and rot2 == -1:  # no rotation and pad = 1
            policy, mean, logstd, value = self.pi_and_v(F.pad(x, (pad, pad, pad, pad), mode='replicate'), mod)
        else: # rotation
            x = torch.rot90(x, rot1, [2, 3])
            policy, mean, logstd, value = self.pi_and_v(F.pad(x, (0, pad, 0, pad), mode='replicate'), mod)

            policy = torch.rot90(policy, rot2, [2, 3])
            mean = torch.rot90(mean, rot2, [2, 3])
            # logstd = torch.rot90(logstd, rot2, [2, 3])
            value = torch.rot90(value, rot2, [2, 3])
        # print(logstds)
        # print(logstd.size())
        policys += policy
        means += mean
        logstds += logstd
        values += value

        if mod == self.mods[-1]:
            # policys /= len(self.mods)
            policys = self.round_func(policys/len(self.mods))
            # means /= len(self.mods)
            means = self.round_func(means/len(self.mods))
            # logstds /= len(self.mods)
            logstds = self.round_func(logstds/len(self.mods))
            # values /= len(self.mods)
            values = self.round_func(values/len(self.mods))
        return policys, means, logstds, values

    def forward(self, x):

        B, C, H, W = x.size()
        x_in = x.reshape(B * C, 1, H, W)

        policy1, mean1, logstd1, value1 = 0, 0, 0, 0
        policy2, mean2, logstd2, value2 = 0, 0, 0, 0
        policy3, mean3, logstd3, value3 = 0, 0, 0, 0
        policy4, mean4, logstd4, value4 = 0, 0, 0, 0
        for mod in self.mods:
            if mod == 'a':
                policy1, mean1, logstd1, value1 = self.ensemble_pi_and_v(x_in, mod, policy1 ,mean1, logstd1, value1,
                                                                     rot1=1, rot2=3, pad=1)
            else:
                policy1, mean1, logstd1, value1 = self.ensemble_pi_and_v(x_in, mod, policy1 ,mean1, logstd1, value1,
                                                                     rot1=1, rot2=3, pad=2)


        for mod in self.mods:
            if mod == 'a':
                policy2, mean2, logstd2, value2 = self.ensemble_pi_and_v(x_in, mod, policy2, mean2, logstd2, value2,
                                                                     rot1=2, rot2=2, pad=1)
            else:
                policy2, mean2, logstd2, value2 = self.ensemble_pi_and_v(x_in, mod, policy2, mean2, logstd2, value2,
                                                                     rot1=2, rot2=2, pad=2)


        for mod in self.mods:
            if mod == 'a':
                policy3, mean3, logstd3, value3 = self.ensemble_pi_and_v(x_in, mod, policy3, mean3, logstd3, value3,
                                                                     rot1=3, rot2=1, pad=1)
            else:
                policy3, mean3, logstd3, value3 = self.ensemble_pi_and_v(x_in, mod, policy3, mean3, logstd3, value3,
                                                                     rot1=3, rot2=1, pad=2)

        for mod in self.mods:
            if mod == 'a':
                policy4, mean4, logstd4, value4 = self.ensemble_pi_and_v(x_in, mod, policy4, mean4, logstd4, value4,
                                                                     rot1=0, rot2=0, pad=1)
            else:
                policy4, mean4, logstd4, value4 = self.ensemble_pi_and_v(x_in, mod, policy4, mean4, logstd4, value4,
                                                                     rot1=0, rot2=0, pad=2)



        # policy2, mean2, logstd2, value2 = self.ensemble_pi_and_v(x, rot1=2, rot2=2, pad=2)
        # policy3, mean3, logstd3, value3 = self.ensemble_pi_and_v(x, rot1=3, rot2=1, pad=2)
        # policy4, mean4, logstd4, value4 = self.ensemble_pi_and_v(x, rot1=0, rot2=0, pad=2)
        # policy5, mean5, logstd5, value5 = self.ensemble_pi_and_v(x, rot1=-1, rot2=-1, pad=1)

        policy = F.softmax((policy1 + policy2 + policy3 + policy4) / 4., dim=1)
        value = (value1 + value2 + value3 + value4) / 4.

        mean = self.parse_p((mean1 + mean2 + mean3 + mean4) / 4.)
        logstd = (logstd1 + logstd2 + logstd3 + logstd4) / 4.
        return policy, mean, logstd, value


# fcn = PPO(10)
# x = torch.randn(1, 65, 11, 11)
# policy, value, h_t = fcn.pi_and_v(x)
# print(policy.shape)
# print(value.shape)