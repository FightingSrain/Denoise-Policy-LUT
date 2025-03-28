
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import copy
import torch.optim as optim
torch.manual_seed(1234)
class Conv(nn.Module):
    """ 2D convolution w/ MSRA init. """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=bias)
        nn.init.kaiming_normal_(self.conv.weight)
        if bias:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.conv(x)
class DenseConv(nn.Module):
    """ Dense connected Conv. with activation. """

    def __init__(self, in_nf, nf=64):
        super(DenseConv, self).__init__()
        self.act = nn.ReLU()
        self.conv1 = Conv(in_nf, nf, 1)

    def forward(self, x):
        feat = self.act(self.conv1(x))
        out = torch.cat([x, feat], dim=1)
        return out
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.n_act = 27
        self.conv1a = nn.Conv2d(3, 64, 2, stride=1, padding=0, dilation=1)
        self.conv1b = nn.Conv2d(3, 64, 2, stride=1, padding=0, dilation=2)
        self.conv1c = nn.Conv2d(3, 64, (1, 4), stride=1, padding=0, dilation=1)

        self.conv2 = nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1)
        self.conv3 = nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1)
        self.conv4 = nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1)

        self.conv5 = nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1)
        self.conv6 = nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1)
        self.conv7 = nn.Conv2d(64, self.n_act, 1, stride=1, padding=0, dilation=1)

        # nf = 32
        # self.conv2 = DenseConv(nf, nf)
        # self.conv3 = DenseConv(nf + nf * 1, nf)
        # self.conv4 = DenseConv(nf + nf * 2, nf)
        # self.conv5 = DenseConv(nf + nf * 3, nf)
        # self.conv6 = Conv(nf * 5, 1, 1)
        self.act = nn.ReLU()

        self.mods = ['a', 'b', 'c']
        # Init weights
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.lower().find('conv') != -1:
                if m == self.conv1c:
                    continue
                nn.init.kaiming_normal(m.weight)
                if m.bias is not None:
                    nn.init.constant(m.bias, 0)
            elif classname.find('bn') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def cal(self, x):
        # x = self.act(x)
        # x = self.conv2(x)
        # x = self.conv3(x)
        # x = self.conv4(x)
        # x = self.conv5(x)
        # res = self.conv6(x)

        xs = self.conv2(F.relu(x))
        xs = self.conv3(F.relu(xs))
        xs = self.conv4(F.relu(xs))
        xs = self.conv5(F.relu(xs))
        xs = self.conv6(F.relu(xs))
        res = (self.conv7(F.relu(xs)))
        return res
    def transfer_lut(self, x, mod):
        if mod == 'a':
            res = self.cal(self.conv1a(x))
            return res
        elif mod == 'b':
            res = self.cal(self.conv1b(x))
            return res
        elif mod == 'c':
            self.K = 3
            self.S = 1
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
        x = self.cal(self.conv1c(x))  # B*C*L,K,K
        x = x.squeeze(1)
        x = x.reshape(B, C, (H - self.P) * (W - self.P), -1)  # B,C,K*K,L
        x = x.permute((0, 1, 3, 2))  # B,C,K*K,L
        x = x.reshape(B, -1, (H - self.P) * (W - self.P))  # B,C*K*K,L
        x = F.fold(x, ((H - self.P) * self.S, (W - self.P) * self.S),
                   self.S, stride=self.S)
        return x

    def pi_and_v(self, x, mod):
        if mod == 'a':
            B, C, H, W = x.size()
            # x_in = x.reshape(B*C, 1, H, W)
            x_in = x.reshape(B, C, H, W)
            res = self.cal(self.conv1a(x_in))
            res = res.reshape(B, self.n_act, H-1, W-1)
            return res
        elif mod == 'b':
            B, C, H, W = x.size()
            x_in = x.reshape(B, C, H, W)
            res = self.cal(self.conv1b(x_in))
            res = res.reshape(B, self.n_act, H-2, W-2)
            return res
        elif mod == 'c':
            self.K = 3
            self.S = 1
        self.P = self.K - 1
        B, C, H, W = x.shape
        x = F.unfold(x, self.K)  # B,C*K*K,L
        x = x.view(B, C, self.K * self.K, (H - self.P) * (W - self.P))  # B,C,K*K,L
        x = x.permute((0, 3, 1, 2))  # B,L,C,K*K
        x = x.reshape(B * (H - self.P) * (W - self.P),
                      C, self.K, self.K)  # B*L,C,K,K
        # x = x.unsqueeze(1)  # B*L,C,K,K

        if mod == 'c':
            x = torch.cat([x[:, :, 0, 0], x[:, :, 1, 1],
                           x[:, :, 1, 2], x[:, :, 2, 1]], dim=1)

            x = x.unsqueeze(1).unsqueeze(1)
        x = self.cal(self.conv1c(x))  # B*C*L,K,K
        x = x.squeeze(1)
        x = x.reshape(B, 27, (H - self.P) * (W - self.P), -1)  # B,27,K*K,L
        x = x.permute((0, 1, 3, 2))  # B,C,K*K,L
        x = x.reshape(B, -1, (H - self.P) * (W - self.P))  # B,C*K*K,L
        x = F.fold(x, ((H - self.P) * self.S, (W - self.P) * self.S),
                   self.S, stride=self.S)
        return x

    @staticmethod
    def round_func(input):
        # Backward Pass Differentiable Approximation (BPDA)
        # This is equivalent to replacing round function (non-differentiable)
        # with an identity function (differentiable) only when backward,
        forward_value = torch.round(input)
        out = input.clone()
        out.data = forward_value.data
        return out

    def ensemble_pi_and_v(self, x, mod, residuels, rot1=0, rot2=0, pad=2):
        if rot1 == 0 and rot2 == 0:  # no rotation
            residuel = self.pi_and_v(F.pad(x, (0, pad, 0, pad), mode='replicate'), mod)
        else: # rotation
            x = torch.rot90(x, rot1, [2, 3])
            residuel = self.pi_and_v(F.pad(x, (0, pad, 0, pad), mode='replicate'), mod)
            residuel = torch.rot90(residuel, rot2, [2, 3])
        if mod == 'a':
            residuels += (residuel * 0.4)
        elif mod == 'b':
            residuels += (residuel * 0.3)
        elif mod == 'c':
            residuels += (residuel * 0.3)
            # residuels /= 3.
            self.round_func(residuels)
        return residuels

    def forward(self, x):
        residuel1, residuel2, residuel3, residuel4 = 0, 0, 0, 0
        for mod in self.mods:
            if mod == 'a':
                residuel1 = self.ensemble_pi_and_v(x, mod, residuel1, rot1=1, rot2=3, pad=1)
            else:
                residuel1 = self.ensemble_pi_and_v(x, mod, residuel1, rot1=1, rot2=3, pad=2)

        for mod in self.mods:
            if mod == 'a':
                residuel2 = self.ensemble_pi_and_v(x, mod, residuel2, rot1=2, rot2=2, pad=1)
            else:
                residuel2 = self.ensemble_pi_and_v(x, mod, residuel2, rot1=2, rot2=2, pad=2)


        for mod in self.mods:
            if mod == 'a':
                residuel3 = self.ensemble_pi_and_v(x, mod, residuel3, rot1=3, rot2=1, pad=1)
            else:
                residuel3 = self.ensemble_pi_and_v(x, mod, residuel3, rot1=3, rot2=1, pad=2)

        for mod in self.mods:
            if mod == 'a':
                residuel4 = self.ensemble_pi_and_v(x, mod, residuel4, rot1=0, rot2=0, pad=1)
            else:
                residuel4 = self.ensemble_pi_and_v(x, mod, residuel4, rot1=0, rot2=0, pad=2)

        # res = (torch.clamp(residuel1, -1, 1) * 127 + torch.clamp(residuel2, -1, 1) * 127)
        # res += (torch.clamp(residuel3, -1, 1) * 127 + torch.clamp(residuel4, -1, 1) * 127)
        # res /= 4.
        # res = torch.clamp(res + 127, 0, 255)
        # res /= 255.0
        res = torch.tanh((residuel1 + residuel2 + residuel3 + residuel4) / 4.0) + x
        # res = ((residuel1 + residuel2 + residuel3 + residuel4) / 4.0) * 0.5 + x
        return res

# test
net = Net().cuda()
x = torch.randn(1, 3, 64, 64).cuda()
y = net(x)
print(y.shape)
