
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
        # self.conv1 = nn.Conv2d(1, 64, [2, 2], stride=1, padding=2, dilation=4)
        self.conv1 = nn.Conv2d(1, 64, [2, 2], stride=1, padding=1, dilation=2)
        self.conv2 = nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1)
        self.conv3 = nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1)
        self.conv4 = nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1)

        self.conv4p = nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1)
        self.conv5p = nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1)
        self.conv6p = nn.Conv2d(64, self.action_n, 1, stride=1, padding=0, dilation=1)

        self.conv4v = nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1)
        self.conv5v = nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1)
        self.conv6v = nn.Conv2d(64, 1, 1, stride=1, padding=0, dilation=1)

        self.conv7_Wz = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv7_Uz = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv7_Wr = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv7_Ur = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv7_W = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv7_U = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)


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

        # self.conv1.apply(self.weight_init)
        # self.conv2.apply(self.weight_init)
        # self.conv3.apply(self.weight_init)
        #
        # self.conv4p.apply(self.weight_init)
        # self.conv5p.apply(self.weight_init)
        # self.conv6p.apply(self.weight_init)
        #
        # self.conv4v.apply(self.weight_init)
        # self.conv5v.apply(self.weight_init)
        # self.conv6v.apply(self.weight_init)

    def weight_init(self, m):
        classname = m.__class__.__name__
        if classname.find("Conv2d") != -1:
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('Linear') != -1:
            m.weight.data.normal_(0, 0.01)
            m.bias.data = torch.ones(m.bias.data.size())

    def pi_and_v(self, x):
        B, _, H, W = x.size()
        # h_t = x[:, -64:, :, :]
        x_in = x[:, 0:1, :, :].reshape(B * 1, 1, H, W)

        x = self.conv1(x_in)
        x = self.conv2(F.relu(x))
        x = self.conv3(F.relu(x))
        x = self.conv4(F.relu(x))


        p = self.conv4p(F.relu(x))
        p = self.conv5p(F.relu(p))
        GRU_in = p
        ht = x[:, -64:, :, :]
        z_t = torch.sigmoid(self.conv7_Wz(GRU_in) + self.conv7_Uz(ht))
        r_t = torch.sigmoid(self.conv7_Wr(GRU_in) + self.conv7_Ur(ht))
        h_title_t = torch.tanh(self.conv7_W(GRU_in) + self.conv7_U(r_t * ht))
        h_t = (1 - z_t) * ht + z_t * h_title_t
        policy = F.softmax(self.conv6p(h_t), dim=1)

        v = self.conv4v(F.relu(x))
        v = self.conv5v(F.relu(v))
        value = self.conv6v(F.relu(v))

        return policy, value, h_t


# fcn = PPO(10)
#
# x = torch.randn(1, 65, 11, 11)
# policy, value, h_t = fcn.pi_and_v(x)
# print(policy.shape)
# print(value.shape)