# 导入必要的库
import torch
import torch.nn as nn
import torch.nn.functional as F
# import pytorch_lightning as pl


# 定义双边网格层
class BilateralGridLayer(nn.Module):
    def __init__(self, grid_size=16, sigma_spatial=0.1, sigma_color=0.1):
        super(BilateralGridLayer, self).__init__()
        self.grid_size = grid_size  # 双边网格的大小
        self.sigma_spatial = sigma_spatial  # 空间距离的标准差
        self.sigma_color = sigma_color  # 颜色距离的标准差

    def forward(self, x):
        # x: 输入图像，形状为 (batch_size, 3, height, width)
        batch_size, _, height, width = x.shape

        # 计算空间坐标和颜色坐标
        grid_x = torch.linspace(0.0, 1.0, self.grid_size).to(x.device)  # 归一化到 [0.0, 1.0]
        grid_y = torch.linspace(0.0, 1.0, self.grid_size).to(x.device)
        grid_z = torch.linspace(0.0, 1.0, self.grid_size).to(x.device)

        coord_x = torch.linspace(0.0, 1.0, width).view(1, 1, 1, width).expand(batch_size, -1, height, -1).to(x.device)
        coord_y = torch.linspace(0.0, 1.0, height).view(1, 1, height, 1).expand(batch_size, -1,-1, width).to(x.device)

        coord_z = x.permute(0, 2, 3, 1)  # 颜色坐标为 RGB 值

        print(coord_x.shape)
        print(coord_y.shape)
        print(coord_z.shape)
        print(coord_x.unsqueeze(-2).shape)
        print(grid_x.unsqueeze(-2).shape)
        # 计算空间距离和颜色距离
        # 形状为(batch_size, height, width, grid_size)
        # dist_x = (coord_x.unsqueeze() - grid_x.view(-2)) ** 2 / (2 * self.sigma_spatial ** 2)
        dist_x = (coord_x.unsqueeze(-2) -
                  grid_x.unsqueeze(-2)) ** 2 / (2 * self.sigma_spatial ** 2)  # 形状为 (batch_size,
        # 形状为 (batch_size, height, width, grid_size)
        dist_y = (coord_y.unsqueeze(-2) -
                  grid_y.unsqueeze(-2)) ** 2 / (2 * self.sigma_spatial ** 2)
        # 形状为 (batch_size, height, width, grid_size)
        dist_z = (coord_z.unsqueeze(-3) -
                  grid_z.unsqueeze(-3)) ** 2 / (2 * self.sigma_color ** 2)

        weight_x = torch.exp(-dist_x)  # 形状为 (batch_size, height, width, grid_size)
        weight_y = torch.exp(-dist_y) # 形状为 (batch_size, height, width, grid_size)
        weight_z = torch.exp(-dist_z) # 形状为 (batch_size, height, width, grid_size)

        norm_x = weight_x.sum(dim=-1, keepdim=True)
        norm_y = weight_y.sum(dim=-1, keepdim=True)
        norm_z = weight_z.sum(dim=-2, keepdim=True)

        weight_x = weight_x / norm_x
        weight_y = weight_y / norm_y
        weight_z = weight_z / norm_z

        grid_value = torch.einsum("bhwg, bhw->bg", weight_x, x[:, 0]) + \
                    torch.einsum("bhwg, bhw->bg", weight_y, x[:,1]) + \
                    torch.einsum("bhwg, bhw->bg", weight_z, x[:, 2])

        return grid_value.unsqueeze(1), weight_x.unsqueeze(1), weight_y.unsqueeze(1), weight_z.unsqueeze(1)

class BilateralGridDenoisingNet(nn.Module):
    def __init__(self, ):
        super(BilateralGridDenoisingNet, self).__init__()
        self.grid_layer = BilateralGridLayer()  # 双边网格层
        self.conv_layer_1 = nn.Conv2d(4, 16, kernel_size=3, padding=1) # 卷积层
        self.conv_layer_2 = nn.Conv2d(16, 32, kernel_size=3, padding=1) # 卷积层
        self.conv_layer_3 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # 卷积层
        self.conv_layer_4 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # 卷积层
        self.conv_layer_5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv_layer_6 = nn.Conv2d(256 + 64 + 32 + 16 + 4, 3, kernel_size=1)  # 卷积层

    def forward(self, x):
        # x: 输入图像，形状为 (batch_size, 3, height, width)
        batch_size, _, height, width = x.shape
        print(x.shape)
        print("********")
        # 计算双边网格的值和权重
        grid_value, weight_x, weight_y, weight_z = self.grid_layer(x)

        # 将双边网格的值和权重拼接起来
        grid_input = torch.cat([grid_value, weight_x, weight_y, weight_z], dim=1)

        conv_output_1 = F.relu(self.conv_layer_1(grid_input))
        conv_output_2 = F.relu(self.conv_layer_2(conv_output_1))
        conv_output_3 = F.relu(self.conv_layer_3(conv_output_2))
        conv_output_4 = F.relu(self.conv_layer_4(conv_output_3))
        conv_output_5 = F.relu(self.conv_layer_5(conv_output_4))

        concat_output = torch.cat([conv_output_5, conv_output_4, conv_output_3, conv_output_2, conv_output_1, grid_input], dim=1)

        denoised_image = self.conv_layer_6(concat_output)

        return denoised_image



# test
if __name__ == "__main__":
    model = BilateralGridDenoisingNet()
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    print(y.shape)

