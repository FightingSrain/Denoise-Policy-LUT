# 导入opencv和numpy库
import cv2
import numpy as np
import matplotlib.pyplot as plt
# 读取原始图片
img = cv2.imread('D://Dataset/Kodak24/kodim01.png')

# 定义双边网格的参数
sigma_s = 16  # 空间域标准差
sigma_r = 0.1  # 亮度域标准差
grid_size = int(1 + sigma_s * 4)  # 网格大小 65

# 将图片转换为浮点数类型，并归一化到[0,1]区间
img = img.astype(np.float32) / 255.0

# 获取图片的高度和宽度
height, width = img.shape[:2]

# 计算空间域和亮度域的采样步长
step_s = max(1, round(sigma_s)) # 16
step_r = max(1 / 255.0, sigma_r) # 0.1


# 计算空间域和亮度域的采样个数
num_s = int((width - 1) / step_s + 1) # 32
num_r = int(1 / step_r + 1) # 11
print(num_s)
print(num_r)


# 创建一个三维数组，用来存储双边网格中每个单元的亮度值之和和像素个数之和
grid_data_sum = np.zeros((num_s, num_s, num_r))  # 亮度值之和
grid_data_cnt = np.zeros((num_s, num_s, num_r))  # 像素个数之和

# 遍历每个像素，将其映射到对应的网格单元，并累加亮度值和像素个数
for y in range(height):
    for x in range(width):
        # 获取当前像素的灰度值（取平均）
        value = img[y, x].mean()

        # 计算当前像素在空间域和亮度域上的索引（取整）
        index_x = int(x / step_s)
        index_y = int(y / step_s)
        index_z = int(value / step_r)

        # 将当前像素的灰度值累加到对应的网格单元中，并增加该单元的像素个数
        grid_data_sum[index_y, index_x, index_z] += value
        grid_data_cnt[index_y, index_x, index_z] += 1
# print(grid_data_sum)
# print(grid_data_cnt)
# for i in range(11):
#     fig = plt.figure()
#     fig.add_subplot(1, 2, 1)
#     plt.imshow(grid_data_sum[:, :, i])
#     fig.add_subplot(1, 2, 2)
#     plt.imshow(grid_data_cnt[:, :, i])
#     plt.show()
print("+++++++++++")
print(grid_data_sum / (grid_data_cnt + 0.000001))
print((grid_data_sum / (grid_data_cnt + 0.000001)).shape)
print("IIIIIIIIIIIIIIIIIIII")
# 对每个网格单元进行高斯模糊处理，得到一个平滑后的三维网格（使用opencv自带的高斯模糊函数）
grid_data_blur = cv2.GaussianBlur(grid_data_sum / (grid_data_cnt + 0.000001), (grid_size, grid_size), sigmaX=sigma_s)

# 创建一个二维数组，用来存储去噪后的图像数据（初始化为全零）
denoised_img_data = np.zeros((height, width))

# 遍历每个像素，根据其所属的网格单元，插值得到去噪后的灰度值，并赋值给去噪后图像数据中对应位置（使用线性插值）
for y in range(height):
    for x in range(width):
        # 获取当前像素在空间域上的索引（取整）
        index_x_0 = int(x / step_s)
        index_y_0 = int(y / step_s)

        # 获取当前像素在空间域上相邻的网格单元的索引（取整）
        index_x_1 = min(index_x_0 + 1, num_s - 1)
        index_y_1 = min(index_y_0 + 1, num_s - 1)

        # 获取当前像素在亮度域上的索引（取整）
        value = img[y, x].mean()
        index_z = int(value / step_r)

        # 获取当前像素在空间域和亮度域上的小数部分（用于插值）
        dx = (x - index_x_0 * step_s) / step_s
        dy = (y - index_y_0 * step_s) / step_s
        dz = (value - index_z * step_r) / step_r

        # 对当前像素所属的网格单元及其相邻的网格单元进行插值，得到去噪后的灰度值
        denoised_value = \
            grid_data_blur[index_y_0, index_x_0, index_z] * (1 - dx) * (1 - dy) * (1 - dz) + \
            grid_data_blur[index_y_0, index_x_1, index_z] * dx * (1 - dy) * (1 - dz) + \
            grid_data_blur[index_y_1, index_x_0, index_z] * (1 - dx) * dy * (1 - dz) + \
            grid_data_blur[index_y_1, index_x_1, index_z] * dx * dy * (1 - dz)

        # 将去噪后的灰度值赋值给去噪后图像数据中对应位置
        denoised_img_data[y, x] = denoised_value
# for c in range(3):
#     for y in range(height):
#         for x in range(width):
#             value_old = img[y, x].mean()
#             value_new = denoised_img_data[y, x] / 255.0
#             denoised_img_data = np.clip(denoised_img_data * 255.0, 0.0, 255.0).astype(np.uint8)
#             denoised_img_color = np.zeros((height, width, 3), dtype=np.uint8)
#             color_value = img[y, x, c] + (value_new - value_old)
#             denoised_img_color[y, x, c] = np.clip(color_value * 255.0, 0.0, 255.0)

# print(denoised_img_data.shape)
# print(denoised_img_data)
# cv2.imshow("Original Image", img)
cv2.imshow("Denoised Color Image",
           (np.clip(denoised_img_data, a_min=0., a_max=1.)*255).astype(np.uint8))
cv2.waitKey(0)