import copy
import torch
import cv2
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np
# from Train_Hybrid_Policy import State_Gaussian as State

from Train_denoise_net.net3 import Net
from Train_denoise_net.config import config

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def paint_amap(acmap):
    image = np.asanyarray(acmap.squeeze(), dtype=np.uint8)
    plt.imshow(image, vmin=1, vmax=9)
    plt.colorbar()
    # plt.pause(1)
    plt.show()
    # plt.close('all')

SAMPLING_INTERVAL = 4
sigma = 25
model = Net().to(device)
model.load_state_dict(torch.load("./DenoiseNetModelMax_{}/83700_26.190159255783744_0.8766308300422899.pth".format(sigma)))

print("-----------------")
mods = ['a', 'b', 'c']

with torch.no_grad():
    model.eval()
    # 1D input
    # base = torch.arange(0, 257, 1)
    base = torch.arange(0, 257, 2 ** SAMPLING_INTERVAL)  # 0-256 像素值范围，下采样，只采样2**4=16个种类像素值
    base[-1] -= 1
    L = base.size(0)
    # [  0,  16,  32,  48,  64,  80,  96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255]

    # 2D input
    first = base.cuda().unsqueeze(1).repeat(1, L).reshape(-1)  # 256*256   0 0 0...    |1 1 1...     |...|255 255 255...
    second = base.cuda().repeat(L)  # 256*256   0 1 2 .. 255|0 1 2 ... 255|...|0 1 2 ... 255
    onebytwo = torch.stack([first, second], 1)  # [256*256, 2]

    # 3D input
    third = base.cuda().unsqueeze(1).repeat(1, L * L).reshape(-1)  # 256*256*256   0 x65536|1 x65536|...|255 x65536
    onebytwo = onebytwo.repeat(L, 1)
    onebythree = torch.cat([third.unsqueeze(1), onebytwo], 1)  # [256*256*256, 3]

    # 4D input
    fourth = base.cuda().unsqueeze(1).repeat(1, L * L * L).reshape(
        -1)  # 256*256*256*256   0 x16777216|1 x16777216|...|255 x16777216
    onebythree = onebythree.repeat(L, 1)
    onebyfourth = torch.cat([fourth.unsqueeze(1), onebythree], 1)  # [256*256*256*256, 4]

    # Rearange input: [N, 4] -> [N, C=1, H=2, W=2]
    input_tensor = onebyfourth.unsqueeze(1).unsqueeze(1).reshape(-1, 1, 2, 2).float()
    print("Input size: ", input_tensor.size())
    # -----------------------------------------------
    # Inputs = []
    for mod in mods:
        if mod == 'a':
            intputs = torch.zeros((input_tensor.size(0), 1, 2, 2))
            intputs[:, :, 0, 0] = input_tensor[:, :, 0, 0]
            intputs[:, :, 0, 1] = input_tensor[:, :, 0, 1]
            intputs[:, :, 1, 0] = input_tensor[:, :, 1, 0]
            intputs[:, :, 1, 1] = input_tensor[:, :, 1, 1]
        elif mod == 'b':
            intputs = torch.zeros((input_tensor.size(0), 1, 3, 3))
            intputs[:, :, 0, 0] = input_tensor[:, :, 0, 0]
            intputs[:, :, 0, 2] = input_tensor[:, :, 0, 1]
            intputs[:, :, 2, 0] = input_tensor[:, :, 1, 0]
            intputs[:, :, 2, 2] = input_tensor[:, :, 1, 1]
        else:
            intputs = torch.zeros((input_tensor.size(0), 1, 3, 3))
            intputs[:, :, 0, 0] = input_tensor[:, :, 0, 0]
            intputs[:, :, 1, 1] = input_tensor[:, :, 0, 1]
            intputs[:, :, 1, 2] = input_tensor[:, :, 1, 0]
            intputs[:, :, 2, 1] = input_tensor[:, :, 1, 1]
        # Inputs.append(intputs)


        NUM = 1000 # 采样>=5时，调整为10
        # Split input to not over GPU memory
        B = input_tensor.size(0) // NUM
        LUT = []
        for b in range(NUM):
            print("Processing: ", b)
            # res = 0
            # 遍历三种模式
            # for mn in range(len(mods)):
            if b == NUM-1:
                raw_x = intputs[b*B:]
            else:
                raw_x = intputs[b*B:(b+1)*B]

            ins = np.asarray(copy.deepcopy(raw_x))/255.
            print(ins.shape)
            print("******eqwe")
            # res += model(torch.from_numpy(ins).cuda()).detach().cpu().numpy()
            temp = model.transfer_lut(torch.from_numpy(ins).cuda(), mod).detach().cpu().numpy()
            print(temp.shape)
            print("#######")
            # print(res/3.)
            # print(res.shape)
            # print(np.concatenate([res.squeeze()], axis=1).shape)
            # print("********")
            # LUT += [np.concatenate([temp.squeeze().reshape(temp.shape[0], -1)], axis=1)]
            LUT += [temp]


        LUTs = np.concatenate(LUT, 0)
        print(LUTs.shape)
        print("Resulting LUT size: ", LUTs.shape)
        # np.save("./Hybrid_LUTs_15/sample_{}_LUTs_{}".format(SAMPLING_INTERVAL, config.SIGMA), LUTs)
        # np.save("./Hybrid_LUTs_save/sample_{}_LUTs_{}".format(SAMPLING_INTERVAL, config.SIGMA), LUTs)
        np.save("./DenoiseNet_LUTs_{}/sample_{}_LUTs{}_{}".format(sigma, SAMPLING_INTERVAL, mod, sigma), LUTs)
