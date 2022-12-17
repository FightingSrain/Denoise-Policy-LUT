import copy

import cv2
import matplotlib.pyplot as plt
import torch.optim as optim

from Train_Net import State as State
# from FCN import *
from Train_Net.FCN_sm import *
from Train_Net.pixelwise_a3c import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def paint_amap(acmap):
    image = np.asanyarray(acmap.squeeze(), dtype=np.uint8)
    plt.imshow(image, vmin=1, vmax=9)
    plt.colorbar()
    # plt.pause(1)
    plt.show()
    # plt.close('all')

MOVE_RANGE = 3
EPISODE_LEN = 1
MAX_EPISODE = 100000
GAMMA = 0.95
N_ACTIONS = 9
LR = 0.0001
SAMPLING_INTERVAL = 4


model = PPO(N_ACTIONS).to(device)
model.load_state_dict(torch.load("../GaussianFilterModel/GaussianModela46800_.pth"))
optimizer = optim.Adam(model.parameters(), lr=LR)
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
    # 2*2 inputs -> 3*3 inputs
    intputs = torch.zeros((input_tensor.size(0), 1, 3, 3))
    intputs[:, :, 0, 0] = input_tensor[:, :, 0, 0]
    intputs[:, :, 0, 2] = input_tensor[:, :, 0, 1]
    intputs[:, :, 2, 0] = input_tensor[:, :, 1, 0]
    intputs[:, :, 2, 2] = input_tensor[:, :, 1, 1]
    # Split input to not over GPU memory
    B = input_tensor.size(0) // 100
    LUT = []
    for b in range(100):
        # Get Denoise LUT
        # kernel：
        #       X 0 X
        #       0 0 0
        #       X 0 X
        # inputs:
        #    nums 0 nums
        #     0   0   0
        #    nums 0 nums
        print("Processing: ", b)
        if b == 99:
            raw_x = intputs[b*B:].numpy() / 255.
        else:
            raw_x = intputs[b*B:(b+1)*B].numpy() / 255.
        # raw_x = intputs.numpy() / 255.  # [N, 1, 3, 3]
        current_state = State.State((raw_x.shape[0], 1, 3, 3), MOVE_RANGE)
        agent = PixelWiseA3C_InnerState(model, optimizer, raw_x.shape[0], EPISODE_LEN, GAMMA)


        label = copy.deepcopy(raw_x)
        raw_n = np.zeros_like(raw_x)
        current_state.reset(raw_x, raw_n)
        reward = np.zeros(label.shape, label.dtype)
        sum_reward = 0

        for t in range(EPISODE_LEN):
            previous_image = np.clip(current_state.image.copy(), a_min=0., a_max=1.)
            action, inner_state, action_prob = agent.act_and_train(current_state.tensor, reward, test=True)
            # print(paint_amap(action[0]))
            # LUT = copy.deepcopy(action[:, 1, 1])
            LUT += [copy.deepcopy(action[:, 1, 1])]
            current_state.step(action, inner_state)
            reward = np.square(label - previous_image) * 255 - \
                     np.square(label - current_state.image) * 255
            sum_reward += np.mean(reward) * np.power(GAMMA, t)

    LUTs = np.concatenate(LUT, 0)
    print("Resulting LUT size: ", LUTs.shape)
    np.save("../LUTs/sample_{}_LUTs".format(SAMPLING_INTERVAL), LUTs)
