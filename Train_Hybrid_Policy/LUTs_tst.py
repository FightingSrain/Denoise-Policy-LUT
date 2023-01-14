import copy
import cv2
import collections
import torch
import torch.nn.functional as F
import time
from tqdm import tqdm
from PIL import Image
import numpy as np
import glob
import matplotlib.pyplot as plt
from Train_Hybrid_Policy.Transfer_LUTs import transfer_lut
from scipy.special import softmax
from config import config
from utils import *
# import Train_Hybrid_Policy.State as State
# import Train_Hybrid_Policy.State_Bilateral as State
import Train_Hybrid_Policy.State_Gaussian as State
from collections import Counter





SAMPLING_INTERVAL = 4        # N bit uniform sampling
SIGMA = config.SIGMA                  # Gaussian noise std
L = 2 ** (8 - SAMPLING_INTERVAL) + 1
q = 2**SAMPLING_INTERVAL

LUT_PATH = "./Hybrid_LUTs/sample_{}_LUTs.npy".format(SAMPLING_INTERVAL)    # Trained SR net params
# TEST_DIR = '../img_tst/'      # Test images
# TEST_DIR = 'D://Dataset/BSD68/'      # Test images
TEST_DIR = 'D://Dataset/Set12/'      # Test images
def paint_amap(acmap, num_action):
    image = np.asanyarray(acmap.squeeze(), dtype=np.uint8)
    plt.imshow(image, vmin=0, vmax=num_action)
    plt.colorbar()
    plt.show()
    # plt.pause(1)
    # plt.close()

# Load action LUT
# LUT = np.load(LUT_PATH).astype(np.float32).reshape(-1, 1)  # N(=(2^SAMPLING_INTERVAL + 1)^4D), 1

# Load policy LUT
LUT = np.load(LUT_PATH).astype(np.float32).reshape(-1, config.N_ACTIONS*2)  # N(=(2^SAMPLING_INTERVAL + 1)^4D), 9(=action num)

# Test clean images
files_gt = glob.glob(TEST_DIR + '*.png')
files_gt.sort()
# ---------------
action_num = np.zeros((5, config.N_ACTIONS))


# ---------------
for ti, fn in enumerate(tqdm(files_gt)):
    # Load noise image and gt
    img_gt = np.array(Image.open(files_gt[ti])).astype(np.int) / 255.
    h, w = img_gt.shape  # (481, 321)
    # print(img_gt)
    # Add noise
    # img_noisy = img_gt + np.random.normal(0, SIGMA, img_gt.shape).astype(img_gt.dtype)/255.
    # img_noisy = np.clip(img_noisy, 0, 1)
    # img_noisy = (img_noisy * 255.).astype(np.uint8)
    # q = 2**SAMPLING_INTERVAL
    # print(h, w)

    # img_noisy = np.pad(img_noisy, ((1, 1), (1, 1)), mode='reflect')
    # img_noisy = np.expand_dims(img_noisy, 0)
    #
    # img_a1 = img_noisy[:, 0:0 + h, 0:0 + w] // q
    # img_b1 = img_noisy[:, 0:0 + h, 2:2 + w] // q
    # img_c1 = img_noisy[:, 2:2 + h, 0:0 + w] // q
    # img_d1 = img_noisy[:, 2:2 + h, 2:2 + w] // q
    #
    # out_action = LUT[img_a1.flatten().astype(np.int_) * L * L * L +
    #                img_b1.flatten().astype(np.int_) * L * L +
    #                img_c1.flatten().astype(np.int_) * L +
    #                img_d1.flatten().astype(np.int_)]. \
    #     reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2]))

    # -------------------------------------------
    img_gts = copy.deepcopy(img_gt)
    img_gts = np.reshape(img_gts, (1, 1, h, w))
    current_state = State.State((1, 1, h, w), config.MOVE_RANGE)
    raw_n = np.random.normal(0, SIGMA, img_gts.shape).astype(img_gts.dtype) / 255.
    ins_noisy = np.clip(img_gts + raw_n, a_min=0., a_max=1.)
    current_state.reset(img_gts, raw_n)


    res = None
    t1 = time.time()

    for i in range(5):
        cv2.imshow('current_state.image_ins', (current_state.image[0, 0, :, :] * 255).astype(np.uint8))
        # cv2.waitKey(0)
        # out_action = transfer_lut((current_state.image[0, 0, :, :]*255).astype(np.uint8),
        #                           LUT, h, w, q, L)
        # paint_amap(out_action, 10)

        # rotation
        ins1 = current_state.image[0, 0, :, :] * 255
        D_policy1, C_policy1 = transfer_lut(ins1.astype(np.uint8),
                                  LUT, h, w, config.N_ACTIONS, q, L, 0)

        ins2 = np.rot90(current_state.image[0, 0, :, :] * 255, 1)
        D_policy2, C_policy2 = transfer_lut(ins2.astype(np.uint8),
                                   LUT, w, h, config.N_ACTIONS, q, L, 3)


        ins3 = current_state.image[0, 0, :, :] * 255
        D_policy3, C_policy3 = transfer_lut((np.rot90(ins3, 2)).astype(np.uint8),
                                      LUT, h, w, config.N_ACTIONS, q, L, 2)


        ins4 = np.rot90(current_state.image[0, 0, :, :] * 255, 3)
        D_policy4, C_policy4 = transfer_lut(ins4.astype(np.uint8),
                                      LUT, w, h, config.N_ACTIONS, q, L, 1)
        # print(out_policy1[0, 0, 0, :])
        # print(out_policy1.shape)
        # print(out_policy2.shape)
        # print(out_policy3.shape)
        # print(out_policy4.shape)
        # print("888")
        # print(((out_policy1 + out_policy2 +
        #               out_policy3 + out_policy4) / 4.)[0, 0, 0, :])
        D_action = torch.argmax(F.softmax(torch.Tensor((D_policy1 + D_policy2 +
                      D_policy3 + D_policy4) / 4.), dim=3), dim=3).numpy()
        # 更具D_action的值选择C_action维度3的下标
        # C_action = torch.Tensor((C_policy1 + C_policy2 + C_policy3 + C_policy4) / 4.).\
        #     gather(3, torch.Tensor(D_action).unsqueeze(3).long()).numpy()
        C_action = torch.Tensor((C_policy1 + C_policy2 + C_policy3 + C_policy4) / 4.).numpy().reshape(1, -1, config.N_ACTIONS).mean(1)

        # print(D_action.shape)
        # print(C_action.shape)
        # print("8888888888888")

        # paint_amap(out_action, 10)

        data_count = collections.Counter(D_action.reshape((-1,))).items()
        for key, value in data_count:
            action_num[i][key] += (value/(D_action.shape[1]*D_action.shape[2]))
            print(key, value)

        # print(data_count)
        # print(ti)
        # 输出动作参数
        # print([current_state.hybrid_act(3, 0, C_action),
        #        current_state.hybrid_act(4, 0, C_action),
        #        current_state.hybrid_act(5, 0, C_action),
        #        current_state.hybrid_act(6, 0, C_action)])
        current_state.step(torch.Tensor(D_action), C_action)
        if i == 4:
            res = copy.deepcopy(current_state.image[0, 0, :, :])
            # cv2.imwrite("../res_img/BSD68/res{}.png".format(ti), (res * 255).astype(np.uint8))
            cv2.imwrite("./res_img/Hybrid_Set12/res{}.png".format(ti), (res * 255).astype(np.uint8))
            # cv2.imwrite("../res_img/Bilateral_Set12/res{}.png".format(ti),
            #             cv2.bilateralFilter((ins_noisy[0, 0, :, :] * 255).astype(np.uint8),
            #                                 d=5, sigmaColor=100, sigmaSpace=20)
            #             )
            t1s = time.time()
            # cv2.imwrite("../res_img/NLM_BSD68/res{}.png".format(ti),
            #             cv2.fastNlMeansDenoising((ins_noisy[0, 0, :, :] * 255).astype(np.uint8),
            #                                 h=15, templateWindowSize=7, searchWindowSize=21)
            #             )
            nlm = cv2.fastNlMeansDenoising((ins_noisy[0, 0, :, :] * 255).astype(np.uint8),
                                     h=15, templateWindowSize=7, searchWindowSize=21)
            t2s = time.time()
            print("NLM time: ", (t2s - t1s)*1000, "ms")
        # ori_psnr = cv2.PSNR((ins_noisy[0, 0, :, :]*255).astype(np.uint8),
        #                     (img_gts[0, 0, :, :]*255).astype(np.uint8))
        # print('ori_psnr: ', ori_psnr)
        # tmp_psnr = cv2.PSNR((current_state.image[0, 0, :, :] * 255).astype(np.uint8),
        #                     (img_gts[0, 0, :, :] * 255).astype(np.uint8))
        #
        # print("PSNR: ", tmp_psnr)
        # print("---------------------------------")
        # print(current_state.image.shape)
        # print("------------")
        cv2.imshow('current_state.image', (current_state.image[0, 0, :, :]*255).astype(np.uint8))
        cv2.waitKey(1)

    t2 = time.time()
    print('消耗时间：', (t2 - t1)*1000, "ms")

# --------------------------
# category_names = ['Pixel-1', 'Do nothing',
#                   'Pixel+1', 'G1',
#                   'G2', 'G3',
#                   'G4', 'G5',
#                   'G6']
# results = {
#     'Step 1': np.round(action_num[0]/68., 4),
#     'Step 2': np.round(action_num[1]/68., 4),
#     'Step 3': np.round(action_num[2]/68., 4),
#     'Step 4': np.round(action_num[3]/68., 4),
#     'Step 5': np.round(action_num[4]/68., 4),
# }
# # print(action_num/68.)
# survey(results, category_names)
# plt.show()
# --------------------------













