import copy
import cv2
import torch
import time
from tqdm import tqdm
from PIL import Image
import numpy as np
import glob
import matplotlib.pyplot as plt
from Policy_LUT.Transfer_LUTs import transfer_lut
from scipy.special import softmax
from config import config
# import Train_Net.State as State
# import Train_Net.State_Bilateral as State
import Train_Net.State_Gaussian as State

SAMPLING_INTERVAL = 2        # N bit uniform sampling
SIGMA = config.SIGMA                  # Gaussian noise std
L = 2 ** (8 - SAMPLING_INTERVAL) + 1
q = 2**SAMPLING_INTERVAL

LUT_PATH = "../LUTs/sample_{}_LUTs.npy".format(SAMPLING_INTERVAL)    # Trained SR net params
TEST_DIR = '../img_tst/'      # Test images

def paint_amap(acmap, num_action):
    image = np.asanyarray(acmap.squeeze(), dtype=np.uint8)
    plt.imshow(image, vmin=0, vmax=num_action)
    plt.colorbar()
    plt.show()
    # plt.pause(1)
    # plt.close()

# Load action LUT
# LUT = np.load(LUT_PATH).astype(np.float32).reshape(-1, 1)  # N(=(2^SAMPLING_INTERVAL + 1)^4D), 1

# Load action LUT
LUT = np.load(LUT_PATH).astype(np.float32).reshape(-1, 9)  # N(=(2^SAMPLING_INTERVAL + 1)^4D), 9(=action num)

# Test clean images
files_gt = glob.glob(TEST_DIR + '*.png')
files_gt.sort()

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
    inner_state = np.zeros((1, 64, h, w))


    res = None
    t1 = time.time()

    for i in range(5):
        cv2.imshow('current_state.image_ins', (current_state.image[0, 0, :, :] * 255).astype(np.uint8))
        # cv2.waitKey(0)
        # out_action = transfer_lut((current_state.image[0, 0, :, :]*255).astype(np.uint8),
        #                           LUT, h, w, q, L)
        # paint_amap(out_action, 10)

        # rotation
        ins1 = copy.deepcopy(current_state.image[0, 0, :, :] * 255)
        out_policy1 = transfer_lut(ins1.astype(np.uint8),
                                  LUT, ins1.shape[0], ins1.shape[1], q, L, 0)

        ins2 = np.rot90(copy.deepcopy(current_state.image[0, 0, :, :] * 255), 1)
        out_policy2 = transfer_lut(ins2.astype(np.uint8),
                                   LUT, ins2.shape[0], ins2.shape[1], q, L, 3)


        ins3 = copy.deepcopy(current_state.image[0, 0, :, :] * 255)
        out_policy3 = transfer_lut((np.rot90(ins3, 2)).astype(np.uint8),
                                      LUT, h, w, q, L, 2)


        ins4 = np.rot90(copy.deepcopy(current_state.image[0, 0, :, :] * 255), 3)
        out_policy4 = transfer_lut(ins4.astype(np.uint8),
                                      LUT, ins4.shape[0], ins4.shape[1], q, L, 1)
        # print(out_policy1[0, 0, 0, :])
        # print(out_policy1.shape)
        # print(out_policy2.shape)
        # print(out_policy3.shape)
        # print(out_policy4.shape)
        # print("888")
        print(((out_policy1 + out_policy2 +
                      out_policy3 + out_policy4) / 4.)[0, 0, 0, :])
        # out_action = np.argmax(softmax((out_policy1 + out_policy2 +
        #               out_policy3 + out_policy4) / 4., axis=3), axis=3)
        out_action = np.argmax((out_policy1 + out_policy2 +
                                        out_policy3 + out_policy4) / 4., axis=3)
        paint_amap(out_action, 10)
        # print(out_action.shape)
        # print("lll")
        current_state.step(torch.Tensor(out_action), inner_state)
        if i == 4:
            res = copy.deepcopy(current_state.image[0, 0, :, :])
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
        cv2.waitKey(0)

    t2 = time.time()
    print('消耗时间：',(t2 - t1)*1000, "ms")
















