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
from Train_Hybrid_Policy.Transfer_LUTs_interp import FourSimplexInterp
from scipy.special import softmax
from config import config
from utils import *
from numba import jit
# import Train_Hybrid_Policy.State as State
# import Train_Hybrid_Policy.State_Bilateral as State
import Train_Hybrid_Policy.State_Gaussian as State
from collections import Counter
from skimage.metrics import structural_similarity as ssim




SAMPLING_INTERVAL = 4       # N bit uniform sampling
SIGMA = config.SIGMA                  # Gaussian noise std
L = 2 ** (8 - SAMPLING_INTERVAL) + 1
q = 2**SAMPLING_INTERVAL

LUT_PATH = "./selfHybrid_LUTs_15/sample_{}_LUTs_{}.npy".format(SAMPLING_INTERVAL, SIGMA)    # Trained DP net params
# TEST_DIR = '../img_tst/'      # Test images
TEST_DIR = 'D://Dataset/BSD68/'      # Test images
# TEST_DIR = 'D://Dataset/Set12/'      # Test images
def paint_amap(acmap, num_action):
    image = np.asanyarray(acmap.squeeze(), dtype=np.uint8)
    plt.imshow(image, vmin=0, vmax=num_action)
    plt.colorbar()
    plt.show()
    # plt.pause(1)
    # plt.close()

def interp_LUTs_main():
    # Load action LUT
    # LUT = np.load(LUT_PATH).astype(np.float32).reshape(-1, 1)  # N(=(2^SAMPLING_INTERVAL + 1)^4D), 1

    # Load policy LUT
    LUT = np.load(LUT_PATH).astype(np.float32).reshape(-1, config.N_ACTIONS*2)  # N(=(2^SAMPLING_INTERVAL + 1)^4D), 9(=action num)

    # Test clean images
    files_gt = glob.glob(TEST_DIR + '*.png')
    files_gt.sort()
    lens = len(files_gt)
    # ---------------
    action_num = np.zeros((5, config.N_ACTIONS))
    total_psnr = 0
    total_ssim = 0
    # ---------------
    for ti, fn in enumerate(tqdm(files_gt)):
        # Load noise image and gt
        img_gt = np.asanyarray(Image.open(files_gt[ti])) / 255.
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
        current_state.reset(ins_noisy)


        res = None
        t1 = time.time()

        for i in range(5):
            cv2.imshow('current_state.image_ins', (current_state.image[0, 0, :, :] * 255).astype(np.uint8))
            # cv2.waitKey(0)


            # rotation
            ins1 = current_state.image[0, 0, :, :] * 255.
            ins1 = np.pad(ins1, ((0, 2), (0, 2)), mode='reflect')
            ins1 = np.expand_dims(ins1, 0)
            policy1 = FourSimplexInterp(LUT, ins1.astype(np.uint8),
                                          h, w, q, L, config.N_ACTIONS, 0)
            D_policy1 = np.rot90(policy1[:, :, :, 0:config.N_ACTIONS], 0, axes=[1, 2])
            C_policy1 = np.rot90(policy1[:, :, :, config.N_ACTIONS:config.N_ACTIONS*2], 0, axes=[1, 2])


            ins2 = np.rot90(current_state.image[0, 0, :, :] * 255., 1)
            ins2 = np.pad(ins2, ((0, 2), (0, 2)), mode='reflect')
            ins2 = np.expand_dims(ins2, 0)
            policy2 = FourSimplexInterp(LUT, ins2.astype(np.uint8),
                                          w, h, q, L, config.N_ACTIONS, 3)
            D_policy2 = np.rot90(policy2[:, :, :, 0:config.N_ACTIONS], 3, axes=[1, 2])
            C_policy2 = np.rot90(policy2[:, :, :, config.N_ACTIONS:config.N_ACTIONS * 2], 3, axes=[1, 2])



            ins3 = np.rot90(current_state.image[0, 0, :, :] * 255., 2)
            ins3 = np.pad(ins3, ((0, 2), (0, 2)), mode='reflect')
            ins3 = np.expand_dims(ins3, 0)
            policy3 = FourSimplexInterp(LUT, ins3.astype(np.uint8),
                                          h, w, q, L, config.N_ACTIONS, 2)
            D_policy3 = np.rot90(policy3[:, :, :, 0:config.N_ACTIONS], 2, axes=[1, 2])
            C_policy3 = np.rot90(policy3[:, :, :, config.N_ACTIONS:config.N_ACTIONS * 2], 2, axes=[1, 2])


            ins4 = np.rot90(current_state.image[0, 0, :, :] * 255., 3)
            ins4 = np.pad(ins4, ((0, 2), (0, 2)), mode='reflect')
            ins4 = np.expand_dims(ins4, 0)
            policy4 = FourSimplexInterp(LUT, ins4.astype(np.uint8),
                                          w, h, q, L, config.N_ACTIONS, 1)
            D_policy4 = np.rot90(policy4[:, :, :, 0:config.N_ACTIONS], 1, axes=[1, 2])
            C_policy4 = np.rot90(policy4[:, :, :, config.N_ACTIONS:config.N_ACTIONS * 2], 1, axes=[1, 2])



            # ins5 = current_state.image[0, 0, :, :] * 255.
            # ins5 = np.pad(ins5, ((1, 1), (1, 1)), mode='reflect')
            # ins5 = np.expand_dims(ins5, 0)
            # policy5 = FourSimplexInterp(LUT, ins5.astype(np.uint8),
            #                               w, h, q, L, config.N_ACTIONS, 1)
            # D_policy5 = policy5[:, :, :, 0:config.N_ACTIONS],
            # C_policy5 = policy5[:, :, :, config.N_ACTIONS:config.N_ACTIONS * 2]


            D_action = np.argmax(Softmax(((D_policy1 + D_policy2 + D_policy3 + D_policy4) / 4.), 3), 3)
            # D_action = torch.argmax(F.softmax(torch.Tensor((D_policy1 + D_policy2 +
            #               D_policy3 + D_policy4) / 4.), dim=3), dim=3).numpy()
            # 更具D_action的值选择C_action维度3的下标
            # C_action = torch.Tensor((C_policy1 + C_policy2 + C_policy3 + C_policy4) / 4.).\
            #     gather(3, torch.Tensor(D_action).unsqueeze(3).long()).numpy()
            C_action = torch.Tensor((C_policy1 + C_policy2 + C_policy3 + C_policy4) / 4.).numpy()[:, 0, 0, :]

            # print(D_action.shape)
            # print(C_action.shape)
            # print("8888888888888")

            # paint_amap(D_action, 10)

            # data_count = collections.Counter(D_action.reshape((-1,))).items()
            # for key, value in data_count:
            #     action_num[i][key] += (value/(D_action.shape[1]*D_action.shape[2]))
            #     print(key, value)

            # print(data_count)
            # print(ti)
            # 输出动作参数
            # print([current_state.hybrid_act(3, 0, C_action),
            #        current_state.hybrid_act(4, 0, C_action),
            #        current_state.hybrid_act(5, 0, C_action),
            #        current_state.hybrid_act(6, 0, C_action)])
            current_state.step(D_action, C_action)
            if i == 4:
                res = current_state.image[0, 0, :, :]

                # guide_res = cv2.ximgproc.guidedFilter(ins_noisy[0, 0, :, :].astype(np.float32),
                #                                       res.astype(np.float32), 5, 0.01, -1)
                # cv2.imshow('res', (res * 255).astype(np.uint8))
                # cv2.imshow('guide_res', (guide_res * 255).astype(np.uint8))
                # cv2.waitKey(0)
                # cv2.imwrite("./res_img/Noise_img_Set12/Noise{}.png".format(ti), (ins_noisy[0, 0, :, :] * 255).astype(np.uint8))
                # cv2.imwrite("./res_img/Hybrid_Set12/res{}.png".format(ti), (res * 255).astype(np.uint8))
                # cv2.imwrite("../res_img/Bilateral_Set12/res{}.png".format(ti),
                #             cv2.bilateralFilter((ins_noisy[0, 0, :, :] * 255).astype(np.uint8),
                #                                 d=5, sigmaColor=100, sigmaSpace=20)
                #             )
                # t1s = time.time()
                # cv2.imwrite("../res_img/NLM_BSD68/res{}.png".format(ti),
                #             cv2.fastNlMeansDenoising((ins_noisy[0, 0, :, :] * 255).astype(np.uint8),
                #                                 h=15, templateWindowSize=7, searchWindowSize=21)
                #             )
                # nlm = cv2.fastNlMeansDenoising((ins_noisy[0, 0, :, :] * 255).astype(np.uint8),
                #                          h=15, templateWindowSize=7, searchWindowSize=21)
                # t2s = time.time()
                # print("NLM time: ", (t2s - t1s)*1000, "ms")
                # -------------filter psnr and ssim--------------
                # filter = cv2.GaussianBlur(ins_noisy[0, 0, :, :], (5, 5), 0.5)
                # filter = cv2.bilateralFilter(ins_noisy[0, 0, :, :].astype(np.float32), d=5, sigmaColor=0.5, sigmaSpace=5)
                # filter = ins_noisy[0, 0, :, :].astype(np.float32)
                # filter = cv2.fastNlMeansDenoising((ins_noisy[0, 0, :, :] * 255).astype(np.uint8),
                #                                    h=15, templateWindowSize=7, searchWindowSize=21)
                # filter_psnr = cv2.PSNR((filter).astype(np.uint8),
                #                              (img_gts[0, 0, :, :] * 255).astype(np.uint8))
                # filter_ssim = ssim((filter).astype(np.uint8),
                #                              (img_gts[0, 0, :, :] * 255).astype(np.uint8))
                # print(filter_psnr, filter_ssim)
                # total_psnr += filter_psnr
                # total_ssim += filter_ssim
                # print("res_psnr: ", gauss_psnr/lens)
                # print("res_ssim: ", gauss_ssim/lens)

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
        print('消耗时间：', (t2 - t1)*1000, "ms")

        # total_psnr += cv2.PSNR((res * 255).astype(np.uint8),
        #                        (img_gts[0, 0, :, :] * 255).astype(np.uint8))
        #
        # total_ssim += ssim((res * 255).astype(np.uint8),
        #                        (img_gts[0, 0, :, :] * 255).astype(np.uint8))
    print(total_psnr / lens)
    print(total_ssim / lens)

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


if __name__ == '__main__':
    interp_LUTs_main()













