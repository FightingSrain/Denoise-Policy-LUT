import copy
import cv2
import collections
import torch
import torch.nn.functional as F
import time
from tqdm import tqdm
from PIL import Image
from skimage.metrics import structural_similarity as ssim_cal
from skimage.metrics import peak_signal_noise_ratio as psnr_cal
import numpy as np
import glob
import matplotlib.pyplot as plt
from scipy.special import softmax
from config import config
from Train_denoise_net.interp import FourSimplexInterp




SAMPLING_INTERVAL = 4       # N bit uniform sampling
sigma = 25               # Gaussian noise std
L = 2 ** (8 - SAMPLING_INTERVAL) + 1
q = 2**SAMPLING_INTERVAL

LUT_PATH = "./DenoiseNet_LUTs_{}/sample_{}_LUTs_{}.npy".format(sigma, SAMPLING_INTERVAL, sigma)    # Trained DP net params
# TEST_DIR = '../img_tst/'      # Test images
# TEST_DIR = 'D://Dataset/BSD68_color/'      # Test images
# TEST_DIR = 'D://Dataset/Set5/'      # Test images
TEST_DIR = 'D://Dataset/Kodak24/'      # Test images
def paint_amap(acmap, num_action):
    image = np.asanyarray(acmap.squeeze(), dtype=np.uint8)
    plt.imshow(image, vmin=0, vmax=num_action)
    plt.colorbar()
    plt.show()
    # plt.pause(1)
    # plt.close()

def interp_LUTs_main():
    mods = ['a', 'b', 'c']
    # Test clean images
    files_gt = glob.glob(TEST_DIR + '*.png')
    files_gt.sort()
    lens = len(files_gt)
    # ---------------
    total_psnr = 0
    total_ssim = 0

    for ti, fn in enumerate(tqdm(files_gt)):
        # Load noise image and gt
        img_gt = np.asanyarray(Image.open(files_gt[ti])).astype(np.float32) / 255.
        # 转为RGB
        img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
        h, w, c = img_gt.shape  # (481, 321)
        raw_n = np.random.normal(0, 15, img_gt.shape).astype(img_gt.dtype) / 255.
        ins_noisy = (np.clip(img_gt + raw_n, a_min=0., a_max=1.) * 255).astype(np.uint8)

        res1s, res2s, res3s, res4s = 0, 0, 0, 0
        for mod in mods:
            LUT_PATH = "./DenoiseNet_LUTs_{}/sample_{}_LUTs{}_{}.npy".format(sigma, SAMPLING_INTERVAL, mod,
                                                                           sigma)  # Trained DP net params
            # Load policy LUT
            LUT = np.load(LUT_PATH).astype(np.float32).reshape(-1, 1)
            # ---------------



            # rotation
            ins1 = ins_noisy.copy()

            if mod == 'a':
                # print(ins1.shape)
                ins1 = np.pad(ins1, ((0, 1), (0, 1), (0, 0)), mode='reflect')
                # print(ins1.shape)
            else:
                # print(ins1.shape)
                ins1 = np.pad(ins1, ((0, 2), (0, 2), (0, 0)), mode='reflect')
                # print(ins1.shape)
            # print(ins1.shape)
            # print("UUUUUUUU")
            ins1s = ins1.transpose(2, 0, 1)
            # print(ins1s.shape)
            # print(len(LUT))
            # print(mod)
            res1s += FourSimplexInterp(LUT, ins1s.astype(np.uint8),
                                        h, w, q, L, mod)
            # res1 += np.rot90(res1, 0, axes=[1, 2])


            ins2 = np.rot90(ins_noisy.copy(), 1, axes=[0, 1])

            if mod == 'a':
                ins2 = np.pad(ins2, ((0, 1), (0, 1), (0, 0)), mode='reflect')
            else:
                ins2 = np.pad(ins2, ((0, 2), (0, 2), (0, 0)), mode='reflect')
            ins2s = ins2.transpose(2, 0, 1)
            # print(ins2s.shape)
            res2 = FourSimplexInterp(LUT, ins2s.astype(np.uint8),
                                     w, h, q, L, mod)

            res2s += np.rot90(res2, 3, axes=[0, 1])

            ins3 = np.rot90(ins_noisy.copy(), 2, axes=[0, 1])

            if mod == 'a':
                ins3 = np.pad(ins3, ((0, 1), (0, 1), (0, 0)), mode='reflect')
            else:
                ins3 = np.pad(ins3, ((0, 2), (0, 2), (0, 0)), mode='reflect')
            ins3s = ins3.transpose(2, 0, 1)
            res3 = FourSimplexInterp(LUT, ins3s.astype(np.uint8),
                                     h, w, q, L, mod)
            res3s += np.rot90(res3, 2, axes=[0, 1])

            ins4 = np.rot90(ins_noisy.copy(), 3, axes=[0, 1])

            if mod == 'a':
                ins4 = np.pad(ins4, ((0, 1), (0, 1), (0, 0)), mode='reflect')
            else:
                ins4 = np.pad(ins4, ((0, 2), (0, 2), (0, 0)), mode='reflect')
            ins4s = ins4.transpose(2, 0, 1)
            res4 = FourSimplexInterp(LUT, ins4s.astype(np.uint8),
                                     w, h, q, L, mod)
            res4s += np.rot90(res4, 1, axes=[0, 1])

        res = np.tanh((res1s/3. + res2s/3. + res3s/3. + res4s/3.) / 4.) + ins_noisy / 255.
        res = np.clip(res, a_min=0., a_max=1.)

        cv2.imshow("gt", (img_gt * 255).astype(np.uint8))
        cv2.imshow("denoise", (res * 255).astype(np.uint8))
        cv2.imshow("noise", ins_noisy)
        cv2.waitKey(0)

        # ---------------
        psnr = psnr_cal(img_gt, res)
        ssim = ssim_cal(img_gt, res, multichannel=True)
        total_psnr += psnr
        total_ssim += ssim
        # ---------------
    print(total_psnr / lens, total_ssim / lens)
if __name__ == '__main__':
    interp_LUTs_main()