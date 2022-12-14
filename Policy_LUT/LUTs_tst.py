from tqdm import tqdm
from PIL import Image
import numpy as np
import glob
import matplotlib.pyplot as plt
from Policy_LUT.TransferLUTs import Interp
import cv2
from config import config

SAMPLING_INTERVAL = config.SAMPLING_INTERVAL        # N bit uniform sampling
SIGMA = config.SIGMA                  # Gaussian noise std
L = 2 ** (8 - SAMPLING_INTERVAL) + 1
LUT_PATH = "../LUTs/sample_{}_LUTs.npy".format(SAMPLING_INTERVAL)    # Trained SR net params
TEST_DIR = '../img_tst/'      # Test images

def paint_amap(acmap, num_action):
    image = np.asanyarray(acmap.squeeze(), dtype=np.uint8)
    plt.imshow(image, vmin=0, vmax=num_action)
    plt.colorbar()
    plt.show()
    # plt.pause(1)
    # plt.close()

# Load LUT
LUT = np.load(LUT_PATH).astype(np.float32).reshape(-1, 1)  # N(=(2^SAMPLING_INTERVAL + 1)^4D), 16(=r*r)


# Test clean images
files_gt = glob.glob(TEST_DIR + '*.png')
files_gt.sort()

for ti, fn in enumerate(tqdm(files_gt)):
    # Load noise image and gt
    img_gt = np.array(Image.open(files_gt[ti])).astype(np.int) / 255.
    h, w = img_gt.shape  # (481, 321)
    print(img_gt)
    # Add noise
    img_noisy = img_gt + np.random.normal(0, SIGMA, img_gt.shape).astype(img_gt.dtype)/255.
    img_noisy = np.clip(img_noisy, 0, 1)
    img_noisy = (img_noisy * 255.).astype(np.uint8)
    q = 2**SAMPLING_INTERVAL
    print(h, w)

    img_noisy = np.pad(img_noisy, ((1, 1), (1, 1)), mode='reflect')
    print(img_noisy)
    cv2.imshow('img_noisy', img_noisy)
    cv2.waitKey(0)
    img_noisy = np.expand_dims(img_noisy, 0)
    # Interp(LUT, np.expand_dims(img_noisy, 0), h, w, q, rot=0)
    # print(img_noisy.shape)
    # print("------")

    img_a1 = img_noisy[:, 0:0 + h, 0:0 + w] // q
    img_b1 = img_noisy[:, 0:0 + h, 2:2 + w] // q
    img_c1 = img_noisy[:, 2:2 + h, 0:0 + w] // q
    img_d1 = img_noisy[:, 2:2 + h, 2:2 + w] // q

    out_action = LUT[img_a1.flatten().astype(np.int_) * L * L * L +
                   img_b1.flatten().astype(np.int_) * L * L +
                   img_c1.flatten().astype(np.int_) * L +
                   img_d1.flatten().astype(np.int_)]. \
        reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2]))
    paint_amap(out_action, 10)
    print(out_action.shape)
    print(out_action)

















