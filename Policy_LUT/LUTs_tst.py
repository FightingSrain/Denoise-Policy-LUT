from tqdm import tqdm
from PIL import Image
import numpy as np
import glob
from Policy_LUT.TransferLUTs import Interp

SAMPLING_INTERVAL = 4        # N bit uniform sampling
SIGMA = 15                  # Gaussian noise std
LUT_PATH = "../LUTs/sample_{}_LUTs.npy".format(SAMPLING_INTERVAL)    # Trained SR net params
TEST_DIR = '../img_tst/'      # Test images



# Load LUT
LUT = np.load(LUT_PATH).astype(np.float32).reshape(-1, 1)  # N(=(2^SAMPLING_INTERVAL + 1)^4D), 16(=r*r)


# Test clean images
files_gt = glob.glob(TEST_DIR + '*.png')
files_gt.sort()

for ti, fn in enumerate(tqdm(files_gt)):
    # Load noise image and gt
    img_gt = np.array(Image.open(files_gt[ti])).astype(np.float32)
    h, w = img_gt.shape # (481, 321)

    # Add noise
    img_noisy = img_gt + np.random.normal(0, SIGMA, img_gt.shape)
    img_noisy = np.clip(img_noisy, 0, 255)
    q = 2**SAMPLING_INTERVAL
    print(h, w)

    img_noisy = np.pad(img_noisy, ((1, 1), (1, 1)), mode='reflect')
    Interp(LUT, np.expand_dims(img_noisy, 0), h, w, q, rot=0)
    print(img_noisy.shape)
    print("------")



















