from tqdm import tqdm
from PIL import Image
import numpy as np
import glob


SAMPLING_INTERVAL = 4        # N bit uniform sampling
SIGMA = 15                  # Gaussian noise std
LUT_PATH = "sample_{}_LUTs.npy".format(SAMPLING_INTERVAL)    # Trained SR net params
TEST_DIR = '../img_tst/'      # Test images



# Load LUT
LUT = np.load(LUT_PATH).astype(np.float32).reshape(-1, 1)  # N(=(2^SAMPLING_INTERVAL + 1)^4D), 16(=r*r)


# Test clean images
files_gt = glob.glob(TEST_DIR + '*.png')
files_gt.sort()

for ti, fn in enumerate(tqdm(files_gt)):
    # Load noise image and gt
    img_gt = np.array(Image.open(files_gt[ti])).astype(np.float32)
    h, w, c = img_gt.shape

    # Add noise
    img_noisy = img_gt + np.random.normal(0, SIGMA, img_gt.shape)
    img_noisy = np.clip(img_noisy, 0, 255)



















