
import numpy as np


@staticmethod
def FourSimplexInterp(weight, img_in, h, w, q, rot, upscale=4):

    # Extract MSBs
    img_a1 = img_in[:, 0:0 + h, 0:0 + w] // q
    img_b1 = img_in[:, 0:0 + h, 2:2 + w] // q
    img_c1 = img_in[:, 2:2 + h, 0:0 + w] // q
    img_d1 = img_in[:, 2:2 + h, 2:2 + w] // q

    img_a2 = img_a1 + 1
    img_b2 = img_b1 + 1
    img_c2 = img_c1 + 1
    img_d2 = img_d1 + 1

    # Extract LSBs
    fa_ = img_in[:, 0:0 + h, 0:0 + w] % q
    fb_ = img_in[:, 0:0 + h, 2:2 + w] % q
    fc_ = img_in[:, 2:2 + h, 0:0 + w] % q
    fd_ = img_in[:, 2:2 + h, 2:2 + w] % q
















