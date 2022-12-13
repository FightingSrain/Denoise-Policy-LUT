
import numpy as np


@ staticmethod
def FourSimplexInterp(weight, img_in, h, w, q, rot, SAMPLING_INTERVAL=4):
    L = 2 ** (8 - SAMPLING_INTERVAL) + 1
    upscale = 1
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

    # Vertices (O in Eq3 and Table3 in the paper)
    p0000 = weight[img_a1.flatten().astype(np.int_) * L * L * L +
                   img_b1.flatten().astype(np.int_) * L * L +
                   img_c1.flatten().astype(np.int_) * L +
                   img_d1.flatten().astype(np.int_)].\
        reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))

    p0001 = weight[img_a1.flatten().astype(np.int_) * L * L * L +
                   img_b1.flatten().astype(np.int_) * L * L +
                   img_c1.flatten().astype(np.int_) * L +
                   img_d2.flatten().astype(np.int_)].\
        reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))

    p0010 = weight[img_a1.flatten().astype(np.int_) * L * L * L +
                   img_b1.flatten().astype(np.int_) * L * L +
                   img_c2.flatten().astype(np.int_) * L +
                   img_d1.flatten().astype(np.int_)].\
        reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))

    p0011 = weight[img_a1.flatten().astype(np.int_) * L * L * L +
                   img_b1.flatten().astype(np.int_) * L * L +
                   img_c2.flatten().astype(np.int_) * L +
                   img_d2.flatten().astype(np.int_)].\
        reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))

    p0100 = weight[img_a1.flatten().astype(np.int_) * L * L * L +
                   img_b2.flatten().astype(np.int_) * L * L +
                   img_c1.flatten().astype(np.int_) * L +
                   img_d1.flatten().astype(np.int_)].\
        reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))

    p0101 = weight[img_a1.flatten().astype(np.int_) * L * L * L +
                   img_b2.flatten().astype(np.int_) * L * L +
                   img_c1.flatten().astype(np.int_) * L +
                   img_d2.flatten().astype(np.int_)].\
        reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))

    p0110 = weight[img_a1.flatten().astype(np.int_) * L * L * L +
                   img_b2.flatten().astype(np.int_) * L * L +
                   img_c2.flatten().astype(np.int_) * L +
                   img_d1.flatten().astype(np.int_)].\
        reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))

    p0111 = weight[img_a1.flatten().astype(np.int_) * L * L * L +
                   img_b2.flatten().astype(np.int_) * L * L +
                   img_c2.flatten().astype(np.int_) * L +
                   img_d2.flatten().astype(np.int_)].\
        reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    # -----------------------------------------
    p1000 = weight[img_a2.flatten().astype(np.int_) * L * L * L +
                   img_b1.flatten().astype(np.int_) * L * L +
                   img_c1.flatten().astype(np.int_) * L +
                   img_d1.flatten().astype(np.int_)].\
        reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))

    p1001 = weight[img_a2.flatten().astype(np.int_) * L * L * L +
                   img_b1.flatten().astype(np.int_) * L * L +
                   img_c1.flatten().astype(np.int_) * L +
                   img_d2.flatten().astype(np.int_)].\
        reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))

    p1010 = weight[img_a2.flatten().astype(np.int_) * L * L * L +
                   img_b1.flatten().astype(np.int_) * L * L +
                   img_c2.flatten().astype(np.int_) * L +
                   img_d1.flatten().astype(np.int_)].\
        reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))

    p1011 = weight[img_a2.flatten().astype(np.int_) * L * L * L +
                   img_b1.flatten().astype(np.int_) * L * L +
                   img_c2.flatten().astype(np.int_) * L +
                   img_d2.flatten().astype(np.int_)].\
        reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))

    p1100 = weight[img_a2.flatten().astype(np.int_) * L * L * L +
                   img_b2.flatten().astype(np.int_) * L * L +
                   img_c1.flatten().astype(np.int_) * L +
                   img_d1.flatten().astype(np.int_)].\
        reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))

    p1101 = weight[img_a2.flatten().astype(np.int_) * L * L * L +
                   img_b2.flatten().astype(np.int_) * L * L +
                   img_c1.flatten().astype(np.int_) * L +
                   img_d2.flatten().astype(np.int_)].\
        reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))

    p1110 = weight[img_a2.flatten().astype(np.int_) * L * L * L +
                   img_b2.flatten().astype(np.int_) * L * L +
                   img_c2.flatten().astype(np.int_) * L +
                   img_d1.flatten().astype(np.int_)].\
        reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))

    p1111 = weight[img_a2.flatten().astype(np.int_) * L * L * L +
                   img_b2.flatten().astype(np.int_) * L * L +
                   img_c2.flatten().astype(np.int_) * L +
                   img_d2.flatten().astype(np.int_)].\
        reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))














