
import numpy as np
from numba import jit


# 4D equivalent of triangular interpolation
@jit(nopython=True)
def FourSimplexInterp(weight, img_in, h, w, q, L, num_act, rot):
    # L = 2 ** (8 - SAMPLING_INTERVAL) + 1

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
                   img_d1.flatten().astype(np.int_), :].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], num_act))
    p0001 = weight[img_a1.flatten().astype(np.int_) * L * L * L +
                   img_b1.flatten().astype(np.int_) * L * L +
                   img_c1.flatten().astype(np.int_) * L +
                   img_d2.flatten().astype(np.int_), :].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], num_act))
    p0010 = weight[img_a1.flatten().astype(np.int_) * L * L * L +
                   img_b1.flatten().astype(np.int_) * L * L +
                   img_c2.flatten().astype(np.int_) * L +
                   img_d1.flatten().astype(np.int_), :].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], num_act))
    p0011 = weight[img_a1.flatten().astype(np.int_) * L * L * L +
                   img_b1.flatten().astype(np.int_) * L * L +
                   img_c2.flatten().astype(np.int_) * L +
                   img_d2.flatten().astype(np.int_), :].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], num_act))
    p0100 = weight[img_a1.flatten().astype(np.int_) * L * L * L +
                   img_b2.flatten().astype(np.int_) * L * L +
                   img_c1.flatten().astype(np.int_) * L +
                   img_d1.flatten().astype(np.int_), :].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], num_act))
    p0101 = weight[img_a1.flatten().astype(np.int_) * L * L * L +
                   img_b2.flatten().astype(np.int_) * L * L +
                   img_c1.flatten().astype(np.int_) * L +
                   img_d2.flatten().astype(np.int_), :].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], num_act))
    p0110 = weight[img_a1.flatten().astype(np.int_) * L * L * L +
                   img_b2.flatten().astype(np.int_) * L * L +
                   img_c2.flatten().astype(np.int_) * L +
                   img_d1.flatten().astype(np.int_), :].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], num_act))
    p0111 = weight[img_a1.flatten().astype(np.int_) * L * L * L +
                   img_b2.flatten().astype(np.int_) * L * L +
                   img_c2.flatten().astype(np.int_) * L +
                   img_d2.flatten().astype(np.int_), :].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], num_act))

    p1000 = weight[img_a2.flatten().astype(np.int_) * L * L * L +
                   img_b1.flatten().astype(np.int_) * L * L +
                   img_c1.flatten().astype(np.int_) * L +
                   img_d1.flatten().astype(np.int_), :].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], num_act))
    p1001 = weight[img_a2.flatten().astype(np.int_) * L * L * L +
                   img_b1.flatten().astype(np.int_) * L * L +
                   img_c1.flatten().astype(np.int_) * L +
                   img_d2.flatten().astype(np.int_), :].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], num_act))
    p1010 = weight[img_a2.flatten().astype(np.int_) * L * L * L +
                   img_b1.flatten().astype(np.int_) * L * L +
                   img_c2.flatten().astype(np.int_) * L +
                   img_d1.flatten().astype(np.int_), :].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], num_act))
    p1011 = weight[img_a2.flatten().astype(np.int_) * L * L * L +
                   img_b1.flatten().astype(np.int_) * L * L +
                   img_c2.flatten().astype(np.int_) * L +
                   img_d2.flatten().astype(np.int_), :].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], num_act))
    p1100 = weight[img_a2.flatten().astype(np.int_) * L * L * L +
                   img_b2.flatten().astype(np.int_) * L * L +
                   img_c1.flatten().astype(np.int_) * L +
                   img_d1.flatten().astype(np.int_), :].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], num_act))
    p1101 = weight[img_a2.flatten().astype(np.int_) * L * L * L +
                   img_b2.flatten().astype(np.int_) * L * L +
                   img_c1.flatten().astype(np.int_) * L +
                   img_d2.flatten().astype(np.int_), :].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], num_act))
    p1110 = weight[img_a2.flatten().astype(np.int_) * L * L * L +
                   img_b2.flatten().astype(np.int_) * L * L +
                   img_c2.flatten().astype(np.int_) * L +
                   img_d1.flatten().astype(np.int_), :].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], num_act))
    p1111 = weight[img_a2.flatten().astype(np.int_) * L * L * L +
                   img_b2.flatten().astype(np.int_) * L * L +
                   img_c2.flatten().astype(np.int_) * L +
                   img_d2.flatten().astype(np.int_), :].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], num_act))

    # Output image holder
    out = np.zeros((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], num_act))

    # Naive pixelwise output value interpolation (Table3 in the paper)
    # It would be faster implemented with a parallel operation
    for c in range(img_a1.shape[0]):
        for y in range(img_a1.shape[1]):
            for x in range(img_a1.shape[2]):
                fa = fa_[c, y, x]
                fb = fb_[c, y, x]
                fc = fc_[c, y, x]
                fd = fd_[c, y, x]
                if fa > fb:
                    if fb > fc:
                        if fc > fd:
                            out[c, y, x] = (q - fa) * p0000[c, y, x] + (fa - fb) * p1000[c, y, x] + (fb - fc) * p1100[
                                c, y, x] + (fc - fd) * p1110[c, y, x] + (fd) * p1111[c, y, x]
                        elif fb > fd:
                            out[c, y, x] = (q - fa) * p0000[c, y, x] + (fa - fb) * p1000[c, y, x] + (fb - fd) * p1100[
                                c, y, x] + (fd - fc) * p1101[c, y, x] + (fc) * p1111[c, y, x]
                        elif fa > fd:
                            out[c, y, x] = (q - fa) * p0000[c, y, x] + (fa - fd) * p1000[c, y, x] + (fd - fb) * p1001[
                                c, y, x] + (fb - fc) * p1101[c, y, x] + (fc) * p1111[c, y, x]
                        else:
                            out[c, y, x] = (q - fd) * p0000[c, y, x] + (fd - fa) * p0001[c, y, x] + (fa - fb) * p1001[
                                c, y, x] + (fb - fc) * p1101[c, y, x] + (fc) * p1111[c, y, x]
                    elif fa > fc:
                        if fb > fd:
                            out[c, y, x] = (q - fa) * p0000[c, y, x] + (fa - fc) * p1000[c, y, x] + (fc - fb) * p1010[
                                c, y, x] + (fb - fd) * p1110[c, y, x] + (fd) * p1111[c, y, x]
                        elif fc > fd:
                            out[c, y, x] = (q - fa) * p0000[c, y, x] + (fa - fc) * p1000[c, y, x] + (fc - fd) * p1010[
                                c, y, x] + (fd - fb) * p1011[c, y, x] + (fb) * p1111[c, y, x]
                        elif fa > fd:
                            out[c, y, x] = (q - fa) * p0000[c, y, x] + (fa - fd) * p1000[c, y, x] + (fd - fc) * p1001[
                                c, y, x] + (fc - fb) * p1011[c, y, x] + (fb) * p1111[c, y, x]
                        else:
                            out[c, y, x] = (q - fd) * p0000[c, y, x] + (fd - fa) * p0001[c, y, x] + (fa - fc) * p1001[
                                c, y, x] + (fc - fb) * p1011[c, y, x] + (fb) * p1111[c, y, x]
                    else:
                        if fb > fd:
                            out[c, y, x] = (q - fc) * p0000[c, y, x] + (fc - fa) * p0010[c, y, x] + (fa - fb) * p1010[
                                c, y, x] + (fb - fd) * p1110[c, y, x] + (fd) * p1111[c, y, x]
                        elif fc > fd:
                            out[c, y, x] = (q - fc) * p0000[c, y, x] + (fc - fa) * p0010[c, y, x] + (fa - fd) * p1010[
                                c, y, x] + (fd - fb) * p1011[c, y, x] + (fb) * p1111[c, y, x]
                        elif fa > fd:
                            out[c, y, x] = (q - fc) * p0000[c, y, x] + (fc - fd) * p0010[c, y, x] + (fd - fa) * p0011[
                                c, y, x] + (fa - fb) * p1011[c, y, x] + (fb) * p1111[c, y, x]
                        else:
                            out[c, y, x] = (q - fd) * p0000[c, y, x] + (fd - fc) * p0001[c, y, x] + (fc - fa) * p0011[
                                c, y, x] + (fa - fb) * p1011[c, y, x] + (fb) * p1111[c, y, x]

                else:
                    if fa > fc:
                        if fc > fd:
                            out[c, y, x] = (q - fb) * p0000[c, y, x] + (fb - fa) * p0100[c, y, x] + (fa - fc) * p1100[
                                c, y, x] + (fc - fd) * p1110[c, y, x] + (fd) * p1111[c, y, x]
                        elif fa > fd:
                            out[c, y, x] = (q - fb) * p0000[c, y, x] + (fb - fa) * p0100[c, y, x] + (fa - fd) * p1100[
                                c, y, x] + (fd - fc) * p1101[c, y, x] + (fc) * p1111[c, y, x]
                        elif fb > fd:
                            out[c, y, x] = (q - fb) * p0000[c, y, x] + (fb - fd) * p0100[c, y, x] + (fd - fa) * p0101[
                                c, y, x] + (fa - fc) * p1101[c, y, x] + (fc) * p1111[c, y, x]
                        else:
                            out[c, y, x] = (q - fd) * p0000[c, y, x] + (fd - fb) * p0001[c, y, x] + (fb - fa) * p0101[
                                c, y, x] + (fa - fc) * p1101[c, y, x] + (fc) * p1111[c, y, x]
                    elif fb > fc:
                        if fa > fd:
                            out[c, y, x] = (q - fb) * p0000[c, y, x] + (fb - fc) * p0100[c, y, x] + (fc - fa) * p0110[
                                c, y, x] + (fa - fd) * p1110[c, y, x] + (fd) * p1111[c, y, x]
                        elif fc > fd:
                            out[c, y, x] = (q - fb) * p0000[c, y, x] + (fb - fc) * p0100[c, y, x] + (fc - fd) * p0110[
                                c, y, x] + (fd - fa) * p0111[c, y, x] + (fa) * p1111[c, y, x]
                        elif fb > fd:
                            out[c, y, x] = (q - fb) * p0000[c, y, x] + (fb - fd) * p0100[c, y, x] + (fd - fc) * p0101[
                                c, y, x] + (fc - fa) * p0111[c, y, x] + (fa) * p1111[c, y, x]
                        else:
                            out[c, y, x] = (q - fd) * p0000[c, y, x] + (fd - fb) * p0001[c, y, x] + (fb - fc) * p0101[
                                c, y, x] + (fc - fa) * p0111[c, y, x] + (fa) * p1111[c, y, x]
                    else:
                        if fa > fd:
                            out[c, y, x] = (q - fc) * p0000[c, y, x] + (fc - fb) * p0010[c, y, x] + (fb - fa) * p0110[
                                c, y, x] + (fa - fd) * p1110[c, y, x] + (fd) * p1111[c, y, x]
                        elif fb > fd:
                            out[c, y, x] = (q - fc) * p0000[c, y, x] + (fc - fb) * p0010[c, y, x] + (fb - fd) * p0110[
                                c, y, x] + (fd - fa) * p0111[c, y, x] + (fa) * p1111[c, y, x]
                        elif fc > fd:
                            out[c, y, x] = (q - fc) * p0000[c, y, x] + (fc - fd) * p0010[c, y, x] + (fd - fb) * p0011[
                                c, y, x] + (fb - fa) * p0111[c, y, x] + (fa) * p1111[c, y, x]
                        else:
                            out[c, y, x] = (q - fd) * p0000[c, y, x] + (fd - fc) * p0001[c, y, x] + (fc - fb) * p0011[
                                c, y, x] + (fb - fa) * p0111[c, y, x] + (fa) * p1111[c, y, x]
    # print(out.shape)
    # print(out.sum(3))
    # out = np.transpose(out, (0, 1, 3, 2, 4)).reshape(
    #     (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2]))
    # out = np.rot90(out, rot, axes=[1, 2])
    out = out / q
    return out