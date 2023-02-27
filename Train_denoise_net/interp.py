
import numpy as np
from numba import jit


# 4D equivalent of triangular interpolation
# @jit(nopython=True)
def FourSimplexInterp(weight, img_in, h, w, q, L, mode):
    # L = 2 ** (8 - SAMPLING_INTERVAL) + 1

    if mode == "a":
        # Extract MSBs
        img_a1 = img_in[:, 0:0 + h, 0:0 + w] // q
        img_b1 = img_in[:, 0:0 + h, 1:1 + w] // q
        img_c1 = img_in[:, 1:1 + h, 0:0 + w] // q
        img_d1 = img_in[:, 1:1 + h, 1:1 + w] // q

        # Extract LSBs
        fa = img_in[:, 0:0 + h, 0:0 + w] % q
        fb = img_in[:, 0:0 + h, 1:1 + w] % q
        fc = img_in[:, 1:1 + h, 0:0 + w] % q
        fd = img_in[:, 1:1 + h, 1:1 + w] % q

    elif mode == 'b':
        img_a1 = img_in[:, 0:0 + h, 0:0 + w] // q
        img_b1 = img_in[:, 0:0 + h, 2:2 + w] // q
        img_c1 = img_in[:, 2:2 + h, 0:0 + w] // q
        img_d1 = img_in[:, 2:2 + h, 2:2 + w] // q

        fa = img_in[:, 0:0 + h, 0:0 + w] % q
        fb = img_in[:, 0:0 + h, 2:2 + w] % q
        fc = img_in[:, 2:2 + h, 0:0 + w] % q
        fd = img_in[:, 2:2 + h, 2:2 + w] % q

    else: # mode == 'c':
        img_a1 = img_in[:, 0:0 + h, 0:0 + w] // q
        img_b1 = img_in[:, 1:1 + h, 1:1 + w] // q
        img_c1 = img_in[:, 1:1 + h, 2:2 + w] // q
        img_d1 = img_in[:, 2:2 + h, 1:1 + w] // q

        fa = img_in[:, 0:0 + h, 0:0 + w] % q
        fb = img_in[:, 1:1 + h, 1:1 + w] % q
        fc = img_in[:, 1:1 + h, 2:2 + w] % q
        fd = img_in[:, 2:2 + h, 1:1 + w] % q
    img_a2 = img_a1 + 1
    img_b2 = img_b1 + 1
    img_c2 = img_c1 + 1
    img_d2 = img_d1 + 1

    p0000 = weight[img_a1.flatten().astype(np.int_) * L * L * L + img_b1.flatten().astype(
        np.int_) * L * L + img_c1.flatten().astype(np.int_) * L + img_d1.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2]))
    p0001 = weight[img_a1.flatten().astype(np.int_) * L * L * L + img_b1.flatten().astype(
        np.int_) * L * L + img_c1.flatten().astype(np.int_) * L + img_d2.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2]))
    p0010 = weight[img_a1.flatten().astype(np.int_) * L * L * L + img_b1.flatten().astype(
        np.int_) * L * L + img_c2.flatten().astype(np.int_) * L + img_d1.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2]))
    p0011 = weight[img_a1.flatten().astype(np.int_) * L * L * L + img_b1.flatten().astype(
        np.int_) * L * L + img_c2.flatten().astype(np.int_) * L + img_d2.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2]))
    p0100 = weight[img_a1.flatten().astype(np.int_) * L * L * L + img_b2.flatten().astype(
        np.int_) * L * L + img_c1.flatten().astype(np.int_) * L + img_d1.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2]))
    p0101 = weight[img_a1.flatten().astype(np.int_) * L * L * L + img_b2.flatten().astype(
        np.int_) * L * L + img_c1.flatten().astype(np.int_) * L + img_d2.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2]))
    p0110 = weight[img_a1.flatten().astype(np.int_) * L * L * L + img_b2.flatten().astype(
        np.int_) * L * L + img_c2.flatten().astype(np.int_) * L + img_d1.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2]))
    p0111 = weight[img_a1.flatten().astype(np.int_) * L * L * L + img_b2.flatten().astype(
        np.int_) * L * L + img_c2.flatten().astype(np.int_) * L + img_d2.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2]))

    p1000 = weight[img_a2.flatten().astype(np.int_) * L * L * L + img_b1.flatten().astype(
        np.int_) * L * L + img_c1.flatten().astype(np.int_) * L + img_d1.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2]))
    p1001 = weight[img_a2.flatten().astype(np.int_) * L * L * L + img_b1.flatten().astype(
        np.int_) * L * L + img_c1.flatten().astype(np.int_) * L + img_d2.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2]))
    p1010 = weight[img_a2.flatten().astype(np.int_) * L * L * L + img_b1.flatten().astype(
        np.int_) * L * L + img_c2.flatten().astype(np.int_) * L + img_d1.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2]))
    p1011 = weight[img_a2.flatten().astype(np.int_) * L * L * L + img_b1.flatten().astype(
        np.int_) * L * L + img_c2.flatten().astype(np.int_) * L + img_d2.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2]))
    p1100 = weight[img_a2.flatten().astype(np.int_) * L * L * L + img_b2.flatten().astype(
        np.int_) * L * L + img_c1.flatten().astype(np.int_) * L + img_d1.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2]))
    p1101 = weight[img_a2.flatten().astype(np.int_) * L * L * L + img_b2.flatten().astype(
        np.int_) * L * L + img_c1.flatten().astype(np.int_) * L + img_d2.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2]))
    p1110 = weight[img_a2.flatten().astype(np.int_) * L * L * L + img_b2.flatten().astype(
        np.int_) * L * L + img_c2.flatten().astype(np.int_) * L + img_d1.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2]))
    p1111 = weight[img_a2.flatten().astype(np.int_) * L * L * L + img_b2.flatten().astype(
        np.int_) * L * L + img_c2.flatten().astype(np.int_) * L + img_d2.flatten().astype(np.int_)].reshape(
        (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2]))
    # Output image holder
    out = np.zeros((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2]))
    sz = img_a1.shape[0] * img_a1.shape[1] * img_a1.shape[2]
    out = out.reshape(sz, -1)
    # Output image holder
    # out = np.zeros((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2]))

    # Naive pixelwise output value interpolation (Table3 in the paper)
    # It would be faster implemented with a parallel operation
    p0000 = p0000.reshape(sz, -1)
    p0100 = p0100.reshape(sz, -1)
    p1000 = p1000.reshape(sz, -1)
    p1100 = p1100.reshape(sz, -1)
    fa = fa.reshape(-1, 1)

    p0001 = p0001.reshape(sz, -1)
    p0101 = p0101.reshape(sz, -1)
    p1001 = p1001.reshape(sz, -1)
    p1101 = p1101.reshape(sz, -1)
    fb = fb.reshape(-1, 1)
    fc = fc.reshape(-1, 1)

    p0010 = p0010.reshape(sz, -1)
    p0110 = p0110.reshape(sz, -1)
    p1010 = p1010.reshape(sz, -1)
    p1110 = p1110.reshape(sz, -1)
    fd = fd.reshape(-1, 1)

    p0011 = p0011.reshape(sz, -1)
    p0111 = p0111.reshape(sz, -1)
    p1011 = p1011.reshape(sz, -1)
    p1111 = p1111.reshape(sz, -1)

    fab = fa > fb;
    fac = fa > fc;
    fad = fa > fd

    fbc = fb > fc;
    fbd = fb > fd;
    fcd = fc > fd

    i1 = i = np.logical_and.reduce((fab, fbc, fcd)).squeeze(1)
    out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * p1110[
        i] + (fd[i]) * p1111[i]
    i2 = i = np.logical_and.reduce((~i1[:, None], fab, fbc, fbd)).squeeze(1)
    out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * p1101[
        i] + (fc[i]) * p1111[i]
    i3 = i = np.logical_and.reduce((~i1[:, None], ~i2[:, None], fab, fbc, fad)).squeeze(1)
    out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * p1101[
        i] + (fc[i]) * p1111[i]
    i4 = i = np.logical_and.reduce((~i1[:, None], ~i2[:, None], ~i3[:, None], fab, fbc)).squeeze(1)

    out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * p1101[
        i] + (fc[i]) * p1111[i]

    i5 = i = np.logical_and.reduce((~(fbc), fab, fac, fbd)).squeeze(1)
    out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * p1110[
        i] + (fd[i]) * p1111[i]
    i6 = i = np.logical_and.reduce((~(fbc), ~i5[:, None], fab, fac, fcd)).squeeze(1)
    out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * p1011[
        i] + (fb[i]) * p1111[i]
    i7 = i = np.logical_and.reduce((~(fbc), ~i5[:, None], ~i6[:, None], fab, fac, fad)).squeeze(1)
    out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * p1011[
        i] + (fb[i]) * p1111[i]
    i8 = i = np.logical_and.reduce((~(fbc), ~i5[:, None], ~i6[:, None], ~i7[:, None], fab, fac)).squeeze(1)
    out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * p1011[
        i] + (fb[i]) * p1111[i]

    i9 = i = np.logical_and.reduce((~(fbc), ~(fac), fab, fbd)).squeeze(1)
    out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * p1110[
        i] + (fd[i]) * p1111[i]
    # Fix the overflow bug in SR-LUT's implementation, should compare fd with fa first!
    # i10 = i = np.logical_and.reduce((~(fbc), ~(fac), ~i9[:,None], fab, fcd)).squeeze(1)
    # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fa[i]) * p0010[i] + (fa[i]-fd[i]) * p1010[i] + (fd[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
    # i11 = i = np.logical_and.reduce((~(fbc), ~(fac), ~i9[:,None], ~i10[:,None], fab, fad)).squeeze(1)
    # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fd[i]) * p0010[i] + (fd[i]-fa[i]) * p0011[i] + (fa[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
    i10 = i = np.logical_and.reduce((~(fbc), ~(fac), ~i9[:, None], fab, fad)).squeeze(1)  # c > a > d > b
    out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * p1011[
        i] + (fb[i]) * p1111[i]
    i11 = i = np.logical_and.reduce((~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], fab, fcd)).squeeze(1)  # c > d > a > b
    out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * p1011[
        i] + (fb[i]) * p1111[i]
    i12 = i = np.logical_and.reduce((~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], ~i11[:, None], fab)).squeeze(1)
    out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * p1011[
        i] + (fb[i]) * p1111[i]

    i13 = i = np.logical_and.reduce((~(fab), fac, fcd)).squeeze(1)
    out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * p1110[
        i] + (fd[i]) * p1111[i]
    i14 = i = np.logical_and.reduce((~(fab), ~i13[:, None], fac, fad)).squeeze(1)
    out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * p1101[
        i] + (fc[i]) * p1111[i]
    i15 = i = np.logical_and.reduce((~(fab), ~i13[:, None], ~i14[:, None], fac, fbd)).squeeze(1)
    out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * p1101[
        i] + (fc[i]) * p1111[i]
    i16 = i = np.logical_and.reduce((~(fab), ~i13[:, None], ~i14[:, None], ~i15[:, None], fac)).squeeze(1)
    out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * p1101[
        i] + (fc[i]) * p1111[i]

    i17 = i = np.logical_and.reduce((~(fab), ~(fac), fbc, fad)).squeeze(1)
    out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * p1110[
        i] + (fd[i]) * p1111[i]
    i18 = i = np.logical_and.reduce((~(fab), ~(fac), ~i17[:, None], fbc, fcd)).squeeze(1)
    out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * p0111[
        i] + (fa[i]) * p1111[i]
    i19 = i = np.logical_and.reduce((~(fab), ~(fac), ~i17[:, None], ~i18[:, None], fbc, fbd)).squeeze(1)
    out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * p0111[
        i] + (fa[i]) * p1111[i]
    i20 = i = np.logical_and.reduce((~(fab), ~(fac), ~i17[:, None], ~i18[:, None], ~i19[:, None], fbc)).squeeze(1)
    out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * p0111[
        i] + (fa[i]) * p1111[i]

    i21 = i = np.logical_and.reduce((~(fab), ~(fac), ~(fbc), fad)).squeeze(1)
    out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * p1110[
        i] + (fd[i]) * p1111[i]
    i22 = i = np.logical_and.reduce((~(fab), ~(fac), ~(fbc), ~i21[:, None], fbd)).squeeze(1)
    out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * p0111[
        i] + (fa[i]) * p1111[i]
    i23 = i = np.logical_and.reduce((~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], fcd)).squeeze(1)
    out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * p0111[
        i] + (fa[i]) * p1111[i]
    i24 = i = np.logical_and.reduce((~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], ~i23[:, None])).squeeze(1)
    out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * p0111[
        i] + (fa[i]) * p1111[i]
    out = out.reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2]))
    # print(out.sum(3))
    # out = np.transpose(out, (0, 1, 3, 2, 4)).reshape(
    #     (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2]))
    # out = np.rot90(out, rot, axes=[1, 2])
    out = out / q
    return out.transpose(1, 2, 0)