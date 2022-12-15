import matplotlib.pyplot as plt
import numpy as np

def paint_amap(acmap, num_action):
    image = np.asanyarray(acmap.squeeze(), dtype=np.uint8)
    plt.imshow(image, vmin=0, vmax=num_action)
    plt.colorbar()
    plt.show()
    # plt.pause(1)
    # plt.close()
"""
img_noisy [h, w]
LUT [N,]
q = 2**SAMPLING_INTERVAL
L = 2 ** (8 - SAMPLING_INTERVAL) + 1
"""
def transfer_lut(img_noisy, LUT, h, w, q, L):

    img_noisy = np.pad(img_noisy, ((1, 1), (1, 1)), mode='reflect')
    img_noisy = np.expand_dims(img_noisy, 0)

    img_a1 = img_noisy[:, 0:0 + h, 0:0 + w] // q
    img_b1 = img_noisy[:, 0:0 + h, 2:2 + w] // q
    img_c1 = img_noisy[:, 2:2 + h, 0:0 + w] // q
    img_d1 = img_noisy[:, 2:2 + h, 2:2 + w] // q

    out_action = LUT[img_a1.flatten().astype(np.int_) * L * L * L +
                     img_b1.flatten().astype(np.int_) * L * L +
                     img_c1.flatten().astype(np.int_) * L +
                     img_d1.flatten().astype(np.int_)]. \
        reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2]))
    return out_action

def Interp(weight, img_in, h, w, q, rot, SAMPLING_INTERVAL=4):
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
    pass

    # p0010 = weight[img_a1.flatten().astype(np.int_) * L * L * L +
    #                img_b1.flatten().astype(np.int_) * L * L +
    #                img_c2.flatten().astype(np.int_) * L +
    #                img_d1.flatten().astype(np.int_)].\
    #     reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    #
    # p0011 = weight[img_a1.flatten().astype(np.int_) * L * L * L +
    #                img_b1.flatten().astype(np.int_) * L * L +
    #                img_c2.flatten().astype(np.int_) * L +
    #                img_d2.flatten().astype(np.int_)].\
    #     reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    #
    # p0100 = weight[img_a1.flatten().astype(np.int_) * L * L * L +
    #                img_b2.flatten().astype(np.int_) * L * L +
    #                img_c1.flatten().astype(np.int_) * L +
    #                img_d1.flatten().astype(np.int_)].\
    #     reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    #
    # p0101 = weight[img_a1.flatten().astype(np.int_) * L * L * L +
    #                img_b2.flatten().astype(np.int_) * L * L +
    #                img_c1.flatten().astype(np.int_) * L +
    #                img_d2.flatten().astype(np.int_)].\
    #     reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    #
    # p0110 = weight[img_a1.flatten().astype(np.int_) * L * L * L +
    #                img_b2.flatten().astype(np.int_) * L * L +
    #                img_c2.flatten().astype(np.int_) * L +
    #                img_d1.flatten().astype(np.int_)].\
    #     reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    #
    # p0111 = weight[img_a1.flatten().astype(np.int_) * L * L * L +
    #                img_b2.flatten().astype(np.int_) * L * L +
    #                img_c2.flatten().astype(np.int_) * L +
    #                img_d2.flatten().astype(np.int_)].\
    #     reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    # # -----------------------------------------
    # p1000 = weight[img_a2.flatten().astype(np.int_) * L * L * L +
    #                img_b1.flatten().astype(np.int_) * L * L +
    #                img_c1.flatten().astype(np.int_) * L +
    #                img_d1.flatten().astype(np.int_)].\
    #     reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    #
    # p1001 = weight[img_a2.flatten().astype(np.int_) * L * L * L +
    #                img_b1.flatten().astype(np.int_) * L * L +
    #                img_c1.flatten().astype(np.int_) * L +
    #                img_d2.flatten().astype(np.int_)].\
    #     reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    #
    # p1010 = weight[img_a2.flatten().astype(np.int_) * L * L * L +
    #                img_b1.flatten().astype(np.int_) * L * L +
    #                img_c2.flatten().astype(np.int_) * L +
    #                img_d1.flatten().astype(np.int_)].\
    #     reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    #
    # p1011 = weight[img_a2.flatten().astype(np.int_) * L * L * L +
    #                img_b1.flatten().astype(np.int_) * L * L +
    #                img_c2.flatten().astype(np.int_) * L +
    #                img_d2.flatten().astype(np.int_)].\
    #     reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    #
    # p1100 = weight[img_a2.flatten().astype(np.int_) * L * L * L +
    #                img_b2.flatten().astype(np.int_) * L * L +
    #                img_c1.flatten().astype(np.int_) * L +
    #                img_d1.flatten().astype(np.int_)].\
    #     reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    #
    # p1101 = weight[img_a2.flatten().astype(np.int_) * L * L * L +
    #                img_b2.flatten().astype(np.int_) * L * L +
    #                img_c1.flatten().astype(np.int_) * L +
    #                img_d2.flatten().astype(np.int_)].\
    #     reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    #
    # p1110 = weight[img_a2.flatten().astype(np.int_) * L * L * L +
    #                img_b2.flatten().astype(np.int_) * L * L +
    #                img_c2.flatten().astype(np.int_) * L +
    #                img_d1.flatten().astype(np.int_)].\
    #     reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
    #
    # p1111 = weight[img_a2.flatten().astype(np.int_) * L * L * L +
    #                img_b2.flatten().astype(np.int_) * L * L +
    #                img_c2.flatten().astype(np.int_) * L +
    #                img_d2.flatten().astype(np.int_)].\
    #     reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))





# test
# if __name__ == '__main__':
#
#
#     # Rotational ensemble
#     img_in = np.pad(img_lr, ((0, 1), (0, 1), (0, 0)), mode='reflect').transpose((2, 0, 1))
#     out_r0 = FourSimplexInterp(LUT, img_in, h, w, q, 0, upscale=UPSCALE)








