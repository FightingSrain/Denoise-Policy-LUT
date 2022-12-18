#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
from sklearn import linear_model
import scipy.misc
from matplotlib import pyplot as plt
import cv2
import time

class KSVD(object):
    def __init__(self, n_components, max_iter=30, tol=5000,
                 n_nonzero_coefs=None):
        """
        稀疏模型Y = DX，Y为样本矩阵，使用KSVD动态更新字典矩阵D和稀疏矩阵X
        :param n_components: 字典所含原子个数（字典的列数）
        :param max_iter: 最大迭代次数
        :param tol: 稀疏表示结果的容差
        :param n_nonzero_coefs: 稀疏度
        """
        self.dictionary = None
        self.sparsecode = None
        self.max_iter = max_iter
        self.tol = tol
        self.n_components = n_components
        self.n_nonzero_coefs = n_nonzero_coefs

    def _initialize(self, y):
        """
        初始化字典矩阵
        """
        u, s, v = np.linalg.svd(y)
        self.dictionary = u[:, :self.n_components]
        print(self.dictionary.shape)

    def _update_dict(self, y, d, x):
        """
        使用KSVD更新字典的过程
        """
        for i in range(self.n_components):
            index = np.nonzero(x[i, :])[0]
            if len(index) == 0:
                continue

            d[:, i] = 0
            r = (y - np.dot(d, x))[:, index]
            u, s, v = np.linalg.svd(r, full_matrices=False)
            d[:, i] = u[:, 0].T
            x[i, index] = s[0] * v[0, :]
        return d, x

    def fit(self, y):
        """
        KSVD迭代过程
        """
        self._initialize(y)
        for i in range(self.max_iter):
            x = linear_model.orthogonal_mp(self.dictionary, y, n_nonzero_coefs=self.n_nonzero_coefs)
            e = np.linalg.norm(y - np.dot(self.dictionary, x))
            if e < self.tol:
                break
            self._update_dict(y, self.dictionary, x)

        self.sparsecode = linear_model.orthogonal_mp(self.dictionary, y, n_nonzero_coefs=self.n_nonzero_coefs)
        return self.dictionary, self.sparsecode


if __name__ == '__main__':
    im_ascent = cv2.imread("./img_tst/test001.png", 0).astype(np.float)
    # raw_n = np.random.normal(0, 15, im_ascent.shape).astype(np.float32) / 255.
    # imgs = np.clip(im_ascent / 255. + raw_n, a_min=0., a_max=1.)
    # im_ascent = (im_ascent * 255).astype(np.uint8)
    print(im_ascent.shape)
    t1 = time.time()
    ksvd = KSVD(300)
    dictionary, sparsecode = ksvd.fit(im_ascent)

    output = dictionary.dot(sparsecode)
    t2 = time.time()

    print("time: ", t2 - t1)
    output = np.clip(output, 0, 255)
    cv2.imwrite("./output.png", output.astype(np.uint8))