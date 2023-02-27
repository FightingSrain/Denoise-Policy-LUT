import numpy as np
import matplotlib.pyplot as plt


import pandas as pd
import os



def survey(results, category_names):
    """
    Parameters
    ----------
    results : dict
        A mapping from question labels to a list of answers per category.
        It is assumed all lists contain the same number of entries and that
        it matches the length of *category_names*.
    category_names : list of str
        The category labels.
    """
    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.colormaps['RdYlGn'](
        np.linspace(0.15, 0.85, data.shape[1]))

    fig, ax = plt.subplots(figsize=(9.2, 5))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        print((colname, color))
        widths = data[:, i]
        starts = data_cum[:, i] - widths

        rects = ax.barh(labels, widths, left=starts, height=0.5,
                        label=colname, color=color)

        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        # ax.bar_label(rects, label_type='center', color=text_color)
    ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),
              loc='lower left', fontsize=8)

    return fig, ax


# survey(results, category_names)
# plt.show()



def sigmoid(x):
    return 1/(1+np.exp(-x))
def Softmax(x, dim=0):
    return np.exp(x) / np.sum(np.exp(x), axis=dim, keepdims=True)

def savevaltocsv(records, filename, sigma):
    col_name = ['n_epi', 'val_psnr', 'val_ssim']
    # records = [['001', '小明', 18]]
    # 先转为DataFrame格式
    df = pd.DataFrame(columns=col_name, data=records)
    # index=False表示存储csv时没有默认的id索引
    # 如果文件不存在，则创建文件，如果文件存在，则追加内容
    df.to_csv("./Val_PSNR_and_SSIM_{}/{}".format(sigma, filename), encoding='utf-8', index=False)