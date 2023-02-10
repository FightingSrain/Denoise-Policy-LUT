import numpy as np

import matplotlib.pylab as plt

x = [1, 2, 3, 4, 5]
y1 = [28.6404, 28.6172, 28.6113, 28.6254, 28.5129]
y2 = [28.6194, 28.6346, 28.6356, 28.7195, 28.8493]
y3 = [29.2120, 29.2120, 29.1819, 29.2141, 29.2801]
#maker是设置折点的样式	markersize是设置结点大小，后面两个参数分别设置折点内部
# plt.plot(x, y1, marker='o',
#          color='blue',
#          markersize=8,
#          markerfacecolor='red',
#          markeredgecolor='red')
# plt.plot(x, y2, marker='*',
#          color='blue',
#          markersize=8,
#          markerfacecolor='red',
#          markeredgecolor='red')
plt.plot(x, y3, marker='o',
         color='blue',
         markersize=8,
         markerfacecolor='red',
         markeredgecolor='red')
plt.xlim(0, 6)
plt.grid()
plt.show()