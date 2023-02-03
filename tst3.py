import torch.nn.functional as F




import torch
import numpy as np
# ins = torch.Tensor([[[[3, 2, 1],
#                     [1, 3, 2],
#                     [1, 2, 3]]]])
# res = F.pad(ins, (0, 1, 0, 1), mode='reflect')
# print(res)

# res2 = torch.rot90(ins, 0, [2,3])
# print(res2)

# ins2 = ins.numpy()[0][0]
# res3 = np.pad(ins2, ((0, 2), (0, 2)), mode='constant')
# print(res3.shape)
# print(res3)


# ins = torch.Tensor([[[[3, 2, 1],
#                     [1, 3, 2],
#                     [1, 2, 3]]]])
# 给定一个不含重复数字的数组 nums ，返回其 所有可能的全排列
def permute(nums):
    res = []
    def backtrack(nums, tmp):
        if not nums:
            res.append(tmp)
            return
        for i in range(len(nums)):
            backtrack(nums[:i] + nums[i+1:], tmp + [nums[i]])
    backtrack(nums, [])
    return res
print(permute([1,2,3]))








