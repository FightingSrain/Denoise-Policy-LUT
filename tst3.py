import torch.nn.functional as F




import torch

ins = torch.Tensor([[[[3, 2, 1],
                    [1, 3, 2],
                    [1, 2, 3]]]])
# res = F.pad(ins, (0, 1, 0, 1), mode='reflect')
# print(res)

res2 = torch.rot90(ins, 2, [2,3])
print(res2)



