import torch


# 默认标签用one-hot编码
# 二分类交叉熵问题

def my_BCE(o, t):
    output1 = o
    output2 = 1 - o
    target1 = t
    target2 = 1 - t
    output1_log = torch.log(output1)
    output2_log = torch.log(output2)
    BCE_loss1 = -torch.sum(output1_log * target1 + output2_log * target2) / len(o.flatten())
    return BCE_loss1

# # case 1
# # 假设：output_1=[0.8 0.2 0.4], target_1 = [1 0 0]
# output = torch.tensor([0.8, 0.2, 0.4], dtype=torch.float32)
# target = torch.tensor([1, 0, 0], dtype=torch.float32)
# print("手搓二分类交叉熵为{}".format(my_BCE(output, target)))
# BCE_loss2 = torch.nn.BCELoss()
# print("PyTorch计算二分类交叉熵为{}".format(BCE_loss2(output, target)))

# case 2
# 假设 output_1=[[0.8, 0.2, 0.4], [0.5, 0.7, 0.2]], target_1=[[1, 0, 0], [0, 1, 0]]
# output = torch.tensor([[0.8, 0.2, 0.4], [0.5, 0.7, 0.2]], dtype=torch.float32)
# target = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.float32)
# print("手搓二分类交叉熵为{}".format(my_BCE(output, target)))
# BCE_loss2 = torch.nn.BCELoss()
# print("PyTorch计算二分类交叉熵为{}".format(BCE_loss2(output, target)))

# # case 3
# # 假设 output_1=[1.8, 2.2, 3.4], target_1=[1, 0, 0]
# output = torch.tensor([1.8, 2.2, 3.4], dtype=torch.float32)
# target = torch.tensor([1, 0, 0], dtype=torch.float32)
# print("手搓二分类交叉熵为{}".format(my_BCE(output, target)))
# BCE_loss2 = torch.nn.BCELoss()
# print("PyTorch计算二分类交叉熵为{}".format(BCE_loss2(output, target)))
# # 报错 all elements of input should be between 0 and 1, 应用sigmoid将output归一化
# # sigmoid = torch.sigmoid(output)
# # print("手搓二分类交叉熵为{}".format(my_BCE(sigmoid, target)))
# # print("PyTorch计算二分类交叉熵为{}".format(BCE_loss2(sigmoid, target)))

# # case 4
# # 为解决上述问题，也可以用BCEWITHLOGITSLOSS函数，该函数将torch.sigmoid与torch.nn.BCELoss集合起来
# output = torch.tensor([1.8, 2.2, 3.4], dtype=torch.float32)
# target = torch.tensor([1, 0, 0], dtype=torch.float32)
# BCEWITHLOGITSLOSS = torch.nn.BCEWithLogitsLoss()
# sigmoid = torch.sigmoid(output)
# print("手搓二分类交叉熵为{}".format(my_BCE(sigmoid, target)))
# print("PyTorch计算二分类交叉熵为{}".format(BCEWITHLOGITSLOSS(output, target)))
