import torch
import torch.nn.functional as F


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
# # 假设：output_1=[[0.8], [0.2], [0.6]], target_1 = [[1], [0], [0]],
# # 三样本输入，每个样本可能预测为正类或者负类，标签采用one-hot简写
# output = torch.tensor([[0.8, 0.2], [0.2, 0.8], [0.6, 0.4]], dtype=torch.float32)
# target = torch.tensor([[1, 0], [0, 1], [0, 1]], dtype=torch.float32)
# target_1 = torch.tensor([0, 1, 1], dtype=torch.float32)
# print("手搓二分类交叉熵为{}".format(my_BCE(output, target)))
# BCE_loss2 = torch.nn.BCELoss()
# print("PyTorch计算二分类交叉熵为{}".format(BCE_loss2(output, target)))
# # print("PyTorch计算二分类交叉熵为{}".format(BCE_loss2(output, target_1))) 报错，接受one-hot

# case 2
# # 假设 output_1=[[1.8], [2.2], [3.4]], target_1=[[1], [0], [0]]
output = torch.tensor([[1.8], [2.2], [3.4]], dtype=torch.float32)
target = torch.tensor([[1], [0], [0]], dtype=torch.float32)
# print("手搓二分类交叉熵为{}".format(my_BCE(output, target)))
BCE_loss2 = torch.nn.BCELoss()
# print("PyTorch计算二分类交叉熵为{}".format(BCE_loss2(output, target)))
# 报错 all elements of input should be between 0 and 1, 应用sigmoid将output归一化
print("手搓二分类交叉熵为{}".format(my_BCE(torch.sigmoid(output), target)))
print("PyTorch计算二分类交叉熵为{}".format(BCE_loss2(torch.sigmoid(output), target)))

# # case 3
# # 为解决上述问题，也可以用BCEWITHLOGITSLOSS函数，该函数将torch.sigmoid与torch.nn.BCELoss集合起来
# output = torch.tensor([[1.8], [2.2], [3.4]], dtype=torch.float32)
# target = torch.tensor([[1], [0], [0]], dtype=torch.float32)
# BCEWITHLOGITSLOSS = torch.nn.BCEWithLogitsLoss()
# sigmoid = torch.sigmoid(output)
# print("手搓二分类交叉熵为{}".format(my_BCE(sigmoid, target)))
# print("PyTorch计算二分类交叉熵为{}".format(BCEWITHLOGITSLOSS(output, target)))


# 多分类问题
def my_CEL(o, t):
    return -torch.sum(o * t) / len(o)


# case 1
# # 假设 output_1=[[0.8, 0.2, 0.4], [0.5, 0.7, 0.2]], target_1=[[1, 0, 0], [0, 1, 0]]
# # 两样本，每个样本有三个预测，对应三类，采用one-hot编码
# output = torch.tensor([[0.4, 0.2, 0.4], [0.5, 0.3, 0.2]], dtype=torch.float32)
# target = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.float32)  # 手搓时的one-hot编码
# target_1 = torch.LongTensor([0, 1])  # torch.nn.NLLLoss()可以自动进行one-hot编码操作，只接受标签值。
# print("手搓三分类交叉熵为{}".format(my_CEL(torch.log(output), target)))
# # CEL_loss2 = torch.nn.NLLLoss()
# CEL_loss2 = F.nll_loss
# print("PyTorch计算三分类交叉熵为{}".format(CEL_loss2(torch.log(output), target_1)))


# case 2
# # 假设 output_1=[[0.8, 0.2, 0.4], [0.5, 0.7, 0.2]], target_1=[[1, 0, 0], [0, 1, 0]]
# # 两样本，每个样本有三个预测，对应三类，采用one-hot编码
# # 手搓
# output = torch.tensor([[1.8, 2.2, 3.4], [2.5, 1.7, 2.2]], dtype=torch.float32)
# target = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.float32)  # 手搓时的one-hot编码
# print("手搓三分类交叉熵为{}".format(my_CEL(torch.log_softmax(output, dim=-1), target)))
# # 采用nll_loss
# target_1 = torch.LongTensor([0, 1])  # torch.nn.NLLLoss()可以自动进行one-hot编码操作，只接受标签值。
# CEL_loss2 = F.nll_loss  # F.nll_loss 包括了softmax 和 交叉熵的计算, 不包含log
# # CEL_loss2 = torch.nn.NLLLoss()
# print("PyTorch计算三分类交叉熵为{}".format(CEL_loss2(output, target_1)))
# print("PyTorch计算三分类交叉熵为{}".format(CEL_loss2(torch.log_softmax(output, dim=-1), target_1)))
# # 采用CrossEntropyLoss
# CEL_loss3 = torch.nn.CrossEntropyLoss()
# print("PyTorch计算CEL为{}".format(CEL_loss3(output, target_1)))
