from itertools import dropwhile

import torch
from torch.distributions import Bernoulli

# Train_flag = None
# activations = torch.rand((5, 5))
# activations = torch.tensor([[1, 2, 3, 4, 5], [3, 4, 5, 7, 8], [2, 4, 6, 7, 1]], dtype=torch.float32)
# m = Bernoulli(probs=0.5)
# mask = m.sample(activations.shape)
# if Train_flag:
#     activations_dropout = activations * mask
# else:
#     activations_dropout = activations
# print(activations_dropout)
activations = torch.rand((5, 5))
dropout = torch.nn.Dropout(p=0.5)
dropout.train()
activations1 = dropout(activations)
dropout.eval()
activations2 = dropout(activations)

print(activations1)
print(activations2)
