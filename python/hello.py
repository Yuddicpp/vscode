import torch 
from torch import nn

print("torch version:", torch.__version__)

a = torch.tensor([[2,1]])
b = torch.tensor([[-1,2]])
c = a@b.t()
print("[[2,1]]@[[-1],[2]] =", c.item())