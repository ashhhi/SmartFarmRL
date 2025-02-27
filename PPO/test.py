import torch

t = torch.tensor([[1, 2], [3, 4]])
print(torch.gather(t, 1, torch.tensor([[0, 0]])))