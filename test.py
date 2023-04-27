import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
x = torch.randn(10, 10)
x = x.to(device)
print(x)
