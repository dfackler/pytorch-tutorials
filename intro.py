from __future__ import print_function
import numpy as np
import torch

"""
WHAT IS PYTORCH?
"""

# create matrix - see output based on initialization
x = torch.empty(5, 3)
y = torch.rand(5, 3)

"""
print(x)

x = torch.rand(5, 3)
print(x)

x = torch.zeros(5, 3, dtype=torch.long)
print(x)

x = torch.tensor([5.5, 3])
print(x)

x = x.new_ones(5, 3, dtype=torch.double)  # new_* methods take in sizes
print(x)

x = torch.randn_like(x, dtype=torch.float)  # override dtype, same size result
print(x)
# operations

y = torch.rand(5, 3)
print(x + y)

print(torch.add(x, y))

result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)

y.add_(x)  # inplace (any operator with post-fixed _)
print(y)

# standard NumPy indexing
print(x[:, 1])

# resize using torch.view
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # size -1 inferred from other dimensions
print(x.size(), y.size(), z.size())

# for single element tensor use .item() to get as python number
x = torch.randn(1)
print(x)
print(x.item())

# converting torch tensor to numpy array
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)

a.add_(1)
print(a)
print(b)

# convert numpy array to torch tensor
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

# CUDA Tensors - using the .to method
print(torch.cuda.is_available())
if torch.cuda.is_available():
    device = torch.device("cuda")  # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create on GPU
    x = x.to(device)  # or use .to("cuda")
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))  # .to can also change dtype
"""
