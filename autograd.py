import torch

# requires_grad=True tracks computation of Tensor
x = torch.ones(2, 2, requires_grad=True)
print(x)

y = x + 2
print(y)

print(y.grad_fn)  # y result of operation, so it has a grad_fn

z = y * y * 3
out = z.mean()

print(z, out)

# .requires_grad() changes existing flag in-place. default to False
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.grad_fn)  # prints None
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)

# Gradients
print("Gradients\n")

print(out)
print(x.grad)
out.backward()  # backprop a single scalar
print(out)
print(x.grad)

# autograd essentially engine to compute vector-Jacobian product
x = torch.randn(3, requires_grad=True)
y = x * 2
print(y)
while y.data.norm() < 1000:
    y = y * 2
print(y)

# to get just vector-jacobian product use backward
v = torch.tensor([0.1, 1.0, .000001], dtype=torch.float)
y.backward(v)
print(x.grad)

# stop autograd tracking history on Tensors
print(x.requires_grad)
print((x**2).requires_grad)
with torch.no_grad():
    print((x**2).requires_grad)

print(x.requires_grad)
y = x.detach()
print(y.requires_grad)
print(x.eq(y).all())
print(y)
print(x)
