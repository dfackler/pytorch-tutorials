import torch
import torch.nn as nn
import torch.nn.functional as F

"""
A typical training procedure for a neural network is as follows:

Define the neural network that has some learnable parameters (or weights)
Iterate over a dataset of inputs
Process input through the network
Compute the loss (how far is the output from being correct)
Propagate gradients back into the network’s parameters
Update the weights of the network, typically using a simple update rule: 
    weight = weight - learning_rate * gradient
"""


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        # 1 input image channel, 6 output channels, 3x3 sq conv
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dim
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)

# learnable parameters return
params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight

# try random 32x32 input
input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)

# zero the gradient buffers of all parameters and backprops with rand grads
net.zero_grad()
out.backward(torch.randn(1, 10))

print(net)

"""
Recap:
torch.Tensor - A multi-dimensional array with support for autograd operations like backward(). Also holds the gradient w.r.t. the tensor.
nn.Module - Neural network module. Convenient way of encapsulating parameters, with helpers for moving them to GPU, exporting, loading, etc.
nn.Parameter - A kind of Tensor, that is automatically registered as a parameter when assigned as an attribute to a Module.
autograd.Function - Implements forward and backward definitions of an autograd operation. 
Every Tensor operation creates at least a single Function node that connects to functions that created a Tensor and encodes its history.
"""

"""
LOSS FUNCTION - computing loss and updating weights
"""

output = net(input)
target = torch.randn(10)  # dummy target
target = target.view(1, -1)  # same shape as output
criterion = nn.MSELoss()

print(output)
print(target)
loss = criterion(output, target)
print(loss)

# follow loss backward
print(loss.grad_fn)
print(loss.grad_fn.next_functions[0][0])  # linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])
