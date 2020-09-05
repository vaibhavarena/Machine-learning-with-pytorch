import torch

def activation(x):
    return 1 / (1+torch.exp(-x))


torch.manual_seed(17)

features = torch.randn((1,5), dtype=torch.float)
weights = torch.randn_like(features)
bias = torch.randn((1,1))

# y = activation(torch.sum(features * weights) + bias)
y = activation(torch.mm(features, weights.view(5, 1)) + bias)   # torch.mm includes torch.sum and torch.dot

print(y)

# Multi-layer Neural Network

features_m = torch.randn(64, 5)

n_input = features_m.shape[1]
n_hidden = 2
n_output = 1

weight1 = torch.randn(features.shape[0] * n_input, n_hidden)
weight2 = torch.randn(n_hidden, n_output)

b1 = torch.randn(1, n_hidden)
b2 = torch.randn(1, n_output)

hidden1 = activation(torch.mm(features_m, weight1) + b1)
output = activation(torch.mm(hidden1, weight2) + b2)

print(output)

print(torch.exp(output).shape)
print(torch.sum(torch.exp(output), dim=1).view(-1, 1).shape)

def softmax(x):
    return torch.exp(x) / torch.sum(torch.exp(x), dim=1).view(-1, 1)

print(softmax(output))
