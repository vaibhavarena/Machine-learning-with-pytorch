import torch
import torch.nn as nn
import torch.nn.functional as F

# class Classifier(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.hidden = nn.Linear(784, 128)
#         self.hidden1 = nn.Linear(128, 64)
#         self.output = nn.Linear(64, 10)

#     def forward(self, x):
#         x = F.sigmoid(self.hidden(x))
#         x = F.sigmoid(self.hidden1(x))
#         x = F.softmax(self.output(x))

#         return x

# model = Classifier()
# print(model)

model = nn.Sequential(nn.Linear(784, 256),
                        nn.ReLU(),
                        nn.Linear(256, 64),
                        nn.ReLU(),
                        nn.Linear(64, 10),
                        nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()

images = images.view(images.shape[0], -1)
logits = model(images)
loss = criterion(logits, labels)

model[0].weight.grad
loss.backward()
model[0].weight.grad