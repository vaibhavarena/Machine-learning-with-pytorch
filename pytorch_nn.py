import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch import optim

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

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = datasets.MNIST('MNIST', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

model = nn.Sequential(nn.Linear(784, 256),
                        nn.ReLU(),
                        nn.Linear(256, 64),
                        nn.ReLU(),
                        nn.Linear(64, 10),
                        nn.LogSoftmax(dim=1))

epochs = 5
optimizer = optim.SGD(model.parameters(), lr=0.003)
criterion = nn.NLLLoss()

for i in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        image = images.view(images.shape[0], -1)
        optimizer.zero_grad()

        logits = model.forward(image)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    print(f'Running Loss : {running_loss/64}')