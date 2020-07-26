import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim

def imshow(images):
    images = images * 0.5 + 0.5
    npimg = images.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

traindata = datasets.CIFAR10('data', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(traindata, batch_size=4, shuffle=True)

testdata = datasets.CIFAR10('data', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testdata, batch_size=4, shuffle=True)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

dataiter = iter(trainloader)
images, labels = dataiter.next()

print(" ".join( classes[labels[j]] for j in range(4)))
# imshow(torchvision.utils.make_grid(images))

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(5 * 5 * 16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(-1, 16 * 5 * 5)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

net = Net()
epochs = 2
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.03)

for i in range(epochs):
    running_loss = 0.0
    min_loss = 10
    for j, data in enumerate(trainloader):

        optimizer.zero_grad()
        
        inputs, labels = data
        output = net(inputs)

        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    print(f"[Epochs : {i}] -> Running loss : {running_loss/2000}")
    if(min_loss > running_loss):
        min_loss = running_loss
        torch.save(net.state_dict(), 'model.pth')

