import torch
import torch.nn as nn
from torch import optim
from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

traindata = datasets.FashionMNIST('FashionMNIST', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(traindata, batch_size=64, shuffle=True)

testdata = datasets.FashionMNIST('FashionMNIST', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testdata, batch_size=64, shuffle=True)

i,l = next(iter(trainloader))
print(i.shape)


model = nn.Sequential(nn.Linear(784, 256),
                        nn.ReLU(),
                        nn.Linear(256, 64),
                        nn.ReLU(),
                        nn.Linear(64, 10),
                        nn.LogSoftmax(dim=1))

def accuracy():
    image, label = next(iter(testloader))
    with torch.no_grad():
        output = model(image.view(image.shape[0], -1))
        print(output.shape)
        output = torch.exp(output)

        top_p, output_class = output.topk(1, dim=1)

        equal = output_class == label.view(*output_class.shape)

        accuracy = torch.mean(equal.type(torch.FloatTensor))

        print(f"Accuracy : {accuracy.item() * 100}")
        
accuracy()

criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.003)

epochs = 5

for i in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        image = images.view(images.shape[0], -1)

        optimizer.zero_grad()

        output = model.forward(image)
        loss = criterion(output, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Running loss : {running_loss/len(trainloader)}")

accuracy() 
