import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch import optim
from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

traindata = datasets.FashionMNIST('FashionMNIST', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(traindata, batch_size=64, shuffle=True)

testdata = datasets.FashionMNIST('FashionMNIST', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testdata, batch_size=64, shuffle=True)

i,l = next(iter(trainloader))


model = nn.Sequential(nn.Linear(784, 256),
                        nn.ReLU(),
                        nn.Dropout(p=0.2),
                        nn.Linear(256, 64),
                        nn.ReLU(),
                        nn.Dropout(p=0.2),
                        nn.Linear(64, 10),
                        nn.LogSoftmax(dim=1))

def accuracy(text = "Accuracy : "):
    image, label = next(iter(testloader))
    with torch.no_grad():
        output = model(image.view(image.shape[0], -1))
        output = torch.exp(output)

        top_p, output_class = output.topk(1, dim=1)

        equal = output_class == label.view(*output_class.shape)

        accuracy = torch.mean(equal.type(torch.FloatTensor))

        print(f"{text}{accuracy.item() * 100}")
        print()


def save_model(name='fashion.pth', input_layers=None, output_layers=None, model=model):
    checkpoint = {'input_layers':input_layers,
                'output_layers':output_layers,
                'hidden_layers':[i.out_features for i in model.hidden_layers],
                'state_dict':model.state_dict()}

    torch.save(checkpoint, name)


def load_model(name='fashion.pth', model=model):
    checkpoint = torch.load(name)

    input_layers = checkpoint['input_layers']
    output_layers = checkpoint['output_layers']
    hidden_layers = checkpoint['hidden_layers']

    state = torch.load(checkpoint['state_dict'])
    model.load_state_dict(state)


accuracy("Accuracy before training : ")

train = []
test = []
accu = []

criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.003)

epochs = 10

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
    

    else:
        test_loss = 0 
        accuracy = 0

        with torch.no_grad():
            model.eval()
            for images, labels in testloader:
                image = images.view(images.shape[0], -1)

                output = model(image)

                loss = criterion(output, labels)
                test_loss += loss.item()

                ps = torch.exp(output)
                top_p, top_class = ps.topk(1, dim=1)

                equals = top_class == labels.view(*top_class.shape)

                accuracy += torch.mean(equals.type(torch.FloatTensor))
        
            model.train()

            train.append(running_loss/len(trainloader))
            test.append(test_loss/len(testloader))
            accu.append(accuracy/len(testloader))

    print("Epochs : {}/{}".format(i+1, epochs))
    print(f"Training loss : {running_loss/len(trainloader)}")
    print(f"Test loss : {test_loss/len(testloader)}")
    print(f"Accuracy : {(accuracy/len(testloader)) * 100}")
    print()

save_model()

plt.plot(train, label='Training loss')
plt.plot(test, label='Test loss')
plt.plot(accu, label='Accuracy')
plt.legend(frameon=False)
plt.show()
