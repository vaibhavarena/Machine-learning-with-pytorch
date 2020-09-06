import torch
import tqdm
import matplotlib.pyplot as plt
from collections import OrderedDict
from torch import nn, optim
from torchvision import datasets, transforms, models


def save_model(name='cats_dogs.pth'):
    checkpoint = {'state_dict':model.state_dict()}

    torch.save(checkpoint, name)

def load_model(name='cats_dogs.pth'):
    checkpoint = torch.load(name)

    state = torch.load(name)
    model.load_state_dict(state)
    
def accuracy(text="Accuracy : "):
    image, label = next(iter(testloader))
    output = model(image)

    ps = torch.exp(output)
    top_p, top_class = ps.topk(1, dim=1)

    equals = top_class == label.view(*top_class.shape)

    accuracy = torch.mean(equals.type(torch.FloatTensor))

    print(f'{text}{accuracy*100}')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                transforms.RandomRotation(30),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                                                                

test_transform = transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
                                                                                                

traindata = datasets.ImageFolder('dogs-vs-cats/train', transform=train_transform)
trainloader = torch.utils.data.DataLoader(traindata, batch_size=64, shuffle=True)

testdata = datasets.ImageFolder('dogs-vs-cats/test', transform=test_transform)
testloader = torch.utils.data.DataLoader(testdata, batch_size=64, shuffle=True)


model = models.densenet121(pretrained=True)

for param in model.parameters():
    param.requires_grad =False

classifier = nn.Sequential(OrderedDict([
    ('fc1',nn.Linear(1024, 500)),
    ('relu',nn.ReLU()),
    ('fc2',nn.Linear(500,2)),
    ('output',nn.LogSoftmax(dim=1))
]))

model.classifier = classifier

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)

accuracy("Accuracy before training : ")

model.to(device)

train = []
test = []
accu = []

epochs = 5

for i in range(epochs):
    train_loss = 0
    print(f'Epochs : {i+1}/{epochs}')

    for images, labels in tqdm.tqdm(trainloader):
        optimizer.zero_grad()
        
        images, labels = images.to(device), labels.to(device)

        output = model(images)
        loss = criterion(output, labels)

        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()

    else:
        with torch.no_grad():
            model.eval()
            test_loss = 0
            accuracy = 0
            
            for images, labels in tqdm.tqdm(testloader):
                images, labels = images.to(device), labels.to(device)

                output = model(images)
                loss = criterion(output, labels)

                test_loss += loss.item()

                ps = torch.exp(output)
                top_p, top_class = ps.topk(1, dim=1)

                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))
            
    save_model()
    
    train.append(train_loss)
    test.append(test_loss)
    accu.append(accuracy)

    print(f'Train loss : {train_loss/len(trainloader)}')
    print(f'Test loss : {test_loss/len(testloader)}')
    print(f'Accuracy : {accuracy * 100/len(testloader)}')
    print()

plt.plot(train, label="Training loss")
plt.plot(test, label="Test loss")
plt.plot(accu, label="Accuracy")
plt.legend(frameon=False)
plt.show()
