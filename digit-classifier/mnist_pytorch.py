import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

BATCH_SIZE = 50
transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

def imshow(img):
    npimg = img.numpy()
    plt.imshow(npimg.transpose(1, 2, 0))
    plt.show()
images, labels = next(iter(trainloader))
imshow(torchvision.utils.make_grid(images))

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=24, kernel_size=5)
        self.d1 = nn.Linear(24 * 24 * 24, 112)
        self.d2 = nn.Linear(112, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = x.flatten(start_dim = 1)
        x = self.d1(x)
        x = F.relu(x)
        x = self.d2(x)
        x = F.softmax(x, dim=1)
        return x

def get_accuracy(y, target):
    corrects = (torch.max(y, 1).indices == target).sum()
    accuracy = 100.0 * corrects / target.shape[0]
    return accuracy.item()

ETA = 0.001
EPOCHS = 5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('using', device)
model = MyModel()
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=ETA)

for epoch in range(EPOCHS):
    train_loss = 0.0
    train_accu = 0.0
    model.train()
    for images, labels in trainloader:
        images = images.to(device)
        labels = labels.to(device)
        y = model(images)
        loss = criterion(y, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        with torch.no_grad():
            train_loss += loss.item() / len(trainset) * BATCH_SIZE
            train_accu += get_accuracy(y, labels) / len(trainset) * BATCH_SIZE
    test_accu = 0.0
    model.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            test_accu += get_accuracy(outputs, labels) / len(testset) * BATCH_SIZE
    print(f'Epoch: {epoch} | Loss: {train_loss:.6f} | Train Accuracy: {train_accu:.2f} | Test Accuracy: {test_accu:.2f}')