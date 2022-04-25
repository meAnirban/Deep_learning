# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# Hyperparameters
in_channels = 3
num_classes = 10
learning_rate = 0.001
batch_size = 1024
num_epochs = 2


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x



def save_checkpoint(state, filename= 'checkpoint.pth.tar'):
    print("=> saving checkpoint")
    torch.save(state, filename)


# Load pretrain model 
model = torchvision.models.vgg16(pretrained =True)
    
for param in model.parameters():
    param.requires_grad = False


model.avgpool = Identity()
model.classifier = nn.Sequential(nn.Linear(512, 100), 
    nn.ReLU(),
    nn.Linear(100, num_classes))
model.to(device)


# Load data
train_dataset = datasets.CIFAR10(root = 'dataset/', train = True, transform = transforms.ToTensor(), download = True)
train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)

test_dataset = datasets.CIFAR10(root = 'dataset/', train = True, transform = transforms.ToTensor(), download = True)
test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = True)

# Initialize network
# model = CNN(in_channels, num_classes).to(device = device)

# Hyperparameters
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)


for epoch in range(num_epochs):
    losses = []

    if epoch % 2 == 0:
        checkpoint = {"state_dict" : model.state_dict(), 'optimizer': optimizer.state_dict()}
        save_checkpoint(checkpoint)

    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device = device)
        targets = targets.to(device = device)

        scores = model(data)
        loss = criterion(scores, targets)
        losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()

        # update weights
        optimizer.step()
    
    print(f'epoch = {epoch}; loss: {sum(losses) / len(losses)}')


def check_accuracy(loader, model):
    if loader.dataset.train:
        print("training accuracy")
    else:
        print("testing, accuracy")

    num_correct = 0
    num_samples = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device = device)
            y = y.to(device = device)
            scores = model(x)

            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'number of correct predictions / number of samples = {float(num_correct) * 100 / float(num_samples):.2f}')

    model.train()



check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
