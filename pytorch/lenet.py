import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


class LeNet(nn.Module):
    def __init__(self, in_channels = 1, num_classes =10):
        super(LeNet, self).__init__()
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(kernel_size = (2,2), stride = (2,2))
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = 6, kernel_size = (4, 4), stride = (1,1), padding = (0,0))
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = (4, 4), stride = (1,1), padding = (0,0))
        self.conv3 = nn.Conv2d(in_channels = 16, out_channels = 120, kernel_size = (4, 4), stride = (1,1), padding = (0,0))
        self.linear1 = nn.Linear(120, 84)
        self.linear2 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x



def save_checkpoint(state, filename= 'checkpoint.pth.tar'):
    print("=> saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint):
    print("=> loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# Hyperparameters
in_channels = 1
num_classes = 10
learning_rate = 0.001
batch_size = 1024
num_epochs = 5
load_model = True

# Load data
train_dataset = datasets.MNIST(root = 'dataset/', train = True, transform = transforms.ToTensor(), download = True)
train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)

test_dataset = datasets.MNIST(root = 'dataset/', train = True, transform = transforms.ToTensor(), download = True)
test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = True)

# Initialize network
model = LeNet(in_channels, num_classes).to(device = device)

# Hyperparameters
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

#if load_model:
#    load_checkpoint(torch.load('checkpoint.pth.tar'))

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
    model.eval()

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


