# https://deeplizard.com/learn/video/0VCOG8IeVf8
import torch
import torch.nn as nn
import torch.nn.functional as F
import os.path

# Optimizer to update the weights
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
import pdb
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from resources.plotcm import plot_confusion_matrix

torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True)  # Already on by default


SAVE_PATH = './25.pyt'

print(torch.__version__)
print(torchvision.__version__)


def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        self.fc1 = nn.Linear(in_features=12 * 4 * 4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)

    def forward(self, t):
        # (1) input layer
        t = t

        # (2) hidden conv layer
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # (3) hidden conv layer
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # (4) hidden linear layer
        t = t.reshape(-1, 12 * 4 * 4)
        t = self.fc1(t)
        t = F.relu(t)

        # (5) hidden linear layer
        t = self.fc2(t)
        t = F.relu(t)

        # (6) output layer
        t = self.out(t)
        return t


train_set = torchvision.datasets.FashionMNIST(
    root='./data/FashionMNIST',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor()
    ])
)

test_set = torchvision.datasets.FashionMNIST(
    './data/FashionMNISTTest',
    download=True,
    train=False,
    transform=transforms.Compose([
        transforms.ToTensor()
    ])
)

network = Network()


train_loader = torch.utils.data.DataLoader(train_set, batch_size=100)
optimizer = optim.Adam(network.parameters(), lr=0.01)
num_epochs = 5
epoch = 0

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# network.to(device)

if os.path.isfile(SAVE_PATH):
    print("Loading model from checkpoint...")
    checkpoint = torch.load(SAVE_PATH)
    epoch = checkpoint['epoch']
    network.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


while epoch < num_epochs:
    total_loss = 0
    total_correct = 0

    for batch in train_loader:  # get batch
        images, labels = batch

        preds = network(images)  # Pass Batch
        loss = F.cross_entropy(preds, labels)  # Calculate loss

        optimizer.zero_grad()  # Clear the grads
        loss.backward()  # Calculate gradients
        optimizer.step()  # Update weights

        total_loss += loss.item()
        total_correct += get_num_correct(preds, labels)

    print("epoch:", epoch, "total_correct:",
          total_correct, "loss:", total_loss)
    print("percent: ", total_correct / len(train_set))
    epoch += 1

    torch.save({
        'epoch': epoch,
        'model_state_dict': network.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, SAVE_PATH)

print("Validating...")
test_data, test_labels = next(iter(test_set))
test_prediction = network(test_data.unsqueeze(dim=1))

names = {}
names[0] = 'T-shirt/top'
names[1] = 'Trouser'
names[2] = 'Pullover'
names[3] = 'Dress'
names[4] = 'Coat'
names[5] = 'Sandal'
names[6] = 'Shirt'
names[7] = 'Sneaker'
names[8] = 'Bag'
names[9] = 'Ankle boot'

plt.title('Guessing: ' +
          names[int(test_prediction.argmax(dim=1).squeeze())]
          + ' Actually: ' + names[test_labels]
          )

# plt.imshow(test_data.squeeze())
# plt.show()


def get_all_preds(model, loader):
    all_preds = torch.tensor([])
    for batch in loader:
        images, labels = batch

        preds = model(images)
        all_preds = torch.cat((all_preds, preds), dim=0)
    return all_preds


with torch.no_grad():
    prediction_loader = torch.utils.data.DataLoader(
        train_set, batch_size=10000)
    train_preds = get_all_preds(network, prediction_loader)

    preds_correct = get_num_correct(train_preds, train_set.targets)

print('total correct: ', preds_correct)
print('accuracy: ', preds_correct / len(train_set))

# Create confusion matrix by hand
stacked = torch.stack((
    train_set.targets,
    train_preds.argmax(dim=1)
), dim=1)

confusion_matrix = torch.zeros(10, 10, dtype=torch.int32)

for prediction in stacked:
    j, k = prediction.tolist()
    confusion_matrix[j, k] = confusion_matrix[j, k] + 1

# print(confusion_matrix)

names = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
         'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')
plt.figure(figsize=(10, 10))
plot_confusion_matrix(confusion_matrix, names)
plt.show()
