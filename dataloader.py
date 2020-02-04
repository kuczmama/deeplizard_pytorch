# https: // deeplizard.com/learn/video/mUueSPmcOBc

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import pdb

torch.set_printoptions(linewidth=120)

train_set = torchvision.datasets.FashionMNIST(
    root='./data/FashionMNIST',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor()
    ])
)

train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=10
)

# Show one image
# image, label = next(iter(train_set))
# plt.imshow(image.squeeze(), cmap='gray')
# plt.show()

batch = next(iter(train_loader))
images, labels = batch

grid = torchvision.utils.make_grid(images, nrow=10)
plt.figure(figsize=(15, 15))
plt.imshow(test_data.squeeze(), cmap='magma')
plt.title('TutorialKart')
plt.show()
