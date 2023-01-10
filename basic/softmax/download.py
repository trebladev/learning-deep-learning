from torchvision import transforms
import torch
import torchvision
from torch.utils import data
import os
os.system('proxyon')
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root="../data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root="../data", train=False, transform=trans, download=True)
print(len(mnist_test))
print(len(mnist_train))

