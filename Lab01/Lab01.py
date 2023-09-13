# ELEC475 - Lab 1

# from torch.distributions import transforms
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import torch as torch
import torch.nn as nn
import torch.nn.functional as F

def getMNISTDataset():
    train_transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST('./data/mnist', train=True, download=True, transform=train_transform)

    return train_set



# # Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     train_set = getMNISTDataset()
#
#     idx = 0
#     plt.imshow(train_set.data[idx], cmap='gray')
#     plt.show()



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
