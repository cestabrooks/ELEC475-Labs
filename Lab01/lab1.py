# ELEC475 - Lab 1

# from torch.distributions import transforms
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import argparse
import torch
import model as m
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def eval(model, loss_fn, loader, device):
    print("Evaluate...")
    model.eval()
    loss_fn = loss_fn
    losses = []
    display_image = True
    with torch.no_grad():
        for imgs, label in loader:
            imgs = imgs.to(device=device)
            # Flattening the images to size (2048, 1, 784)
            flattendImgs = torch.flatten(imgs, start_dim=2)  # flatten the images to the correct size
            # Cast to 32-bit float
            flattendImgs.type('torch.FloatTensor')
            # Ensure that the flattened images are on the correct device
            flattendImgs.to(device=device)

            y = model(flattendImgs)
            loss = loss_fn(flattendImgs, y)
            losses += [loss.item()]

            # display output
            f = plt.figure()
            f.add_subplot(1, 2, 1)
            correct_imgs = torch.squeeze(imgs)
            print("INPUT SIZE", correct_imgs.size())
            plt.imshow(correct_imgs, cmap='gray')
            f.add_subplot(1, 2, 2)
            # reformat output to a (28, 28) tensor
            unflatten = torch.nn.Unflatten(2, (28, 28))
            output = torch.squeeze(unflatten(y))
            print("OUTPUT SIZE", output.size())
            plt.imshow(output, cmap='gray')
            plt.show()
            # calculate accuracy


def getMNISTDataset():
    test_transform = transforms.Compose([transforms.ToTensor()])
    test_set = datasets.MNIST('./data/mnist', train=False, download=True, transform=test_transform)
    return test_set

def getMNISTDatasetWithNoise():

    class addNoise(object):
        def __init__(self, std):
            self.std = std

        def __call__(self, tensor):
            return tensor + torch.randn(tensor.size()) * self.std

    noise_transform = transforms.Compose(
        [transforms.ToTensor(),
         addNoise(0.2)
         ])
    noise_set = datasets.MNIST('./data/mnist', train=False, download=True, transform=noise_transform)
    return noise_set


def createNoise(test_loader):
    noise_set = torch.rand(test_loader.data.size()) * 0.00001
    return noise_set


# # Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Get arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("-l")
    args = vars(parser.parse_args())
    parameters_file = args["l"]
    print(parameters_file)
    test_set = getMNISTDataset()

    # Add noise
    noise_set = getMNISTDatasetWithNoise()

    # instantiate model
    model = m.autoencoderMLP4Layer()
    model.load_state_dict(torch.load(parameters_file))
    # create data loader
    test_loader = torch.utils.data.DataLoader(noise_set, shuffle=False)

    eval(model, nn.MSELoss(), test_loader, "cpu")
