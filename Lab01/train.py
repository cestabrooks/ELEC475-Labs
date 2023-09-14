import sys

import torch
import torch.nn as nn
import model as m
import datetime
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data.dataloader as dataloader
import argparse
from torchsummary import summary


import matplotlib.pyplot as plt

def train(n_epochs, optimizer, model, loss_fn, train_loader, scheduler, device):
    print('training...')
    model.train()
    losses_train = []

    # send model for use on M1 gpu AS WELL as imgs
    model.to(device)

    for epoch in range(1, n_epochs + 1):
        print('epoch ', epoch)
        loss_train = 0.0
        # iterate over each batch of data
        for imgs, labelOfnumWrittenInImg in train_loader:

            imgs = imgs.to(device=device)  # move images to the correct device
            # print("imgs size", imgs.size())

            # Flattening the images to size (2048, 1, 784)
            flattendImgs = torch.flatten(imgs, start_dim=2)  # flatten the images to the correct size
            # Ensure that the flattened images are on the correct device
            flattendImgs.to(device=device)
            # print the flatten tensors shape
            # print("Flatten tensor size: ", flattendImgs.size())

            outputs = model(flattendImgs)                   # forward propagation
            loss = loss_fn(outputs, flattendImgs)           # calculate loss
            optimizer.zero_grad()                           # reset optimizer gradients to zero
            loss.backward()                                 # calculate the loss gradients
            optimizer.step()                                # iterate the optimization, based on the loss gradients
            loss_train += loss.item()                       # update the value of losses

        scheduler.step(loss_train)                          # update optimization hyperparameters

        losses_train += [loss_train / len(train_loader)]    # update value of losses

        print('{} Epoch {}, Training loss {}'.format(datetime.datetime.now(), epoch, loss_train / len(train_loader)))


def getMNISTDataset():
    train_transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST('./data/mnist', train=True, download=True, transform=train_transform)

    return train_set


if __name__ == '__main__':
    # Get arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("-z")
    parser.add_argument("-e")
    parser.add_argument("-b")
    parser.add_argument("-s")
    parser.add_argument("-p")
    args = vars(parser.parse_args())

    bottleneck = args["z"]
    epoch = args["e"]
    batch = args["b"]
    parameterFile = args["s"]
    plotFilePath = args["p"]

    # get training set of data
    train_set = getMNISTDataset()
    # instantiate model
    model = m.autoencoderMLP4Layer()
    # create data loader
    train_loader = torch.utils.data.DataLoader(train_set, batch, shuffle=True)
    # create scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(torch.optim.Adam(model.parameters(), lr=0.001), step_size=5, gamma=0.5)
    # train the model
    train(epoch, torch.optim.Adam(model.parameters(), lr=0.001), model, nn.MSELoss(), train_loader, scheduler, "cpu")
    # summarize the output
    summary(model, (1, 784), batch_size=batch, device="cpu")
