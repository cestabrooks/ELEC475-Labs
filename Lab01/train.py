
import sys

import torch
import torch.nn as nn
import model as m
import datetime
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data.dataloader as dataloader
from torchsummary import summary

import matplotlib.pyplot as plt

def train(n_epochs, optimizer, model, loss_fn, train_loader, scheduler, device):
    print('training...')
    model.train()
    losses_train = []

    # send model for use on M1 gpu AS WELL as imgs
    model.to("mps")

    for epoch in range(1, n_epochs+1):
        print('epoch ', epoch)
        loss_train = 0.0
        count = 0
        for imgs, labelOfnumWrittenInImg in train_loader:
            count = count + 1
            if count > 10:
                break

            imgs = imgs.to(device=device)

            # # Flattening the images
            # imgs = imgs.view(imgs.size(0), -1)
            #iterate through each image and flatten it
            #imgs = <batch size> of (28x28)
            count = 0
            for img in imgs:
                print(img.size())
                imgs[count] = torch.flatten(img)
                print(img)
                count = count + 1
            # CANT JUST OVERRIDE OLD TENSOR NEED TO PREPROCESS BEFORE CALLING TRAIN
            # AND MAKE A NEW TENSOR WITH FLATTEN IMAGES 1X784

            outputs = model(imgs)
            loss = loss_fn(outputs, imgs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()

        scheduler.step(loss_train)

        losses_train += [loss_train/len(train_loader)]

        print('{} Epoch {}, Training loss {}'.format( datetime.datetime.now(), epoch, loss_train/len(train_loader)))

def getMNISTDataset():
    train_transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST('./data/mnist', train=True, download=True, transform=train_transform)

    return train_set

if __name__ == '__main__':
    # Get arguments from command line

    # arg module makes this easier?
    z = int(sys.argv[sys.argv.index("-z") + 1])  # bottleneck
    e = int(sys.argv[sys.argv.index("-e") + 1])  # epoch
    b = int(sys.argv[sys.argv.index("-b") + 1])  # batch size
    s = sys.argv[sys.argv.index("-s") + 1]  # generic file type for a parameter file
    p = sys.argv[sys.argv.index("-p") + 1]  # plot output

    train_set = getMNISTDataset()
    model = m.autoencoderMLP4Layer()

    # data loader
    train_loader = torch.utils.data.DataLoader(train_set, b, shuffle=False)

    scheduler = torch.optim.lr_scheduler.StepLR(torch.optim.Adam(model.parameters(), lr=0.001),  step_size=5, gamma=0.5)
    train(e, torch.optim.Adam(model.parameters(), lr=0.001), model, nn.MSELoss(), train_loader, scheduler, "mps")

    # args model, shape of input data
    summary(model, (1, 28, 28))