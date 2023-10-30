import argparse
import model
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchsummary import summary
import matplotlib.pyplot as plt
import datetime


def getCIFAR_10_Dataset():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.225, 0.225, 0.225])
    transform = transforms.Compose([transforms.ToTensor(), normalize])
    train_set = datasets.CIFAR10('./data/cifar_10', train=True, download=True, transform=transform)
    return train_set


def getCIFAR_100_Dataset():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.225, 0.225, 0.225])
    transform = transforms.Compose([transforms.ToTensor(), normalize])
    train_set = datasets.CIFAR100('./data/cifar_10', train=True, download=True, transform=transform)
    return train_set


def train(model, optimizer, n_epochs, loss_fn, data_laoder, device, plot_file, save_file):
    print("Starting training...")
    model.train()
    losses_train = []
    accuracy_list = []

    # send model for use on M1 gpu AS WELL as imgs
    model.to(device)

    for epoch in range(1, n_epochs + 1):
        print('epoch ', epoch)
        loss_train = 0.0
        accuracy = 0.0
        # iterate over each batch of data
        for imgs, labels in data_laoder:
            # move images and labels to the correct device
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)
            # forward propagation
            outputs = model(imgs)
            # calculate loss
            loss = loss_fn(outputs, labels)
            # calculate accuracy
            predictions = torch.argmax(outputs, dim=1)
            accuracy += ((predictions == labels).sum().item()/len(labels)) * 100
            # reset optimizer gradients to zero
            optimizer.zero_grad()
            # calculate the loss gradients
            loss.backward()
            # iterate the optimization, based on the loss gradients
            optimizer.step()  # iterate the optimization, based on the loss gradients
            loss_train += loss.item()  # update the value of losses

        losses_train += [loss_train / len(data_loader)]  # update value of losses
        accuracy_list += [accuracy / len(data_loader)]
        print('{} Epoch {}, Training loss {} Accuracy {}'.format(datetime.datetime.now(), epoch,
                                                                 loss_train / len(data_loader),
                                                                 accuracy / len(data_loader)))

    # plot the losses_train
    if save_file != None:
        torch.save(model.state_dict(), save_file)
    if plot_file != None:
        plt.figure(2, figsize=(12, 7))
        plt.clf()
        plt.plot(losses_train, label='train')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(loc=1)
        plt.savefig(plot_file)


if __name__ == "__main__":
    # Get arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("-e")
    parser.add_argument("-b")
    parser.add_argument("-encoder")
    parser.add_argument("-save")
    parser.add_argument("-plot")
    parser.add_argument("-cuda")
    parser.add_argument("-mps")

    args = vars(parser.parse_args())

    n_epoch = int(args["e"])
    batch_size = int(args["b"])
    encoder_pth = args["encoder"]
    save_pth = args["save"]
    loss_plot = args["plot"]
    device = "cpu"
    if str(args["cuda"]).upper() == "Y":
        device = "cuda"
    elif str(args["mps"]).upper() == "Y":
        device = "mps"

    print("Creating model and loading data")
    encoder = model.encoder
    encoder.load_state_dict(torch.load(encoder_pth))
    model = model.classifier(10, encoder)

    train_set = getCIFAR_10_Dataset()
    data_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    print("Dataset size: ", len(train_set))

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer_SGD = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

    train(model, optimizer_SGD, n_epoch, torch.nn.CrossEntropyLoss(), data_loader, device, loss_plot, save_pth)

    summary(model, (3, 32, 32), batch_size=batch_size, device=device)
