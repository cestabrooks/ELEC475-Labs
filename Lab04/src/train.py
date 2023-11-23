import model as m
from torchvision import transforms
import torch.utils.data
import matplotlib.pyplot as plt
import argparse
import datetime
import custom_dataset
import os
from PIL import Image



def train(model, optimizer, n_epochs, loss_fn, data_loader, validation_loader, device, plot_file, save_file, completed_epochs=0):
    print("Starting training...")
    model.to(device=device)
    train_losses = []
    validation_losses = []
    train_accuracy = []
    validation_accuracy = []
    for epoch in range(completed_epochs, n_epochs):
        print("Epoch", epoch)
        model.train()
        loss_train = 0.0
        loss_val = 0.0
        acc_train = 0.0
        acc_val = 0.0
        for imgs, labels in data_loader:
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)

            # forward propagation
            outputs = model(imgs)
            # calculate loss
            labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=2)
            loss = loss_fn(outputs, labels_one_hot.float())
            # calculate accuracy
            predictions = torch.argmax(outputs, dim=1)
            acc_train += (predictions == labels).sum().item()/len(labels) * 100


            # reset optimizer gradients to zero
            optimizer.zero_grad()
            # calculate the loss gradients
            loss.backward()
            # iterate the optimization, based on the loss gradients
            optimizer.step()  # iterate the optimization, based on the loss gradients
            loss_train += loss.item()  # update the value of losses

        acc_train = acc_train/len(data_loader)
        train_accuracy.append(acc_train)

        # Calculate validation loss
        model.eval()
        with torch.no_grad():
            for imgs, labels in validation_loader:
                imgs = imgs.to(device=device)
                labels = labels.to(device=device)
                # forward propagation
                outputs = model(imgs)
                # calculate loss
                labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=2)
                loss = loss_fn(outputs, labels_one_hot.float())
                loss_val += loss.item()
                # calculate accuracy
                predictions = torch.argmax(outputs, dim=1)
                acc_val += (predictions == labels).sum().item() / len(labels) * 100

        acc_val = acc_val/len(validation_loader)
        validation_accuracy.append(acc_val)

        print('{} Epoch {}, Training loss {}, Validation Loss {}, Training Accuracy {}, Validation Accuracy {}'.format(datetime.datetime.now(),
                                                                                                                       epoch,
                                                                                                                       loss_train / len(data_loader),
                                                                                                                       loss_val / len(validation_loader),
                                                                                                                       acc_train,
                                                                                                                       acc_val))

        train_losses += [loss_train / len(data_loader)]  # update value of losses
        validation_losses += [loss_val / len(validation_loader)]

        # save the model and optimizer
        path = "./history/model_" + str(epoch) + ".pth"
        opt_path = "./history/optimizer_" + str(epoch) + ".pth"
        print("Saving model to: " + path)
        print("Saving optimizer to: " + opt_path)
        torch.save(model.state_dict(), path)
        torch.save(optimizer.state_dict(), opt_path)

        loss_file = open("./history/losses.txt", "a+")
        loss_file.write(str(train_losses[-1]) + " " + str(validation_losses[-1]) + "\n")
        loss_file.close()


    # plot the losses_train
    if save_file != None:
        torch.save(model.state_dict(), save_file)
    if plot_file != None:
        plt.figure(2, figsize=(12, 7))
        plt.clf()
        plt.plot(train_losses, label='train')
        plt.plot(validation_losses, label='validation')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(loc=1)
        plt.savefig(plot_file)


if __name__ == "__main__":
    # Get arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("-save")
    parser.add_argument("-e")
    parser.add_argument("-b")
    parser.add_argument("-p")
    parser.add_argument("-cuda")
    parser.add_argument("-mps")

    args = vars(parser.parse_args())

    save_pth = args["save"]
    n_epoch = int(args["e"])
    batch_size = int(args["b"])
    loss_plot = args["p"]
    device = "cpu"
    if str(args["cuda"]).upper() == "Y":
        device = "cuda"
    elif str(args["mps"]).upper() == "Y":
        device = "mps"


    model = m.efficientnet_b1
    model.to(device=device) # Model needs to be on correct device before we pass in its parameters to the optimizer.

    # Loading previous model state
    completed_epochs = 0
    if "_" in save_pth:
        print("Loading model: " + save_pth)
        # Load the decoder weights
        model.load_state_dict(torch.load(save_pth))
        # Subtract from the required number of epochs remaining
        completed_epochs = int(save_pth.split("_")[1].split('.')[0]) + 1

    transform = transforms.Compose([
        transforms.Resize(size=(150, 150), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5),)
    ])

    train_dataset = custom_dataset.custom_dataset("../data/Balanced_Kitti8_ROIs/train/", transform)
    print("Dataset size: ", len(train_dataset))
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    validation_dataset = custom_dataset.custom_dataset("../data/Kitti8_ROIs/test/", transform)
    validation_dataloader = torch.utils.data.DataLoader(
        dataset=validation_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)

    # Load previous optimizer and ???????????
    loss_file = None
    os.makedirs("./history/", exist_ok=True)
    if "_" in save_pth:
        print("Loading optimizer...")
        # Load the optimizer
        optimizer.load_state_dict(torch.load("./history/optimizer_" + str(completed_epochs - 1) + ".pth", map_location=device))
        learning_rate = optimizer.param_groups[0]['lr']
        print("   Resume learning rate at", optimizer.param_groups[0]['lr'])
    else:
        # Open existing for "writing" in order to clear it (since we are not resuming)
        loss_file = open("./history/losses.txt", "w")
        loss_file.close()

    train(model, optimizer, n_epoch, torch.nn.BCEWithLogitsLoss(), train_dataloader, validation_dataloader, device, loss_plot, save_pth, completed_epochs)