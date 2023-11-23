import dsntnn
import torch
import matplotlib.pyplot as plt
import argparse
import datetime
import model as m
from torchvision import transforms
from custom_dataset import CustomDataset
from PIL import Image

def train(model, optimizer, n_epochs, data_loader, validation_loader, device, plot_file, save_file):
    print("Starting training...")
    model.to(device=device)
    train_losses = []
    validation_losses = []
    train_accuracy = []
    validation_accuracy = []
    for epoch in range(0, n_epochs):
        print("Epoch", epoch)

        model.train()
        loss_train = 0.0
        loss_val = 0.0
        acc_train = 0.0
        acc_val = 0.0
        for imgs, target_coords in data_loader:
            imgs = imgs.to(device=device)
            target_coords = target_coords.to(device=device)
            # Reshape the target coords to expected shape of output (add extra dimension for number of coordinate positions)
            target_coords = torch.reshape(target_coords, (len(imgs), 1, 2))
            # forward propagation
            coords, heatmaps = model(imgs)

            # --- calculate losses ---
            euclidean_loss = dsntnn.euclidean_losses(coords, target_coords)
            # calculates divergence between heatmaps and spheretical Gaussians with std=sigma_t and mean=target_coords
            reg_loss = dsntnn.js_reg_losses(heatmaps, target_coords, sigma_t=1.0)
            # takes the mean of the losses across all locations (in our case, we only have one location, so not huge deal)
            loss = dsntnn.average_loss(euclidean_loss + reg_loss)
            # ------------------------

            # reset optimizer gradients to zero
            optimizer.zero_grad()
            # calculate the loss gradients
            loss.backward()

            # iterate the optimizer based on the loss gradients and update the model parameters
            optimizer.step()

            loss_train += loss.item()  # update the value of losses

        # Calculate validation loss
        model.eval()
        with torch.no_grad():
            for imgs, target_coords in validation_loader:
                imgs = imgs.to(device=device)
                target_coords = target_coords.to(device=device)
                # Reshape the target coords to expected shape of output (add extra dimension for number of coordinate positions)
                target_coords = torch.reshape(target_coords, (len(imgs), 1, 2))
                # forward propagation
                coords, heatmaps = model(imgs)
                # --- calculate losses ---
                euclidean_loss = dsntnn.euclidean_losses(coords, target_coords)
                # calculates divergence between heatmaps and spheretical Gaussians with std=sigma_t and mean=target_coords
                reg_loss = dsntnn.js_reg_losses(heatmaps, target_coords, sigma_t=1.0)
                # takes the mean of the losses across all locations (in our case, we only have one location, so not huge deal)
                loss = dsntnn.average_loss(euclidean_loss + reg_loss)
                # ------------------------

                loss_val += loss.item()  # update the value of losses

        print('{} Epoch {}, Training loss {}, Validation Loss {}'.format(datetime.datetime.now(),
                epoch, loss_train / len(data_loader), loss_val / len(validation_loader)))

        train_losses += [loss_train / len(data_loader)]  # update value of losses
        validation_losses += [loss_val / len(validation_loader)]

        # # save the model and optimizer
        # path = "./history/model_" + str(epoch) + ".pth"
        # opt_path = "./history/optimizer_" + str(epoch) + ".pth"
        # print("Saving model to: " + path)
        # print("Saving optimizer to: " + opt_path)
        # torch.save(model.state_dict(), path)
        # torch.save(optimizer.state_dict(), opt_path)
        #
        # loss_file = open("./history/losses.txt", "a+")
        # loss_file.write(str(train_losses[-1]) + " " + str(validation_losses[-1]) + "\n")
        # loss_file.close()


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

    model = m.CoordinateRegression()

    transform = transforms.Compose([
        transforms.Resize(size=(256, 256), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5),)
    ])

    train_dataset = CustomDataset("../data", transform, 0, 4800)
    val_dataset = CustomDataset("../data", transform, 4800, 5000)
    # test_dataset = CustomDataset("../data", transform, 5000, -1)

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    validation_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    optimizer = torch.optim.RMSprop(model.parameters(), lr=2.5e-4)

    train(model, optimizer, n_epoch, train_dataloader, validation_dataloader, device, loss_plot, save_pth)
