import model as m
from torchvision import transforms
import torch.utils.data
import matplotlib.pyplot as plt
import argparse
import datetime

#------------- MOVE TO DIFFERENT FILE? --------------------------------
import os
from torch.utils.data import Dataset
from PIL import Image, ImageFile

class custom_dataset(Dataset):
    def __init__(self, dir, transform=None):
        super().__init__()
        Image.MAX_IMAGE_PIXELS = None
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        self.transform = transform

        self.image_files = []
        for file_name in os.listdir(dir):
            if file_name.endswith(".png"):
                self.image_files.append(dir + file_name)

        self.labels = []
        label_file = open(dir + "/labels.txt", "r")
        for line in label_file:
            self.labels.append(line.split(" ")[1])


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image = Image.open(self.image_files[index]).convert('RGB')
        image_sample = self.transform(image)
        # print('break 27: ', index, image, image_sample.shape)
        return image_sample, self.labels[index]
# ---------------------------------------------------------------------

def train(model, optimizer, n_epochs, loss_fn, data_loader, validation_loader, device, plot_file, save_file):
    print("Starting training...")
    model.to(device)
    train_losses = []
    validation_losses = []
    train_accuracy = []
    validation_accuracy = []
    for epoch in range(1, n_epochs + 1):
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
            loss = loss_fn(outputs, labels)
            # calculate accuracy
            predictions = torch.argmax(outputs, dim=1)
            acc_train += (predictions == labels).sum().item()/len(labels) * 100

            # ??????????????????????????????????????????????????????????????????????????

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
                imgs.to(device=device)
                labels.to(device=device)
                # forward propagation
                outputs = model(imgs)
                # calculate loss
                loss = loss_fn(outputs, labels)
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


    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5),)
    ])

    train_dataset = custom_dataset("../data/Kitti8_ROIs/train", transform)
    print("Dataset size: ", len(train_dataset))
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    validation_dataset = custom_dataset("../data/Kitti8_ROIs/test", transform)
    validation_dataloader = torch.utils.data.DataLoader(
        dataset=validation_dataset,
        shuffle=False,
    )

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer_SGD = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)

    train(model, optimizer_SGD, n_epoch, torch.nn.CrossEntropyLoss(), train_dataloader, device, loss_plot, save_pth)