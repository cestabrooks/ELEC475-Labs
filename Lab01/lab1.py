# ELEC475 - Lab 1

# from torch.distributions import transforms
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import argparse
import torch
import model as m
import bottleneck_model
import torch.nn as nn



def eval(model, loss_fn, loader, device):
    model.eval()
    loss_fn = loss_fn
    losses = []

    with torch.no_grad():
        count = 0
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

            plt.imshow(correct_imgs, cmap='gray')
            f.add_subplot(1, 2, 2)
            # reformat output to a (28, 28) tensor
            unflatten = torch.nn.Unflatten(2, (28, 28))
            output = torch.squeeze(unflatten(y))

            plt.imshow(output, cmap='gray')
            plt.show()
            # calculate accuracy

            # Only show first three input/output pairs
            if count >= 2:
                return

            count += 1

def eval_with_noise(model, loss_fn, noise_loader, test_loader, device):
    model.eval()
    loss_fn = loss_fn
    losses = []
    count = 0
    with torch.no_grad():
        # display output
        f = plt.figure()

        for noise_imgs, label in noise_loader:

            noise_imgs = noise_imgs.to(device=device)
            # Flattening the images to size (2048, 1, 784)
            flattendImgs = torch.flatten(noise_imgs, start_dim=2)  # flatten the images to the correct size
            # Cast to 32-bit float
            flattendImgs.type('torch.FloatTensor')
            # Ensure that the flattened images are on the correct device
            flattendImgs.to(device=device)

            y = model(flattendImgs)
            loss = loss_fn(flattendImgs, y)
            losses += [loss.item()]

            # display image without noise
            test_loader.dataset.data.type('torch.FloatTensor')
            no_noise_img = test_loader.dataset.data[count]
            f.add_subplot(3, 3, count*3 + 1)
            plt.imshow(no_noise_img, cmap='gray')

            # display image with noise
            f.add_subplot(3, 3, count*3 + 2)
            correct_imgs = torch.squeeze(noise_imgs)
            plt.imshow(correct_imgs, cmap='gray')

            #display output image
            f.add_subplot(3, 3, count*3 + 3)
            unflatten = torch.nn.Unflatten(2, (28, 28))
            output = torch.squeeze(unflatten(y))
            plt.imshow(output, cmap='gray')


            if count >= 2:
                plt.tight_layout(pad=1)
                plt.show()
                return

            count += 1

def eval_bottleneck_model(model, loader, device):
    model.eval()
    with torch.no_grad():
        for img_pair, label in loader:

            # print(img_pair[0])
            # print(img_pair[1])
            flattendImg1 = torch.flatten(img_pair[0], start_dim=1)  # flatten the images to the correct size
            flattendImg2 = torch.flatten(img_pair[1], start_dim=1)
            # Cast to 32-bit float
            flattendImg1.type('torch.FloatTensor')
            flattendImg2.type('torch.FloatTensor')
            # Ensure that the flattened images are on the correct device
            flattendImg1.to(device=device)
            flattendImg2.to(device=device)

            output = model.forward(flattendImg1, flattendImg2, 8)

            # display output
            f = plt.figure()
            for i in range(0, len(output)):
                f.add_subplot(1, len(output), i + 1)
                # reformat output to a (28, 28) tensor
                unflattened = output[i].reshape((28, 28))

                plt.imshow(unflattened, cmap='gray')

            plt.show()
            break

def getMNISTDataset():
    test_transform = transforms.Compose([transforms.ToTensor()])
    test_set = datasets.MNIST('./data/mnist', train=False, download=True, transform=test_transform)
    return test_set

def getMNISTDatasetWithNoise():

    class addNoise(object):
        def __init__(self, std):
            self.std = std

        def __call__(self, tensor):
            result = tensor + torch.randn(tensor.size()) * self.std
            # result = torch.clamp(result, 0, 1)
            return result

    noise_transform = transforms.Compose(
        [transforms.ToTensor(),
         addNoise(0.2)
         ])
    noise_set = datasets.MNIST('./data/mnist', train=False, download=True, transform=noise_transform)
    return noise_set

# Press the green button in the gutter to run the script.
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
    test_loader = torch.utils.data.DataLoader(test_set, shuffle=False)
    noise_loader = torch.utils.data.DataLoader(noise_set, shuffle=False)
    bottleneck_loader = torch.utils.data.DataLoader(test_set, 2, shuffle=True)
    print("Step 4: Testing Autoencoder Output ------------------ ")
    print("This test will show three input/output pairs as done \n" +
          "in the Lab outline. Close the window to see the next \n" +
          "one. \n\n" +
          "Will continue to Step 5 after going through 3 windows.\n")
    eval(model, nn.MSELoss(), test_loader, "cpu")

    print("Step 5: Image Denoising Output ------------------ ")
    print("This test will show three samples of the original \n" +
          "image, the image with noise, and the output of the \n" +
          "autoencoder when using the noised image.\n\n" +
          "Will continue to Step 6 after closing the window.\n")
    eval_with_noise(model, nn.MSELoss(), noise_loader, test_loader, "cpu")

    print("Step 6: Bottleneck Interpolation ------------------ ")
    print("This test will show an example of a bottleneck tensor \n" +
          "being linearly interpolated between 2 input images for\n" +
          "8 steps. The bottlenecks are then decoded and the resulting\n"
          "images are shown.\n\n" +
          "Will exit the program after closing the window.\n")
    bn_model = bottleneck_model.autoencoderMLP4Layer_bottleneck()
    bn_model.load_state_dict(torch.load(parameters_file))
    eval_bottleneck_model(bn_model, bottleneck_loader, "cpu")

