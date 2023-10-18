import argparse
import AdaIN_net
from torchvision import transforms
from custom_dataset import custom_dataset
import torch.utils.data
import datetime
import matplotlib.pyplot as plt
from PIL import Image

learning_rate = 1e-4
learning_rate_decay = 5e-5
def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = learning_rate / (1.0 + learning_rate_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    # Get arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("-content_dir")
    parser.add_argument("-style_dir")
    parser.add_argument("-gamma")
    parser.add_argument("-e")
    parser.add_argument("-b")
    parser.add_argument("-l")
    parser.add_argument("-s")
    parser.add_argument("-p")
    parser.add_argument("-cuda")

    args = vars(parser.parse_args())

    gamma = float(args["gamma"])
    n_epoch = int(args["e"])
    n_batch = int(args["b"])
    encoder_pth = args["l"]
    decoder_pth = args["s"]
    loss_plot = args["p"]
    device = "cuda" if str(args["cuda"]).upper() == "Y" else "cpu"  # change to cpu

    print("Creating model and loading data")
    encoder_decoder = AdaIN_net.encoder_decoder()
    encoder_decoder.encoder.load_state_dict(torch.load(encoder_pth))

    # Loading previous decoder state
    completed_epochs = 0
    if "_" in decoder_pth:
        print("Loading decoder: " + decoder_pth)
        # Load the decoder weights
        encoder_decoder.decoder.load_state_dict(torch.load(decoder_pth))
        # Subtract from the required number of epochs remaining
        completed_epochs = int(decoder_pth.split("_")[1].split('.')[0]) + 1

    model = AdaIN_net.AdaIN_net(encoder_decoder.encoder, encoder_decoder.decoder)

    model.train()
    model.to(device)


    transform = transforms.Compose([
        transforms.Resize(size=(512,512), interpolation=Image.BICUBIC),
        transforms.RandomCrop(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    content_dataset = custom_dataset(args["content_dir"], transform)
    style_dataset = custom_dataset(args["style_dir"], transform)
    print("Dataset size: ", len(content_dataset))
    print("Batch size: ", int(len(content_dataset)/n_batch))

    content_dataloader = torch.utils.data.DataLoader(
        dataset=content_dataset,
        batch_size=int(len(content_dataset)/n_batch),
        shuffle=True
    )
    style_dataloader = torch.utils.data.DataLoader(
        dataset=style_dataset,
        batch_size=int(len(style_dataset)/n_batch),
        shuffle=True
    )

    optimizer = torch.optim.Adam(encoder_decoder.decoder.parameters(), lr=learning_rate)

    print("Starting training...")
    losses = []
    style_losses = []
    content_losses = []
    for epoch in range(completed_epochs, n_epoch):
        print('epoch ', epoch)
        adjust_learning_rate(optimizer, iteration_count=epoch)
        loss_train = 0.0
        style_loss_train = 0.0
        content_loss_train = 0.0
        for batch in range(1, n_batch):
            print("batch ", batch)
            content_images = next(iter(content_dataloader)).to(device)
            style_images = next(iter(style_dataloader)).to(device)

            loss_c, loss_s = model.forward(content_images, style_images, gamma)
            loss_c *= 1.0
            loss_s *= 3.0
            loss = loss_c + loss_s

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()
            content_loss_train += loss_c.item()
            style_loss_train += loss_s.item()

        losses += [loss_train / n_batch]
        style_losses += [style_loss_train / n_batch]
        content_losses += [content_loss_train / n_batch]

        print('{} Epoch {}, Training loss {}'.format(datetime.datetime.now(), epoch, loss_train / n_batch))

        # save the decoder
        path = "decoder_" + str(epoch) + ".pth"
        print("Saving decoder to: " + path)
        torch.save(model.decoder.state_dict(), path)

    # plot the losses_train
    if decoder_pth is not None:
        torch.save(model.decoder.state_dict(), decoder_pth)
    if loss_plot is not None:
        plt.figure(2, figsize=(12, 7))
        plt.clf()
        plt.plot(losses, label='content+style')
        plt.plot(content_losses, label='content')
        plt.plot(style_losses, label="style")
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(loc=1)
        plt.savefig(loss_plot)
    print("Done!")
