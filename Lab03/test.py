import argparse
import model as m
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def getCIFAR_10_Dataset():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.225, 0.225, 0.225])
    transform = transforms.Compose([transforms.ToTensor(), normalize])
    test_set = datasets.CIFAR10('./data/cifar_10', train=False, download=True, transform=transform)
    return test_set


def getCIFAR_100_Dataset():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.225, 0.225, 0.225])
    transform = transforms.Compose([transforms.ToTensor(), normalize])
    test_set = datasets.CIFAR100('./data/cifar_100', train=False, download=True, transform=transform)
    return test_set


def evaluate(model, data_loader):
    print("Evaluating...")
    model.eval()
    top_one = 0
    top_five = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            # print(total)
            output = model.forward(images)

            # calculate top-1 error rate
            predictions = torch.argmax(output, dim=1)
            # calculate top-5 error rate
            top_five_output = torch.topk(output, 5, dim=1).indices

            for batch in range(0, len(images)):
                total += 1
                if labels[batch] == predictions[batch]:
                    top_one += 1
                if labels[batch] in top_five_output[batch]:
                    top_five += 1

    top_one_accuracy = top_one/total
    top_five_accuracy = top_five/total
    top_one_error = (1-top_one_accuracy) * 100
    top_five_error = (1-top_five_accuracy) * 100
    print("Top one error: ", top_one_error)
    print("Top five error: ", top_five_error)

if __name__ == "__main__":
    # Get arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("-mod")
    parser.add_argument("-nn")
    parser.add_argument("-cifar10")

    args = vars(parser.parse_args())

    nn_pth = args["nn"]
    mod = False
    if str(args["mod"]).upper() == "Y":
        mod = True
    num_classes = 100
    if str(args["cifar10"]).upper() == "Y":
        num_classes = 10

    print("Creating model and loading data")
    # encoder = m.encoder_mod
    # model = model.classifier(10, encoder)
    # model.load_state_dict(torch.load(nn_pth))

    encoder = None
    model = None
    if mod:
        encoder = m.encoder_mod
        model = m.classifier(num_classes, encoder, True)
        model.load_state_dict(torch.load(nn_pth))
    else:
        encoder = m.encoder_vanilla
        model = m.classifier(num_classes, encoder, True)
        model.load_state_dict(torch.load(nn_pth))

    test_set = None
    if num_classes == 10:
        test_set = getCIFAR_10_Dataset()
    else:
        test_set = getCIFAR_100_Dataset()
    data_loader = torch.utils.data.DataLoader(test_set, shuffle=False, batch_size=32)
    print("Dataset size: ", len(data_loader))

    evaluate(model, data_loader)
