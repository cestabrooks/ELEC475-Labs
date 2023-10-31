import argparse
import model
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
    with torch.no_grad():
        for image, label in data_loader:
            output = model.forward(image)
            # calculate top-1 error rate
            prediction = torch.argmax(output, dim=1)
            top_one += (prediction == label).sum().item()
            # calculate top-5 error rate
            top_five_output = torch.topk(output, 5)
            top_five += (label in top_five_output).sum().item()

    top_one_accuracy = top_one/len(data_loader)
    top_five_accuracy = top_five/len(data_loader)
    top_one_error = (1-top_one_accuracy) * 100
    top_five_error = (1-top_five_accuracy) * 100
    print("Top one error: ", top_one_error)
    print("Top five error: ", top_five_error)

if __name__ == "__main__":
    # Get arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("-classifier")

    args = vars(parser.parse_args())
    classifier_pth = args["classifier"]

    print("Creating model and loading data")
    encoder = model.encoder_vanilla
    model = model.classifier(100, encoder)
    model.load_state_dict(torch.load(classifier_pth))

    test_set = getCIFAR_100_Dataset()
    data_loader = torch.utils.data.DataLoader(test_set, shuffle=False)

    evaluate(model, data_loader)
