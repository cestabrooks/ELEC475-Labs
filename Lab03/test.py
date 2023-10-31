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
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            print(total)
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
    parser.add_argument("-classifier")

    args = vars(parser.parse_args())
    classifier_pth = args["classifier"]

    print("Creating model and loading data")
    encoder = model.encoder
    model = model.classifier(100, encoder)
    model.load_state_dict(torch.load(classifier_pth))

    test_set = getCIFAR_100_Dataset()
    data_loader = torch.utils.data.DataLoader(test_set, shuffle=False, batch_size=32)
    print("Dataset size: ", len(data_loader))

    evaluate(model, data_loader)
