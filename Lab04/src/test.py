import model as m
import argparse
import torch
from torchvision import transforms
import custom_dataset
from torch.utils.data import Dataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def evaluate(model, test_dataloader):
    print("Evaluating...")
    model.eval()
    acc_test = 0
    total = 0
    y_pred = []
    y_true = []
    with torch.no_grad():
        for imgs, labels in test_dataloader:
            outputs = model(imgs)
            # calculate accuracy
            predictions = torch.argmax(outputs, dim=1)
            y_pred += torch.Tensor.tolist(predictions)
            y_true += torch.Tensor.tolist(labels)
            acc_test += (predictions == labels).sum().item()
            total += len(labels)

    test_accuracy = acc_test / total * 100
    print(test_accuracy)
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    # Get arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("-model_pth")

    args = vars(parser.parse_args())

    model_pth = args["model_pth"]

    model = m.efficientnet_b1
    model.load_state_dict(torch.load(model_pth))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5),)
    ])

    test_dataset = custom_dataset("../data/Kitti8_ROIs/test/", transform)
    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        shuffle=False
    )
    evaluate(model, test_dataloader)


