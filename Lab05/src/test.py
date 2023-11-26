import argparse

from PIL import Image
from torchvision import transforms
import dsntnn
import model as m
import torch
from custom_dataset import TestingDataset
import cv2
import os

scale = 1
def show_predicted_location(x, y, image_name):

    if os.path.isfile(image_name[0]):
        image = cv2.imread(image_name[0])
        nose = (int(x.item()), int(y.item()))
        dim = (int(image.shape[1] / scale), int(image.shape[0] / scale))
        imageScaled = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        cv2.circle(imageScaled, nose, 2, (0, 0, 255), 1)
        cv2.circle(imageScaled, nose, 8, (0, 255, 0), 1)
        cv2.imshow(image_name[0], imageScaled)
        key = cv2.waitKey(0)
        cv2.destroyWindow(image_name[0])
        if key == ord('q'):
            exit(0)

def evaluate(model, test_dataloader, show_images, device="cpu"):
    print("Evaluating...")
    model.to(device=device)
    model.eval()
    avg_distance_test = 0.0
    with torch.no_grad():
        for imgs, target_coords, width, height, image_name in test_dataloader:
            imgs = imgs.to(device=device)
            target_coords = target_coords.to(device=device)
            target_coords = target_coords.to(dtype=torch.float32)
            width = width.to(device=device)
            height = height.to(device=device)

            # Reshape the target coords to expected shape of output (add extra dimension for number of coordinate positions)
            target_coords = torch.reshape(target_coords, (len(imgs), 1, 2))
            # normalize targets so top-left is (-1, -1) and bottom right is (1, 1)
            target_coords[:, 0, 0] = (2 * target_coords[:, 0, 0]) / width - 1
            target_coords[:, 0, 1] = (2 * target_coords[:, 0, 1]) / height - 1
            # forward propagation
            coords, heatmaps = model(imgs)
            # Calculate the average distance
            avg_distance = dsntnn.calculate_avg_distances(coords, target_coords, width, height)
            avg_distance_test += avg_distance.item()

            # Print the image and the predicted nose location
            if show_images:
                pred_x, pred_y = dsntnn.convert_to_image_location(coords[:, 0, 0], coords[:, 0, 1], width, height)
                show_predicted_location(pred_x, pred_y, image_name)

    print("Avg distance error:", avg_distance_test/len(test_dataloader))

if __name__ == "__main__":
    # Get arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("-model_pth")
    parser.add_argument("-display")
    parser.add_argument("-cuda")
    parser.add_argument("-mps")

    args = vars(parser.parse_args())

    model_pth = args["model_pth"]
    device = "cpu"
    if str(args["cuda"]).upper() == "Y":
        device = "cuda"
    elif str(args["mps"]).upper() == "Y":
        device = "mps"

    show_images = False
    if str(args["display"]).upper() == "Y":
        show_images = True

    model = m.CoordinateRegression()
    model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))

    transform = transforms.Compose([
        transforms.Resize(size=(256, 256), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), )
    ])

    test_dataset = TestingDataset("../data", transform=transform, start_index=5000, end_index=-1)

    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=True
    )

    evaluate(model, test_dataloader, show_images, device)