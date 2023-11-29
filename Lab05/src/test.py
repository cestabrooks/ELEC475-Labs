import argparse

from PIL import Image
from torchvision import transforms
import dsntnn
import model as m
import torch
from custom_dataset import TestingDataset
import cv2
import os
import numpy as np

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
    distances = np.empty(len(test_dataloader), dtype=int)
    count = 0
    max_d = 0
    min_d = 1000
    max_image = {} # Will contain image name and pred coordinates
    min_image = {} # Will contain image name and pred coordinates

    with ((torch.no_grad())):
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

            # Append each distance to a list
            tensor_of_distances = dsntnn.get_list_of_distances(coords, target_coords, width, height)
            distances[count] = tensor_of_distances.item()
            count += 1

            if distances[count] < min_d:
                min_d = distances[count]
                pred_x, pred_y = dsntnn.convert_to_image_location(coords[:, 0, 0], coords[:, 0, 1], width, height)
                min_image["name"] = image_name
                min_image["coord"] = pred_x, pred_y
            elif distances[count] > max_d:
                max_d = distances[count]
                pred_x, pred_y = dsntnn.convert_to_image_location(coords[:, 0, 0], coords[:, 0, 1], width, height)
                max_image["name"] = image_name
                max_image["coord"] = pred_x, pred_y

            # Print the image and the predicted nose location
            if show_images:
                pred_x, pred_y = dsntnn.convert_to_image_location(coords[:, 0, 0], coords[:, 0, 1], width, height)
                show_predicted_location(pred_x, pred_y, image_name)


    mean_d = np.mean(distances)
    median_d = np.median(distances)

    # Target: 5 px eud on average. std=10.

    acc_3px = np.count_nonzero(distances <= 3)/len(distances)
    acc_5px = np.count_nonzero(distances <= 5 and distances > 3)/len(distances)
    acc_10px = np.count_nonzero(distances <= 10 and distances > 5)/len(distances)
    acc_15px = np.count_nonzero(distances <= 15 and distances > 10)/len(distances)
    acc_20px = np.count_nonzero(distances <= 20 and distances > 15)/len(distances)

    print("Avg distance error:", avg_distance_test/len(test_dataloader))

    print("Displaying the best image with a distance of:", min_d)
    x, y = min_image["coord"]
    show_predicted_location(x, y, min_image["name"])
    print("Displaying the worst image with a distance of:", max_d)
    x, y = max_image["coord"]
    show_predicted_location(x, y, max_image["name"])

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
        shuffle=False
    )

    evaluate(model, test_dataloader, show_images, device)