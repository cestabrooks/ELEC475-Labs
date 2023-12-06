import argparse

from PIL import Image
from torchvision import transforms
import dsntnn
import model4 as m
import torch
from custom_dataset import TestingDataset
import cv2
import os
import numpy as np
import time
import matplotlib.pyplot as plt

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
    distances = np.empty(len(test_dataloader), dtype=float)
    eval_times = np.empty(len(test_dataloader), dtype=float)
    count = 0
    max_d = 0
    min_d = 1000
    max_image = {} # Will contain image name and pred coordinates
    min_image = {} # Will contain image name and pred coordinates

    with ((torch.no_grad())):
        for imgs, target_coords, width, height, image_name in test_dataloader:

            start = time.perf_counter()

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
            #
            end = time.perf_counter()
            eval_times[count] = (end - start) * 1000

            # Append each distance to a list
            tensor_of_distances = dsntnn.get_list_of_distances(coords, target_coords, width, height)
            distances[count] = tensor_of_distances.item()

            if distances[count] < min_d:
                min_d = distances[count]
                pred_x, pred_y = dsntnn.convert_to_image_location(coords[:, 0, 0], coords[:, 0, 1], width, height)
                min_image["name"] = image_name
                min_image["coord"] = pred_x, pred_y
                min_image['heatmap'] = heatmaps
            elif distances[count] > max_d:
                max_d = distances[count]
                pred_x, pred_y = dsntnn.convert_to_image_location(coords[:, 0, 0], coords[:, 0, 1], width, height)
                max_image["name"] = image_name
                max_image["coord"] = pred_x, pred_y
                max_image['heatmap'] = heatmaps

            count += 1
            # Print the image and the predicted nose location
            if show_images:
                # Show images
                pred_x, pred_y = dsntnn.convert_to_image_location(coords[:, 0, 0], coords[:, 0, 1], width, height)
                show_predicted_location(pred_x, pred_y, image_name)
                # Show heatmap
                plt.imshow(heatmaps[0, 0].detach().cpu().numpy())
                plt.show()

    # Target: 5 px eud on average. std=10.

    mean_d = np.mean(distances)
    std_d = np.std(distances)
    median_d = np.median(distances)

    percent_perfect = 100 * np.count_nonzero(distances < 1)/len(distances)

    acc_3px = 100 * np.count_nonzero(distances <= 3)/len(distances)
    acc_5px = 100 * (np.count_nonzero(distances <= 5)/len(distances))
    acc_10px = 100 * (np.count_nonzero(distances <= 10)/len(distances))
    acc_15px = 100 * (np.count_nonzero(distances <= 15)/len(distances))
    acc_20px = 100 * (np.count_nonzero(distances <= 20)/len(distances))
    acc_gt20px = 100 - acc_20px

    print("\nmean:", mean_d)
    print("standard deviation:", std_d)
    print("median:", median_d)

    print("\npercent that are perfect:", percent_perfect)

    print("\nacc_3px:", acc_3px)
    print("acc_5px:", acc_5px)
    print("acc_10px:", acc_10px)
    print("acc_15px:", acc_15px)
    print("acc_20px:", acc_20px)
    print("Precent not within 20px:", acc_gt20px)

    avg_eval_time = "{:.3f}".format(np.mean(eval_times))
    print("Average eval time in ms:", avg_eval_time)

    print("\nDisplaying the best image with a distance of:", min_d)
    x, y = min_image["coord"]
    show_predicted_location(x, y, min_image["name"])
    # Show heatmap
    plt.imshow(min_image['heatmap'][0, 0].detach().cpu().numpy())
    plt.show()
    print("Displaying the worst image with a distance of:", max_d)
    x, y = max_image["coord"]
    show_predicted_location(x, y, max_image["name"])
    # Show heatmap
    plt.imshow(max_image['heatmap'][0, 0].detach().cpu().numpy())
    plt.show()

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

    model = m.CoordinateRegression_b3()
    model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))

    transform = transforms.Compose([
        transforms.Resize(size=(256, 256), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), )
    ])

    test_dataset = TestingDataset("../data", transform=transform, start_index=5000, end_index=-1)

    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=1, #MUST KEEP BATCH SISE AT 1
        shuffle=False
    )

    evaluate(model, test_dataloader, show_images, device)