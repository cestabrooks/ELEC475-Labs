import model as m
import argparse
import torch
from torchvision import transforms

import os
from torch.utils.data import Dataset
from PIL import Image, ImageFile

class custom_ROI_testing_dataset(Dataset):
    def __init__(self, dir, num_images, num_grid_segments, transform=None):
        super().__init__()
        Image.MAX_IMAGE_PIXELS = None
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        self.transform = transform

        self.image_files = []
        for index, file_name in os.listdir(dir):
            if index == num_images:
                break
            if file_name.endswith(".png"):
                self.image_files.append(dir + file_name)

        self.labels = []
        label_file = open(dir + "/labels.txt", "r")
        for index, line in label_file:
            if index == num_images:
                break
            # classification
            self.labels.append(line.split(" ")[1])
            # The rest of the label contains the bounding box

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image = Image.open(self.image_files[index]).convert('RGB')
        image_sample = self.transform(image)
        # print('break 27: ', index, image, image_sample.shape)
        return image_sample, self.labels[index]

def evaluate(model, test_dataloader):
    print("Evaluating...")
    model.eval()
    with torch.no_grad():
        for imgs, labels in test_dataloader:
            output = model(imgs)

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

    # Subdivide a Kitti image into a set of ROIs. Save the bounding box coordinates for each ROI
    num_grid_segments = 128
    # CREATE A FUNCTION TO BREAK THE IMAGE INTO SEGMENTS AND THEIR RECORD THEIR BOUNDARY BOXES


    test_dataset = custom_ROI_testing_dataset("../data/OUR CUSTOM FOLDER", 10, num_grid_segments, transform)
    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        shuffle=False,
        batch_size=num_grid_segments
    )

    evaluate(model, test_dataloader)


