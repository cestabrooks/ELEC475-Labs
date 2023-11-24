import os
from torch.utils.data import Dataset
from PIL import Image, ImageFile
import json



class custom_dataset(Dataset):
    def __init__(self, dir, transform=None):
        super().__init__()
        Image.MAX_IMAGE_PIXELS = None
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        self.transform = transform
        self.labels = {}
        self.key_list = []
        self.dir = dir
        label_file = open(self.dir + "/labels.txt", "r")
        for line in label_file:
            split = line.split(" ")
            image_name, label = split[0], split[1]
            self.labels[image_name] = int(label)
            self.key_list.append(image_name)

    def __len__(self):
        return len(self.key_list)

    def __getitem__(self, index):
        # Get the image name
        image_name = self.key_list[index]
        # Open the image and transform it to a tensor
        image = Image.open(self.dir + "/" + image_name).convert('RGB')
        image_tensor = self.transform(image)

        # Return the image tensor and nose coordinates
        return image_tensor, self.labels[image_name]


class step4_custom_dataset(Dataset):
    def __init__(self, dir, transform=None):
        super().__init__()
        Image.MAX_IMAGE_PIXELS = None
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        self.transform = transform

        self.image_files = []
        self.boxes = []
        label_file = open(dir + "/boxes.txt", "r")
        for line in label_file:
            # Always going to be 48 boxes per image
            self.boxes.append(json.loads(line.split("*")[1]))
            self.image_files.append(dir + line.split("*")[0])

        self.kitti_boxes = []
        label_file = open(dir + "/Kitti_boxes.txt", "r")
        for line in label_file:
            self.kitti_boxes.append(json.loads(line.split("*")[1]))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image = Image.open(self.image_files[index]).convert('RGB')
        image_sample = self.transform(image)
        return image_sample, self.boxes[index]

    def get_kitti_boxes(self, image_idx):
        return self.kitti_boxes[image_idx]

    def get_box(self, index):
        return self.boxes[index]