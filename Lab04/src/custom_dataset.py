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

        self.image_files = []
        for file_name in os.listdir(dir):
            if file_name.endswith(".png"):
                self.image_files.append(dir + file_name)

        self.labels = []
        label_file = open(dir + "/labels.txt", "r")
        for line in label_file:
            self.labels.append(line.split(" ")[1])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image = Image.open(self.image_files[index]).convert('RGB')
        image_sample = self.transform(image)
        # print('break 27: ', index, image, image_sample.shape)
        return image_sample, int(self.labels[index])


class step4_custom_dataset(Dataset):
    def __init__(self, dir, transform=None):
        super().__init__()
        Image.MAX_IMAGE_PIXELS = None
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        self.transform = transform

        self.image_files = []
        for file_name in os.listdir(dir):
            if file_name.endswith(".png"):
                self.image_files.append(dir + file_name)

        self.boxes = []
        label_file = open(dir + "/boxes.txt", "r")
        for line in label_file:
            # Always going to be 48 boxes per image
            self.boxes.append(json.loads(line.split("*")[1]))

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