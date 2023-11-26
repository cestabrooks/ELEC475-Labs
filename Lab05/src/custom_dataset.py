import os
from torch.utils.data import Dataset
from PIL import Image, ImageFile
import re
import torch

class CustomDataset(Dataset):
    def __init__(self, dir, transform=None, start_index=0, end_index=6000):
        super().__init__()
        Image.MAX_IMAGE_PIXELS = None
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        self.start_index = start_index
        self.end_index = end_index
        self.transform = transform
        self.coordinates = {}
        self.key_list = []
        self.dir = dir
        label_file = open(self.dir + "/train_noses.3.txt", "r")
        for line in label_file:
            split = line.split(",")
            image_name, coord_x, coord_y = split[0], split[1], split[2]
            x = int(re.search(r'\d+', coord_x).group())
            y = int(re.search(r'\d+', coord_y).group())
            self.coordinates[image_name] = (x, y)
            self.key_list.append(image_name)

        if self.end_index == -1:
            self.end_index = len(self.key_list)
    def __len__(self):
        return self.end_index - self.start_index

    def __getitem__(self, index):
        index = index + self.start_index
        # Get the image name
        image_name = self.key_list[index]
        # Open the image and transform it to a tensor
        image = Image.open(self.dir + "/images/" + image_name).convert('RGB')
        image_tensor = self.transform(image)
        # Return the image tensor and nose coordinates
        return image_tensor, torch.tensor(self.coordinates[image_name]), image.width, image.height

class TestingDataset(Dataset):
    def __init__(self, dir, transform=None, start_index=0, end_index=6000):
        super().__init__()
        Image.MAX_IMAGE_PIXELS = None
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        self.start_index = start_index
        self.end_index = end_index
        self.transform = transform
        self.coordinates = {}
        self.key_list = []
        self.dir = dir
        label_file = open(self.dir + "/train_noses.3.txt", "r")
        for line in label_file:
            split = line.split(",")
            image_name, coord_x, coord_y = split[0], split[1], split[2]
            x = int(re.search(r'\d+', coord_x).group())
            y = int(re.search(r'\d+', coord_y).group())
            self.coordinates[image_name] = (x, y)
            self.key_list.append(image_name)

        if self.end_index == -1:
            self.end_index = len(self.key_list)
    def __len__(self):
        return self.end_index - self.start_index

    def __getitem__(self, index):
        index = index + self.start_index
        # Get the image name
        image_name = self.dir + "/images/" + self.key_list[index]
        # Open the image and transform it to a tensor
        image = Image.open(image_name).convert('RGB')
        image_tensor = self.transform(image)
        # Return the image tensor and nose coordinates
        return image_tensor, torch.tensor(self.coordinates[self.key_list[index]]), image.width, image.height, image_name
