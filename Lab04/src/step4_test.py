# The step4_train module is responsible for subdividing the kitti images into a set of ROIs and saving the coordinates

# Divide the image into a number of grids
# In a text file, save the original ROIs and the grid ROIs

import argparse
import json

from preproc.KittiDataset import KittiDataset
from preproc.KittiAnchors import Anchors
import cv2
import os
import model as m
from torchvision import transforms
import torch
import custom_dataset

# Maybe doesn't need to be done
def createImageROIs(dataset, anchors, output_dir):
    print("Creating images...")
    i = 0
    for item in enumerate(dataset):
        image = item[1][0]
        label = item[1][1]
        car_class = dataset.class_label['Car']
        car_ROIs = dataset.strip_ROIs(class_ID=car_class, label_list=label)

        anchor_centers = anchors.calc_anchor_centers(image.shape, anchors.grid)
        images, boxes = anchors.get_anchor_ROIs(image, anchor_centers, anchors.shapes)
        # Save the images to a file
        for k in range(len(boxes)):
            filename = str(i) + '_' + str(k) + '.png'
            cv2.imwrite(os.path.join(output_dir, filename), images[k])
        # Append the boxes
        with open(os.path.join(output_dir, 'boxes.txt'), 'a') as f:
            for k in range(len(boxes)):
                filename = str(i) + '_' + str(k) + '.png'
                f.write(filename)
                f.write("," + json.dumps(boxes[k]))
                f.write('\n')
        f.close()
        # Append the ground truth boxes
        with open(os.path.join(output_dir, 'Kitti_boxes.txt'), 'a') as f:
            f.write(str(i) + ',' + json.dumps(car_ROIs) + '\n')
        f.close()

        i += 1

def calc_IoU(boxA, boxB):
    # print('break 209: ', boxA, boxB)
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0][1], boxB[0][1])
    yA = max(boxA[0][0], boxB[0][0])
    xB = min(boxA[1][1], boxB[1][1])
    yB = min(boxA[1][0], boxB[1][0])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[1][1] - boxA[0][1] + 1) * (boxA[1][0] - boxA[0][0] + 1)
    boxBArea = (boxB[1][1] - boxB[0][1] + 1) * (boxB[1][0] - boxB[0][0] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou
def evaluate(model, data_loader, show_image):
    print("Evaluating...")
    with torch.no_grad():
        index = 0
        avg_iou = 0
        total = 0
        for imgs, boxes in data_loader:
            outputs = model(imgs)
            boxes_with_cars = torch.argmax(outputs, dim=1)

            if show_image:
                image_idx = 6000 + index
                img_path = "../data/Kitti8/test/image/00" + str(image_idx) + ".png"
                image = cv2.imread(img_path, cv2.IMREAD_COLOR)
                for i in boxes_with_cars:
                    if boxes[i] == 1:
                        box = boxes[i]
                        cv2.rectangle(image, (box[0][0], box[0][1]), (box[1][0], box[1][1]), (0, 0, 255))

                cv2.imshow('image', image)
                key = cv2.waitKey(0)
                if key == ord('x'):
                    break

            # Calculate the IoU for all the boxes with cars
            ground_truth_boxes = data_loader.get_kitti_boxes(index)
            for i in boxes_with_cars:
                if boxes[i] == 1:
                    boxA = boxes[i]
                    for boxB in ground_truth_boxes:
                        # simpler calculation to see if the boxes intersect
                        iou = calc_IoU(boxA, boxB)
                        # Only add the iou's to the average for boxes that actually overlap
                        if iou > 0:
                            avg_iou += iou
                            total += 1

            index += 1

        avg_iou = avg_iou / total
        print("Avg IoU: ", avg_iou)

if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-images")
    argParser.add_argument("-output_dir")
    argParser.add_argument("-model_pth")
    argParser.add_argument("-show_image")

    args = argParser.parse_args()

    model_pth = args.model_pth
    input_dir = args.images
    output_dir = args.output_dir
    show_image = False
    if str(args.show_image).upper() == "Y":
        show_image = True

    model = m.efficientnet_b1
    model.load_state_dict(torch.load(model_pth))

    os.makedirs(output_dir, exist_ok=True)

    dataset = KittiDataset(input_dir, training=False)
    anchors = Anchors()
    createImageROIs(dataset, anchors, output_dir)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5),)
    ])

    test_dataset = custom_dataset.step4_custom_dataset("../data/Kitti8_ROIs/test/", transform)
    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        shuffle=False,
        batch_size=48
    )

    evaluate(model, test_dataloader, show_image)

    print("Done")
